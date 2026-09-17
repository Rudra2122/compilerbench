# CompilerBench: A PyTorch Model Through Every Compilation Path

## Overview

One small CNN, pushed through every major PyTorch compilation and deployment path: `torch.export`, `torch.compile`, ONNX, TVM's Relax IR, int8 quantization, ExecuTorch, and LiteRT. Every stage writes its intermediate output to disk (IR dumps, `.onnx`, `.pte`, `.tflite` files) and every claim below is checked against a reference output, not assumed. The goal was direct exposure to how a model actually moves through graph capture, lowering, operator fusion, and on-device export, not a summary of what the docs say those stages do.

The model itself is intentionally small (two conv blocks, two linear layers). Every stage needs to run in seconds on a laptop CPU so the pipeline stays reproducible without a GPU or target hardware.

## Pipeline

**Stage 1: torch.export and torch.compile.** The model is exported with `torch.export`, producing an `ExportedProgram` whose FX graph and ATen ops are dumped directly. The same model is separately run through `torch.compile` with a custom backend that intercepts the captured FX graph before returning it unmodified, so the ops seen by both compilation entry points can be compared directly.

**Stage 2a: ONNX export.** The model is exported to ONNX and run through ONNXRuntime, with output diffed against the original PyTorch model for correctness.

**Stage 2b: TVM Relax.** The ONNX graph is imported into TVM's current IR (Relax, the successor to the older Relay IR, which is not present in current TVM releases). `LegalizeOps`, `AnnotateTIROpPattern`, `FuseOps`, and `FuseTIR` are run in sequence, with the IR dumped before and after to confirm the fusion and shape-inference passes actually changed the graph rather than assuming they did.

**Stage 3: int8 quantization.** The ONNX model is quantized to int8 with ONNX Runtime's dynamic quantization tool, then benchmarked against the fp32 version for latency and output drift.

**Stage 4a: ExecuTorch.** The exported program is lowered through ExecuTorch's edge dialect and compiled to a `.pte` artifact, then loaded back into ExecuTorch's own runtime and checked against the PyTorch output.

**Stage 4b: LiteRT.** The ONNX model is converted to TFLite/LiteRT format via `onnx2tf`, verified independently against PyTorch output rather than relying only on the converter's own internal check.

## Results

| Stage | Metric | Value |
|---|---|---|
| torch.export | ATen ops | 14 |
| torch.compile (custom backend) | Ops intercepted | 12 (reconciled below) |
| ONNX + ONNXRuntime | Max abs diff vs PyTorch | ~1e-7 |
| TVM Relax | IR size before / after fusion passes | 30 lines / 224 lines |
| TVM Relax | Fusion evidence | `conv2d + add + relu` fused into a single `fused_conv2d1_add1_relu1` TIR function |
| int8 quantization | fp32 latency | 0.050 ms |
| int8 quantization | int8 latency | 0.341 ms (slower, see Limitations) |
| ExecuTorch | Artifact | `tinycnn.pte`, 556,088 bytes |
| ExecuTorch | Max abs diff vs PyTorch | 1.19e-07 |
| LiteRT | Artifact | `tinycnn_float32.tflite`, 550,448 bytes |
| LiteRT | Max abs diff vs PyTorch (independent check) | 8.94e-08 |

The TVM result is the strongest evidence in this project. Operator fusion is not stated as a claim, it is shown directly by diffing the IR before and after the pass runs, and the fused kernel is visible by name in the output.

## Op Count Reconciliation

`torch.export` (14 ops) and `torch.compile` (12 ops) do not report the same count for the same model, and the difference is fully explained rather than left as noise. Both traces agree exactly on conv2d, batch_norm, relu, max_pool2d, and linear. The 2-op gap is two `getitem` calls present only on the export side: `torch.export`'s decomposition lowers BatchNorm to `aten._native_batch_norm_legit_no_training`, which returns a 3-tuple, requiring a `getitem` to extract the output tensor. `torch.compile`'s trace keeps `batch_norm` as a single higher-level call and never produces that tuple, so no `getitem` is needed. Same computation, different internal bookkeeping at the two entry points.

## Engineering Notes

Four version-dependent issues were found and resolved during this project. Each is documented with the actual error, its root cause, and the fix applied, rather than summarized as "some issues came up."

### 1. Cross-stage comparisons were silently invalid due to unseeded weights

**Symptom:** PyTorch vs. LiteRT output comparison showed a large, unexplained divergence (max abs diff of 0.386), suggesting a broken conversion.

**Cause:** The model was instantiated fresh in every stage script with no fixed seed. Since each stage runs as a separate process, every script was creating a *different* randomly initialized model. The divergence was two unrelated random models being diffed against each other, not a conversion bug.

**Fix:** Seed with `torch.manual_seed(42)` and persist weights to a shared checkpoint file (`tinycnn_weights.pt`) that every stage loads, guaranteeing all comparisons run against the same model.

### 2. torch.compile's custom backend reported 1 op instead of ~12

**Symptom:** A backend hook counting `call_function` nodes reported only 1 intercepted op, versus 14 ATen ops from `torch.export` on the same model.

**Cause:** On torch 2.4.1, `torch.compile`'s default trace leaves each `nn.Module` call (`self.conv1(x)`, `self.bn1(x)`, etc.) as an opaque `call_module` node rather than inlining it into ATen-level `call_function` ops. A hook that only checks for `call_function` misses nearly the entire graph.

**Fix:** Set `torch._dynamo.config.inline_inbuilt_nn_modules = True` before calling `torch.compile`, forcing the same ATen-level inlining that `torch.export` already performs by default. Op count went from 1 to 12, correctly reconciled against the export side (see Op Count Reconciliation below).

### 3. ONNX export failed with "Unsupported FX nodes"

**Symptom:**
```
torch.onnx.OnnxExporterError: Failed to export the model to ONNX.
Unsupported FX nodes: {'call_function': ['aten.convolution.default',
'aten.max_pool2d_with_indices.default', ...]}
```

**Cause:** `torch.onnx.export(..., dynamo=True)` on torch 2.4.1 routes through `torch.onnx.dynamo_export`, an early-stage prototype exporter at that version that does not yet support ordinary ops like convolution and max pooling.

**Fix:** Use `dynamo=False`, the mature, long-standing TorchScript-based exporter. Exported the same model without modification and matched PyTorch output to ~1e-7.

### 4. ExecuTorch import chain failed across three separate, stacked causes

**Symptom (first):**
```
ImportError: cannot import name '_common_getitem_elimination_pass'
from 'torch.export.exported_program'
```

**Cause:** The current ExecuTorch release on PyPI requires `torch>=2.9.0`; the installed torch was `2.4.1`.

**Fix:** Installed `executorch==0.3.0`, the release pinned to `torch==2.4.0`, using `pip install --no-deps` to avoid a forced torch downgrade.

**Symptom (second):**
```
ImportError: cannot import name 'to_edge_transform_and_lower' from 'executorch.exir'
```

**Cause:** `executorch==0.3.0` predates the `to_edge_transform_and_lower` convenience API.

**Fix:** Used the lower-level `to_edge()` call available in that version instead.

**Symptom (third):**
```
ImportError: numpy.core.multiarray failed to import
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Cause:** ExecuTorch imports `pandas` internally, which imports `pyarrow`. Both were installed as binaries compiled against NumPy 1.x, while the environment had NumPy 2.2.6 installed, an ABI break unrelated to ExecuTorch's own code.

**Fix:** `pip install --force-reinstall --no-deps pandas pyarrow`, rebuilding both against the installed NumPy version.

## Repository Structure

```
compilerbench/
├── model.py                       # shared model definition, seeded checkpoint
├── stage1_export.py                # torch.export / torch.compile op comparison
├── stage2a_onnx.py                  # ONNX export + ONNXRuntime verification
├── stage2b_tvm.py                   # TVM Relax import, fusion, shape inference
├── stage3_quantize.py               # int8 quantization + latency/accuracy delta
├── stage4a_executorch.py            # ExecuTorch export + runtime verification
├── verify_litert.py                 # independent LiteRT correctness check
├── fx_graph.txt / aten_ir.txt        # Stage 1 IR dumps
├── tvm_ir_before.txt / tvm_ir_after.txt  # Stage 2b IR dumps
├── tinycnn.onnx / tinycnn_int8.onnx  # Stage 2a / 3 artifacts
├── tinycnn.pte                      # Stage 4a artifact
└── litert_out/                      # Stage 4b artifacts (.tflite)
```

## Reproduce

```bash
pip install torch onnx onnxscript onnxruntime apache-tvm onnx2tf tensorflow ai-edge-litert
pip install --no-deps executorch==0.3.0   # match to installed torch version, see Engineering Notes

python3 stage1_export.py
python3 stage2a_onnx.py
python3 stage2b_tvm.py
python3 stage3_quantize.py
python3 stage4a_executorch.py
python3 -m onnx2tf -i tinycnn.onnx -o litert_out -cotof
```

ExecuTorch pins an exact torch version per release. Check the installed torch version before installing ExecuTorch and match accordingly, or the import will fail (see Engineering Notes).

## Tech Stack

| Layer | Technology |
|---|---|
| Model / export | PyTorch, torch.export, torch.compile |
| Interchange format | ONNX, ONNXRuntime |
| Compiler | TVM (Relax IR) |
| Quantization | ONNX Runtime quantization toolkit |
| On-device runtimes | ExecuTorch, LiteRT (TFLite) |

## Limitations

Single small model, no autotuning or kernel-level tuning applied at any stage. int8 quantization is slower than fp32 on this model; dynamic quantization's dequantization overhead is not amortized by a model this small, and this result should not be read as quantization being ineffective in general. No NPU, DSP, or Qualcomm-specific runtime (QAIRT/QNN) is exercised anywhere in this pipeline. TVM and ExecuTorch results are CPU-only; no hardware-accelerated backend was targeted.

## Next Steps

Extend the TVM stage with a hardware-specific target (e.g. Hexagon DSP) instead of the generic `llvm` CPU target. Sweep model size to see whether int8 quantization's overhead crosses over to a net latency win, as it typically does on larger models. Push the ExecuTorch and LiteRT artifacts to an actual mobile or embedded target instead of validating on CPU only.

## Author

**Rudra Brahmbhatt**
MS Computer Science, Texas State University, May 2026
ML Infrastructure / Compiler Systems
