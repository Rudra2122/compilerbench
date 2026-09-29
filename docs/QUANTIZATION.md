# Quantization design notes

Why each quantization choice in CompilerBench v2 was made, what v1 got wrong, and what this project does not cover. Numbers referenced here are in the README results tables, which `report.py` generates from `results/*.json`.

## 1. Static vs dynamic, and why v1's choice was wrong for a CNN

**Dynamic quantization** stores weights as int8 offline but computes each activation's scale and zero-point *at runtime* from that tensor's min/max. It was designed for models whose cost is dominated by weight-heavy MatMuls and whose activation ranges vary a lot from input to input: LSTMs, Linear stacks, Transformers. You get the weight-memory win with no calibration step.

**For a CNN it is the wrong tool.** In ONNX Runtime, `quantize_dynamic` turns every `Conv` into:

```
DynamicQuantizeLinear (min/max pass over the whole activation) -> ConvInteger -> Cast -> Mul (rescale) -> Mul
```

Stage 3 records the op types ORT actually executes after its optimizer runs (`runtime_ops` in `results/stage3_quantize.json`). The dynamic model runs 52 `DynamicQuantizeLinear`, 52 `ConvInteger` and 104 `Mul`. So every conv pays for an extra reduction over its input plus a float rescale. On top of that, `ConvInteger` has a much less optimized CPU kernel than `QLinearConv`.

**Static quantization** runs a calibration set through the model once, offline, fixes every activation's scale and zero-point, and writes them into the graph as QDQ pairs. At load time the ORT CPU EP fuses `DQ -> Conv -> Q` into `QLinearConv`. For MobileNetV2 the runtime graph becomes 52 `QLinearConv` + 10 `QLinearAdd` + 1 `QGemm`, with exactly one Quantize at the input and one Dequantize at the output. The `Clip` (ReLU6) nodes disappear: once the calibrated output range sits inside [0, 6], the clamp is implemented by the uint8 saturation itself.

**Why v1 measured a 6.8x slowdown:** the ConvInteger path described above, plus a measurement problem. The fp32 TinyCNN took 0.05 ms, which is at the level of per-`session.run()` overhead, so the ratio was partly measuring Python/ORT call overhead rather than kernels. v2 uses a ~300 MFLOP model, reports p50/p90 over 200 runs, and separates 1-thread from all-thread latency.

## 2. Granularity and symmetry

| | Weights | Activations |
|---|---|---|
| ORT static (stage 3) | int8, **symmetric** (zero-point 0), per-tensor *or* **per-output-channel** | uint8, **asymmetric**, per-tensor |
| ExecuTorch XNNPACKQuantizer (stage 4a) | int8 symmetric [-127, 127], per-output-channel | int8 asymmetric, per-tensor |
| AI Hub quantize job (stage 5) | int8 (w8a8) | int8 |

**Why symmetric weights.** An int8 conv accumulates `Σ (q_w − z_w)(q_x − z_x)`. Expanded, that is `Σ q_w q_x − z_x Σ q_w − z_w Σ q_x + N z_w z_x`. The `z_x Σ q_w` term depends only on weights, so it is precomputed into the bias. The `z_w Σ q_x` term depends on the *input*, so it would cost an extra reduction on every inference. Forcing `z_w = 0` removes that term. It also matches what integer MAC hardware expects.

**Why asymmetric activations.** After ReLU6, activations are in [0, 6]. A symmetric int8 range would spend half its codes on negative values that never occur, which throws away one bit. An asymmetric uint8 range puts all 256 levels on [0, max].

**Why per-channel matters for MobileNetV2 specifically.** A per-tensor scale is set by the channel with the largest weight magnitude. Every other channel is quantized with that same step size, so a channel whose weights are 50x smaller uses only about 2–3 of the 127 positive int8 levels. MobileNetV2 is the textbook bad case. Its depthwise convs have one filter per channel with nothing averaging them out, and folding BatchNorm into those weights scales each channel by `γ/σ`, which spreads the per-channel ranges even further. Stage 3's `weight_ranges` block measures this on the actual BN-folded weights: per-channel range ratio for depthwise vs pointwise layers, and how many int8 levels the smallest channel gets under per-tensor quantization. Nagel et al., *Data-Free Quantization Through Weight Equalization and Bias Correction* (Qualcomm AI Research, ICCV 2019), used exactly this model to show naive per-tensor int8 PTQ collapsing. Their fixes, cross-layer equalization and bias correction, ship in AIMET. Per-channel weight quantization sidesteps the problem by giving each output channel its own scale. Compare the `static_per_tensor_minmax` and `static_per_channel_minmax` rows in the README for this repo's own measurement.

## 3. Calibration

- **Data:** 256 images, class-balanced, from Imagenette *train*. Evaluation uses Imagenette *val* only, so calibration and test never overlap.
- **Range estimators compared:** MinMax (the observed range, so outliers set the scale), Percentile (clip at the 99.999th percentile, trading a few clipped outliers for finer resolution everywhere else), and Entropy (pick the clipping threshold that minimizes KL divergence between the fp32 and quantized activation histograms).
- **Order matters:** BN is folded into the conv weights *before* quantization (the ONNX exporter constant-folds it; the XNNPACK lowering has a `FuseBatchNormPass`). Quantizing unfolded weights and then folding would compute scales on the wrong tensor.

## 4. How accuracy is measured

- **Top-1 on Imagenette val (3,925 images)** using the pretrained **1000-class** head unchanged. A prediction is correct only if the 1000-way argmax is the right ImageNet index. Imagenette's 10 classes are easy, so absolute top-1 is much higher than on full ImageNet. The number to read is the **delta vs fp32**.
- **Top-1 agreement with fp32**: how often the quantized model predicts the same class as fp32. This is more sensitive than accuracy, because a flip between two wrong classes still counts against it.
- **Logit drift** (max and mean |Δlogit|): v1 reported only this, which is output drift, not task accuracy.
- **Lowering correctness is checked separately from quantization error.** Each ExecuTorch `.pte` is compared against its *own* PyTorch reference on 300 eval images: eager fp32 for the fp32 programs, and the PT2E-converted quantized module for int8.
  - **fp32:** the difference should be ~1e-5 (float reassociation only).
  - **int8:** the logit difference is *not* ~0, and shouldn't be. The PyTorch reference *simulates* int8 (dequantize → fp32 conv → quantize), while XNNPACK runs true integer kernels with fixed-point requantization. The two round differently by up to 1 LSB per layer, and that compounds over 52 layers. The check that matters for int8 is argmax agreement between the `.pte` and its reference. A real lowering bug shows up as agreement collapsing, not as a few logits of drift.

## 5. Where it runs

| Stage | Runtime | Hardware |
|---|---|---|
| 3 | ONNX Runtime CPU EP (MLAS kernels) | laptop CPU |
| 4a | ExecuTorch portable kernels | laptop CPU, reference C++ with no SIMD tuning |
| 4a | ExecuTorch XNNPACK delegate | laptop CPU; the same backend ExecuTorch uses on Android/iOS CPU |
| 5 | QNN via Qualcomm AI Hub | hosted Snapdragon phone. The Hexagon NPU runs int8 natively; fp32 graphs run there in fp16 |

v1's ExecuTorch stage called `to_edge().to_executorch()` with no partitioner, so it ran entirely on portable kernels. Stage 4a now prints the delegation table: which op types XNNPACK took, and which fell back to portable.

## 6. PTQ vs QAT

**PTQ** (this project) quantizes a trained model using only a small unlabeled calibration set. It takes minutes and needs no training pipeline.

**QAT** inserts fake-quantize ops (quantize then immediately dequantize) into the training graph and fine-tunes, so the weights learn to tolerate rounding and clipping. Rounding has zero gradient almost everywhere, so the backward pass uses the straight-through estimator and treats rounding as identity. QAT needs labeled data, a training loop, and GPU time.

**When to move to QAT:** when the PTQ accuracy drop is unacceptable *after* the cheaper PTQ fixes have been tried. The usual escalation order is:

1. per-channel weights (this repo)
2. better range estimation: percentile, entropy or MSE (this repo)
3. cross-layer equalization + bias correction (AIMET; not in this repo)
4. AdaRound, i.e. learned rounding instead of round-to-nearest (AIMET; not in this repo)
5. mixed precision, e.g. int16 activations for the sensitive layers (AI Hub supports w8a16)
6. QAT

Low bit widths (int4 weights), small models with little redundancy, and depthwise-heavy architectures are where you reach step 6 soonest.

## 7. What this project does not do

- No QAT, CLE, bias correction or AdaRound. Per-channel + calibration method is the extent of the PTQ work.
- No per-layer sensitivity analysis (which single layer, kept in fp32, recovers the most accuracy).
- Imagenette rather than full ImageNet, so accuracy deltas are indicative, not a leaderboard number.
- In stage 5 the int8 model is produced by AI Hub's own quantize job (fed this repo's calibration set), not by ORT's quantizer. The on-device int8 row and the stage 3 rows are different quantizers.
- Laptop latency is wall-clock through Python bindings, batch 1. On-device latency is AI Hub's profiler number.
