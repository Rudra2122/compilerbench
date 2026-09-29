"""Stage 2a (v2): fp32 reference + ONNX export of MobileNetV2.

Produces the two things every later stage is measured against:
  * artifacts/fp32_ref_logits.npy : PyTorch fp32 logits on the full eval split
  * artifacts/mobilenet_v2.onnx    : fp32 ONNX, verified against those logits

Exported with the torch.export-based ONNX exporter (dynamo=True). On torch 2.4.1
this path failed on plain conv/maxpool (see v1 Engineering Note #3); on 2.14 it
works and constant-folds BatchNorm into the preceding Conv, so the graph has
52 Conv and zero BatchNorm nodes. That folding matters for quantization later:
BN must be folded *before* weights are quantized or the scale is computed on the
wrong tensor.
"""
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

import common


def main():
    model = common.get_model()
    x0 = torch.from_numpy(common.example_input())

    # ---- fp32 PyTorch reference on the whole eval split
    with torch.no_grad():
        ref, ref_logits = common.evaluate(lambda x: model(torch.from_numpy(x)).numpy())
    np.save(common.art("fp32_ref_logits.npy"), ref_logits)
    with torch.no_grad():
        lat_torch = common.bench(lambda x: model(x), x0)
    print(f"[fp32 torch] top1 {ref['top1']:.2f}% on {ref['n']} imgs, p50 {lat_torch['p50_ms']} ms")

    # ---- ONNX export
    path = common.art("mobilenet_v2.onnx")
    torch.onnx.export(model, (x0,), path, input_names=[common.INPUT_NAME],
                      output_names=["logits"], opset_version=18, dynamo=True,
                      external_data=False)
    g = onnx.load(path)
    ops = dict(Counter(n.op_type for n in g.graph.node))
    print(f"[onnx] ops: {ops}")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    run = lambda x: sess.run(None, {common.INPUT_NAME: x})[0]
    acc, _ = common.evaluate(run, ref_logits=ref_logits)
    print(f"[onnx fp32] top1 {acc['top1']:.2f}%  max|dlogit| vs torch {acc['logit_max_abs_diff']:.2e}")
    assert acc["logit_max_abs_diff"] < 1e-3, "ONNX export diverges from PyTorch"

    common.save_result("stage2a_onnx", {
        "model": "torchvision mobilenet_v2 IMAGENET1K_V1",
        "torch_fp32": {**ref, "latency": lat_torch},
        "onnx_fp32": {**acc, "ops": ops},
    })


if __name__ == "__main__":
    main()
