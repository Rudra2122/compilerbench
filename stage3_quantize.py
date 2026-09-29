"""Stage 3 (v2): int8 post-training quantization of MobileNetV2 with ONNX Runtime.

v1 used quantize_dynamic on a toy CNN and got a 6.8x *slowdown*. Two things were
wrong with that experiment, and this stage fixes both:

1. Wrong tool for the model. Dynamic quantization computes activation scales at
   runtime and is aimed at Linear/LSTM/Transformer-style models where activations
   vary a lot per input and compute is dominated by MatMul. For a CNN, ORT
   turns every Conv into ConvInteger with a DynamicQuantizeLinear in front of it:
   an extra min/max pass over every activation, plus ConvInteger, whose CPU
   kernel is much less optimized than QLinearConv. Static quantization computes
   activation scales once, offline, from calibration data, so the runtime graph
   is pure QLinearConv and nothing is recomputed per inference.
2. Too small to measure. 0.05 ms is at the level of per-call overhead. MobileNetV2
   at 224x224 is ~300 MFLOPs, so kernel time dominates.

Configs (all weights int8; activations uint8 unless dynamic):
  fp32                         reference
  dynamic                      v1's method, kept as the baseline to beat
  static_per_tensor_minmax     one scale per weight tensor
  static_per_channel_minmax    one scale per output channel (the standard CNN recipe)
  static_per_channel_percentile  same, but activation ranges clipped at the 99.999th pct
  static_per_channel_entropy   same, activation ranges chosen by KL divergence

Choices that are fixed across the static configs, and why:
  * QDQ format: the portable representation (what QNN / AI Hub / TensorRT ingest);
    ORT's CPU EP fuses DQ->Conv->Q into QLinearConv at load time.
  * Weights int8 symmetric (zero-point 0): no zero-point cross-term in the
    int32 accumulator, and it is what hardware int8 MAC arrays expect.
  * Activations uint8 asymmetric: post-ReLU6 activations are non-negative, so
    a symmetric int8 range would waste the whole negative half (1 bit).
  * Calibration: 256 class-balanced images from Imagenette *train*; eval is
    Imagenette *val*, so calibration never sees the test images.
"""
import argparse
import os
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.shape_inference import quant_pre_process

import common

CONFIGS = {
    "dynamic": dict(kind="dynamic"),
    "static_per_tensor_minmax": dict(kind="static", per_channel=False, calib=CalibrationMethod.MinMax),
    "static_per_channel_minmax": dict(kind="static", per_channel=True, calib=CalibrationMethod.MinMax),
    "static_per_channel_percentile": dict(kind="static", per_channel=True, calib=CalibrationMethod.Percentile),
    "static_per_channel_entropy": dict(kind="static", per_channel=True, calib=CalibrationMethod.Entropy),
}


class CalibReader(CalibrationDataReader):
    def __init__(self, n):
        self.it = common.iter_batches("calib", 1, n)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {common.INPUT_NAME: x[0]}


def weight_range_report(onnx_path, top=6):
    """Why per-tensor fails on MobileNetV2, measured on the actual (BN-folded) weights.

    For each Conv weight, compare per-output-channel max|w|. Under a single
    per-tensor scale s = max|W|/127, a channel whose own max is m only uses
    about 127*m/max|W| of the 127 positive int8 levels."""
    g = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in g.graph.initializer}
    rows = []
    for n in g.graph.node:
        if n.op_type != "Conv" or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        group = next((a.i for a in n.attribute if a.name == "group"), 1)
        ch_max = np.abs(w.reshape(w.shape[0], -1)).max(1)
        ch_max = ch_max[ch_max > 0]
        levels = 127 * ch_max / ch_max.max()
        rows.append({
            "node": n.name or n.output[0],
            "depthwise": group > 1 and group == w.shape[0],
            "channels": int(w.shape[0]),
            "range_ratio": float(ch_max.max() / ch_max.min()),
            "min_channel_int8_levels": float(levels.min()),
            "channels_under_8_levels": int((levels < 8).sum()),
        })
    rows.sort(key=lambda r: -r["range_ratio"])
    dw = [r for r in rows if r["depthwise"]]
    summary = {
        "conv_layers": len(rows),
        "depthwise_layers": len(dw),
        "median_range_ratio_depthwise": float(np.median([r["range_ratio"] for r in dw])) if dw else None,
        "median_range_ratio_pointwise": float(np.median([r["range_ratio"] for r in rows if not r["depthwise"]])),
        "worst_layers": rows[:top],
    }
    return summary


def entropy_ranges(pre_path, n_calib, cache_path, num_bins=2048, num_quantized_bins=128):
    """Run ORT's KL-divergence (entropy) calibrator with a usable histogram.

    ORT's EntropyCalibrater searches clipping thresholds from num_quantized_bins/2 up to
    num_bins/2 histogram bins, i.e. num_bins/2 - num_quantized_bins/2 + 1 candidates.
    quantize_static builds it with num_bins = num_quantized_bins = 128 and exposes no
    option to change that, which leaves exactly ONE candidate: the full range. So
    calibrate_method=Entropy silently equals MinMax (0 of 171 scales differed on
    MobileNetV2). We calibrate here with 2048 bins / 128 quantized bins, the setup from
    the TensorRT 8-bit calibration method ORT's implementation cites, then hand the
    ranges to quantize_static through its calibration cache."""
    import tempfile
    from onnxruntime.quantization.calibrate import CalibrationMethod as CM
    from onnxruntime.quantization.quantize import save_tensors_data

    def run(method, extra):
        with tempfile.TemporaryDirectory() as tmp:
            cal = create_calibrator(pre_path, None,
                                    augmented_model_path=os.path.join(tmp, "aug.onnx"),
                                    calibrate_method=method, extra_options=extra)
            cal.collect_data(CalibReader(n_calib))
            return cal.compute_data()

    ent = run(CM.Entropy, {"num_bins": num_bins, "num_quantized_bins": num_quantized_bins})
    # Safety check: entropy calibration should only ever *shrink* a range (clip outliers),
    # never extend it past what the calibration data actually produced. Clamp to the
    # observed min/max and report how many ranges needed it (expected: 0; ORT already
    # returns [0, t] for non-negative post-ReLU6 tensors).
    obs = run(CM.MinMax, {})
    clamped = 0
    for k in ent:
        lo_e, hi_e = ent[k].range_value
        lo_o, hi_o = obs[k].range_value
        lo, hi = np.maximum(lo_e, lo_o), np.minimum(hi_e, hi_o)
        if not (np.allclose(lo, lo_e) and np.allclose(hi, hi_e)):
            clamped += 1
        ent[k].lowest, ent[k].highest = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
    save_tensors_data(ent, cache_path)
    print(f"[entropy] {len(list(ent))} activation ranges, {clamped} clamped to observed min/max")
    return cache_path


def op_histogram(path):
    return dict(Counter(n.op_type for n in onnx.load(path).graph.node))


def runtime_ops(path, tag):
    """Op types ORT actually executes after its graph optimizer runs (QDQ fusion etc.).
    This is the evidence for which kernels run, not just what the file contains."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = common.art(f"ort_optimized_{tag}.onnx")
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return op_histogram(so.optimized_model_filepath)


def session(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--n-calib", type=int, default=common.N_CALIB_DEFAULT)
    ap.add_argument("--eval-limit", type=int, default=None, help="subset of val for quick runs")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    fp32 = common.art("mobilenet_v2.onnx")
    if not os.path.exists(fp32):
        raise SystemExit("run stage2a_onnx.py first")
    ref_logits = np.load(common.art("fp32_ref_logits.npy"))
    pre = common.art("mobilenet_v2_pre.onnx")
    quant_pre_process(fp32, pre)

    wr = weight_range_report(pre)
    print(f"[weights] depthwise median per-channel range ratio "
          f"{wr['median_range_ratio_depthwise']:.1f}x, pointwise "
          f"{wr['median_range_ratio_pointwise']:.1f}x")
    for r in wr["worst_layers"][:3]:
        print(f"          {r['node']}: {r['range_ratio']:.0f}x, smallest channel gets "
              f"{r['min_channel_int8_levels']:.1f} int8 levels under per-tensor")

    x0 = common.example_input()
    variants = {"fp32": fp32}
    for name in args.configs:
        cfg, out = CONFIGS[name], common.art(f"mobilenet_v2_{name}.onnx")
        print(f"[quant] {name}")
        if cfg["kind"] == "dynamic":
            quantize_dynamic(pre, out, weight_type=QuantType.QInt8)
        else:
            cache = None
            if cfg["calib"] == CalibrationMethod.Entropy:
                cache = common.art("entropy_ranges.json")
                if os.path.exists(cache):
                    os.remove(cache)
                entropy_ranges(pre, args.n_calib, cache)
            quantize_static(
                pre, out, None if cache else CalibReader(args.n_calib),
                calibration_cache_path=cache,
                quant_format=QuantFormat.QDQ,
                per_channel=cfg["per_channel"],
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=cfg["calib"],
                extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
            )
        variants[name] = out

    rows = {}
    for name, path in variants.items():
        s1 = session(path, 1)
        sN = session(path, None)
        acc, _ = common.evaluate(lambda x: s1.run(None, {common.INPUT_NAME: x})[0],
                                 limit=args.eval_limit, ref_logits=ref_logits)
        feed = lambda s: (lambda x: s.run(None, {common.INPUT_NAME: x}))
        rows[name] = {
            **acc,
            "size_bytes": os.path.getsize(path),
            "latency_1thread": common.bench(feed(s1), x0, iters=args.iters),
            "latency_allthreads": common.bench(feed(sN), x0, iters=args.iters),
            "file_ops": op_histogram(path),
            "runtime_ops": runtime_ops(path, name),
        }
        r = rows[name]
        print(f"  {name:32s} top1 {r['top1']:6.2f}%  agree {r.get('top1_agreement_vs_fp32', 100):6.2f}%  "
              f"1T p50 {r['latency_1thread']['p50_ms']:7.3f} ms  "
              f"NT p50 {r['latency_allthreads']['p50_ms']:7.3f} ms  "
              f"{r['size_bytes'] / 1e6:5.2f} MB")

    base = rows["fp32"]
    for name, r in rows.items():
        r["top1_delta_pp"] = round(r["top1"] - base["top1"], 3)
        r["speedup_1thread"] = round(base["latency_1thread"]["p50_ms"] / r["latency_1thread"]["p50_ms"], 3)
        r["speedup_allthreads"] = round(base["latency_allthreads"]["p50_ms"] / r["latency_allthreads"]["p50_ms"], 3)

    common.save_result("stage3_quantize", {
        "n_calib": args.n_calib,
        "weight_ranges": wr,
        "variants": rows,
    })


if __name__ == "__main__":
    main()
