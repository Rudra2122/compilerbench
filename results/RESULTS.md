### fp32 reference (MobileNetV2, Imagenette val, 1000-way head)

_Apple M3, 8 cores, Darwin 27.0.0; torch 2.14.0, onnxruntime 1.30.0, executorch 1.5.1_

| Path | Top-1 | Images | Max \|Δlogit\| vs PyTorch |
|---|---|---|---|
| PyTorch eager | 79.03% | 3925 | – |
| ONNX Runtime | 79.03% | 3925 | 5.5e-05 |

### Stage 3: int8 PTQ with ONNX Runtime (CPU)

_Apple M3, 8 cores, Darwin 27.0.0; torch 2.14.0, onnxruntime 1.30.0, executorch 1.5.1_

Calibration: 256 Imagenette-train images. Weights int8 symmetric, activations uint8 asymmetric, QDQ format. Latency is batch 1, p50 of 200 runs.

| Variant | Top-1 | Δ vs fp32 | Agrees w/ fp32 | p50 1 thread | Speedup | p50 all threads | Speedup | Size | Kernels ORT actually runs |
|---|---|---|---|---|---|---|---|---|---|
| `fp32` | 79.03% | +0.00 pp | 100.00% | 9.18 ms | 1.00x | 2.94 ms | 1.00x | 14.2 MB | 35 FusedConv, 17 Conv |
| `dynamic` | 77.50% | -1.53 pp | 89.71% | 12.30 ms | 0.75x | 10.19 ms | 0.29x | 3.7 MB | 52 ConvInteger, 52 DynamicQuantizeLinear, 1 DynamicQuantizeMatMul |
| `static_per_tensor_minmax` | 78.11% | -0.92 pp | 88.28% | 3.12 ms | 2.94x | 1.08 ms | 2.73x | 3.8 MB | 52 QLinearConv, 10 QLinearAdd, 1 QGemm, 2 QuantizeLinear, 2 DequantizeLinear |
| `static_per_channel_minmax` | 78.62% | -0.41 pp | 92.28% | 3.14 ms | 2.93x | 1.10 ms | 2.68x | 4.0 MB | 52 QLinearConv, 10 QLinearAdd, 1 QGemm, 2 QuantizeLinear, 2 DequantizeLinear |
| `static_per_channel_percentile` | 78.17% | -0.87 pp | 92.43% | 3.12 ms | 2.94x | 1.09 ms | 2.69x | 4.0 MB | 52 QLinearConv, 10 QLinearAdd, 1 QGemm, 2 QuantizeLinear, 2 DequantizeLinear |
| `static_per_channel_entropy` | 75.36% | -3.67 pp | 88.87% | 3.12 ms | 2.94x | 1.09 ms | 2.69x | 4.0 MB | 52 QLinearConv, 10 QLinearAdd, 1 QGemm, 2 QuantizeLinear, 2 DequantizeLinear |

**Why per-tensor hurts MobileNetV2 (measured on the BN-folded weights):** the median ratio between the largest and smallest per-output-channel max|w| is 32.9x in depthwise convs vs 5.1x in pointwise convs. Worst layer (`node_Conv_811`, depthwise): 2207x, so under one per-tensor scale its smallest channel gets 0.1 of 127 int8 levels, and 9 channels get fewer than 8.

### Stage 4a: ExecuTorch, portable kernels vs XNNPACK delegate (CPU)

_Apple M3, 8 cores, Darwin 27.0.0; torch 2.14.0, onnxruntime 1.30.0, executorch 1.5.1_

XNNPACK threadpool: 4 threads. Batch 1.

| Program | Delegated / fallback ops | Top-1 (n) | Agrees w/ fp32 | p50 | vs portable | .pte size | vs own PyTorch ref: max / mean \|Δlogit\|, argmax agree |
|---|---|---|---|---|---|---|---|
| `portable_fp32` | 0 / 205 (_native_batch_norm_legit_no_training×52, add_tensor×10, addmm×1, convolution×52, hardtanh×35, mean_dim×1, permute_copy×1, view_copy×1) | 75.00% (100) | 100.00% | 416.37 ms | 1.0x | 14.2 MB | 6.1e-05 / 2.7e-06, 100.0% |
| `xnnpack_fp32` | 204 / 1 (none) | 79.03% (3925) | 100.00% | 2.52 ms | 165.4x | 14.0 MB | 4.3e-05 / 4.0e-06, 100.0% |
| `xnnpack_int8` | 365 / 1 (none) | 78.85% (3925) | 90.34% | 1.17 ms | 356.5x | 3.8 MB | 4.2e+00 / 5.3e-01, 92.0% |

