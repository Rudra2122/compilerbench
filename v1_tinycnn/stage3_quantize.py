import time, numpy as np, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("tinycnn.onnx", "tinycnn_preprocessed.onnx")
quantize_dynamic("tinycnn_preprocessed.onnx", "tinycnn_int8.onnx", weight_type=QuantType.QInt8)

def bench(path, n=200):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 32, 32).astype("float32")
    for _ in range(10): sess.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(n): out = sess.run(None, {"input": x})[0]
    return (time.perf_counter() - t0) / n * 1000, out

fp32_ms, fp32_out = bench("tinycnn.onnx")
int8_ms, int8_out = bench("tinycnn_int8.onnx")
print(f"fp32 {fp32_ms:.3f}ms, int8 {int8_ms:.3f}ms")
print(f"max diff: {np.abs(fp32_out-int8_out).max():.4f}")