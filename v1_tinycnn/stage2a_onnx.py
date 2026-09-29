import torch, numpy as np, onnxruntime as ort
from model import get_model_and_input

model, example_input = get_model_and_input()

torch.onnx.export(
    model,
    (example_input,),
    "tinycnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamo=False,
)

with torch.no_grad():
    torch_out = model(example_input).numpy()

sess = ort.InferenceSession("tinycnn.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"input": example_input.numpy()})[0]

max_diff = np.abs(torch_out - ort_out).max()
print(f"max abs diff: {max_diff:.2e}")
assert max_diff < 1e-4
print("PASS")