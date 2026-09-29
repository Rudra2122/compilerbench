from ai_edge_litert.interpreter import Interpreter
import numpy as np, torch
from model import get_model_and_input

interp = Interpreter(model_path="litert_out/tinycnn_float32.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details(), interp.get_output_details()

model, example_input = get_model_and_input()
x_nhwc = np.ascontiguousarray(np.transpose(example_input.numpy(), (0,2,3,1)))
interp.set_tensor(inp[0]["index"], x_nhwc.astype(np.float32))
interp.invoke()
lite_out = interp.get_tensor(out[0]["index"])

with torch.no_grad():
    torch_out = model(example_input).numpy()
print("max abs diff:", np.abs(lite_out - torch_out).max())