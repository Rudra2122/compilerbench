import torch
from executorch.exir import to_edge
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
from model import get_model_and_input

model, example_input = get_model_and_input()

exported_program = torch.export.export(model, (example_input,))
edge_program = to_edge(exported_program)          
executorch_program = edge_program.to_executorch()

with open("tinycnn.pte", "wb") as f:
    f.write(executorch_program.buffer)

print(f"[executorch] wrote tinycnn.pte, {len(executorch_program.buffer)} bytes")

et_module = _load_for_executorch_from_buffer(executorch_program.buffer)
et_out = et_module.run_method("forward", (example_input,))[0]

with torch.no_grad():
    torch_out = model(example_input)

max_diff = (torch_out - et_out).abs().max().item()
print(f"[executorch] max abs diff vs PyTorch: {max_diff:.2e}")