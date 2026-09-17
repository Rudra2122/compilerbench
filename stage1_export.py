import torch
import torch._dynamo.config as dynamo_config
from collections import Counter
from model import get_model_and_input

model, example_input = get_model_and_input()

# --- torch.export: get the ExportedProgram ---
exported_program = torch.export.export(model, (example_input,))

with open("fx_graph.txt", "w") as f:
    f.write(str(exported_program.graph))

with open("aten_ir.txt", "w") as f:
    f.write(str(exported_program.graph_module.code))

aten_ops = [n.target for n in exported_program.graph.nodes if n.op == "call_function"]
print(f"[export] {len(exported_program.graph.nodes)} graph nodes, "
      f"{len(aten_ops)} ATen call_function ops")
print("[export] sample ATen ops:", aten_ops[:8])

dynamo_config.inline_inbuilt_nn_modules = True

seen_ops = []

def inspecting_backend(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.op == "call_function":
            seen_ops.append(str(node.target))
    return gm.forward

compiled_model = torch.compile(model, backend=inspecting_backend)
out = compiled_model(example_input)

with open("torch_compile_ops.txt", "w") as f:
    f.write("\n".join(seen_ops))

print(f"[compile] backend intercepted {len(seen_ops)} ops, output shape {out.shape}")
print("[compile] sample ops seen by custom backend:", seen_ops[:8])

# --- diagnostic: compare op-by-op counts between export and compile ---
print("\n--- export side (aten_ops) ---")
for op, count in Counter(str(op) for op in aten_ops).most_common():
    print(count, op)

print("\n--- compile side (seen_ops) ---")
for op, count in Counter(seen_ops).most_common():
    print(count, op)