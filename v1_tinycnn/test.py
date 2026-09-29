import torch
from model import get_model_and_input

model, example_input = get_model_and_input()

def inspecting_backend(gm, example_inputs):
    for node in gm.graph.nodes:
        print(node.op, "->", node.target)
    return gm.forward

compiled_model = torch.compile(model, backend=inspecting_backend)
compiled_model(example_input)