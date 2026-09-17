import onnx, tvm, numpy as np
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

onnx_model = onnx.load("tinycnn.onnx")
mod = from_onnx(onnx_model, keep_params_in_input=False)
open("tvm_ir_before.txt", "w").write(mod.script())

mod = relax.transform.LegalizeOps()(mod)
mod = relax.transform.AnnotateTIROpPattern()(mod)
mod = relax.transform.FuseOps()(mod)
mod = relax.transform.FuseTIR()(mod)
open("tvm_ir_after.txt", "w").write(mod.script())

before = open("tvm_ir_before.txt").read().count("\n")
after = open("tvm_ir_after.txt").read().count("\n")
print(f"IR before: {before} lines, after: {after} lines")

target = tvm.target.Target("llvm")
ex = relax.build(mod, target=target)
vm = relax.VirtualMachine(ex, tvm.cpu())
inp = tvm.runtime.tensor(np.random.randn(1,3,32,32).astype("float32"), device=tvm.cpu())
out = vm["main"](inp)
print("compiled + ran, output shape:", out.numpy().shape)