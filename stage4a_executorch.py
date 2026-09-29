"""Stage 4a (v2): ExecuTorch lowering of MobileNetV2, portable ops vs the XNNPACK delegate.

v1 lowered with plain to_edge() -> to_executorch(). With no partitioner, every op
runs on ExecuTorch's *portable* kernels: straightforward reference C++ with no
SIMD tuning, meant for correctness and portability, not speed. This stage makes
that explicit and compares three programs:

  portable_fp32  to_edge().to_executorch()                  (what v1 did)
  xnnpack_fp32   to_edge_transform_and_lower(XnnpackPartitioner)
  xnnpack_int8   PT2E static quantization (XNNPACKQuantizer, per-channel
                 symmetric int8 weights, int8 activations, calibrated on the same
                 256 images as stage 3), then lowered to XNNPACK

For each program we record: which ops were delegated and which fell back to
portable kernels (get_delegation_info), .pte size, latency, top-1 and agreement
with fp32. XNNPACK is the CPU backend ExecuTorch uses on Android/iOS; on a
Snapdragon phone this is the path that runs on the Kryo/Oryon CPU cores (the
Hexagon NPU path is the QNN backend, exercised through AI Hub in stage 5).
"""
import argparse
import numpy as np
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.extension.pybindings import portable_lib as et_rt
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

import common


def delegation_summary(edge_manager):
    """Per-op-type count of what XNNPACK took vs what stayed on portable kernels."""
    info = get_delegation_info(edge_manager.exported_program().graph_module)
    df = info.get_operator_delegation_dataframe()
    df = df[df.op_type != "Total"]
    fallback = {r.op_type: int(r.occurrences_in_non_delegated_graphs)
                for r in df.itertuples()
                if r.occurrences_in_non_delegated_graphs and r.op_type != "getitem"}
    return {
        "num_delegated_subgraphs": int(info.num_delegated_subgraphs),
        "delegated_ops": int(info.num_delegated_nodes),
        "non_delegated_ops": int(info.num_non_delegated_nodes),
        "fallback_op_types": fallback,
    }


def build(name, model, x, n_calib):
    ep = torch.export.export(model, (x,))
    ref_module = model  # what the lowered program must reproduce
    if name == "portable_fp32":
        edge = to_edge(ep)
    elif name == "xnnpack_fp32":
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    elif name == "xnnpack_int8":
        q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
        m = prepare_pt2e(ep.module(), q)
        with torch.no_grad():
            for xb, _ in common.iter_batches("calib", 1, n_calib):
                m(torch.from_numpy(xb))
        m = convert_pt2e(m)
        ref_module = m
        ep = torch.export.export(m, (x,))
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    else:
        raise ValueError(name)
    deleg = delegation_summary(edge)
    prog = edge.to_executorch()
    path = common.art(f"mobilenet_v2_{name}.pte")
    with open(path, "wb") as f:
        f.write(prog.buffer)
    return path, bytes(prog.buffer), deleg, ref_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["portable_fp32", "xnnpack_fp32", "xnnpack_int8"])
    ap.add_argument("--n-calib", type=int, default=common.N_CALIB_DEFAULT)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--portable-eval-limit", type=int, default=100,
                    help="portable kernels are slow; score them on a subset")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lowering-check-n", type=int, default=300)
    args = ap.parse_args()

    model = common.get_model()
    x = torch.from_numpy(common.example_input())
    ref_logits = np.load(common.art("fp32_ref_logits.npy"))
    print(f"[et] registered backends: {et_rt._get_registered_backend_names()}")
    print(f"[et] threadpool threads: {et_rt._threadpool_get_thread_count()}")

    rows = {}
    for name in args.variants:
        path, buf, deleg, ref_module = build(name, model, x, args.n_calib)
        mod = et_rt._load_for_executorch_from_buffer(buf)
        run = lambda xb: mod.forward((torch.from_numpy(xb),))[0].numpy()
        limit = args.portable_eval_limit if name.startswith("portable") else args.eval_limit
        acc, et_logits = common.evaluate(run, limit=limit, ref_logits=ref_logits)

        # Lowering check: the .pte vs its own PyTorch reference (eager fp32, or the
        # PT2E-converted module for int8) on the same images. For fp32 this should be
        # ~1e-5 (float reassociation). For int8 it is NOT expected to be ~0: the torch
        # reference simulates quantization (dequant -> fp32 conv -> quant), while XNNPACK
        # runs integer kernels with fixed-point requantization, so rounding differs by up
        # to 1 LSB per layer and compounds over 52 layers. What must hold is that the two
        # agree on the prediction almost always.
        k = min(args.lowering_check_n, len(et_logits))
        with torch.no_grad():
            ref_out = np.concatenate([ref_module(torch.from_numpy(xb)).numpy()
                                      for xb, _ in common.iter_batches("eval", 1, k)])
        d = np.abs(et_logits[:k] - ref_out)
        lowering = {
            "n": int(k),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "top1_agreement_pct": float((et_logits[:k].argmax(1) == ref_out.argmax(1)).mean() * 100),
        }
        iters = min(args.iters, 20) if name.startswith("portable") else args.iters
        lat = common.bench(lambda t: mod.forward((t,)), x, warmup=5 if name.startswith("portable") else 20, iters=iters)
        rows[name] = {**acc, "size_bytes": len(buf), "latency": lat, "delegation": deleg,
                      "lowering_vs_own_ref": lowering}
        print(f"  {name:14s} top1 {acc['top1']:6.2f}% (n={acc['n']})  agree {acc['top1_agreement_vs_fp32']:6.2f}%  "
              f"vs own ref: max|d| {lowering['max_abs_diff']:.1e} mean|d| {lowering['mean_abs_diff']:.1e} "
              f"agree {lowering['top1_agreement_pct']:.1f}%  p50 {lat['p50_ms']:8.3f} ms  {len(buf) / 1e6:5.2f} MB  "
              f"delegated {deleg['delegated_ops']} / fallback {deleg['non_delegated_ops']} {deleg['fallback_op_types']}")

    if "portable_fp32" in rows:
        base = rows["portable_fp32"]["latency"]["p50_ms"]
        for r in rows.values():
            r["speedup_vs_portable"] = round(base / r["latency"]["p50_ms"], 2)

    common.save_result("stage4a_executorch", {
        "threadpool_threads": et_rt._threadpool_get_thread_count(),
        "variants": rows,
    })


if __name__ == "__main__":
    main()
