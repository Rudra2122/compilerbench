"""Render results/*.json into markdown and inject it into README.md.

    python report.py            # writes results/RESULTS.md and updates README.md
    python report.py --no-readme

Smoke-mode results (random weights, random images) are never written to the README.
"""
import argparse
import json
import re
from pathlib import Path

import common

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load(name, smoke=False):
    p = common.RES_DIR / f"{'smoke_' if smoke else ''}{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(v, fmt="{:.2f}", none="–"):
    return none if v is None else fmt.format(v)


def kernels(ops):
    keep = ["QLinearConv", "ConvInteger", "DynamicQuantizeLinear", "FusedConv", "Conv",
            "QLinearAdd", "QGemm", "DynamicQuantizeMatMul", "QuantizeLinear", "DequantizeLinear"]
    return ", ".join(f"{ops[k]} {k}" for k in keep if ops.get(k))


def env_line(e):
    chip = e.get("chip") or e.get("processor")
    return (f"_{chip}, {e.get('cpu_count')} cores, {e.get('system')}; torch {e.get('torch')}, "
            f"onnxruntime {e.get('onnxruntime')}, executorch {e.get('executorch')}_")


def render(smoke=False):
    out = []
    s2 = load("stage2a_onnx", smoke)
    s3 = load("stage3_quantize", smoke)
    s4 = load("stage4a_executorch", smoke)
    s5 = load("stage5_aihub", smoke)
    if smoke:
        out.append("> **SMOKE RUN: random weights, random images. These numbers are not results.**\n")

    if s2:
        t, o = s2["torch_fp32"], s2["onnx_fp32"]
        out += ["### fp32 reference (MobileNetV2, Imagenette val, 1000-way head)", "",
                env_line(s2["env"]), "",
                "| Path | Top-1 | Images | Max \\|Δlogit\\| vs PyTorch |", "|---|---|---|---|",
                f"| PyTorch eager | {t['top1']:.2f}% | {t['n']} | – |",
                f"| ONNX Runtime | {o['top1']:.2f}% | {o['n']} | {o['logit_max_abs_diff']:.1e} |", ""]

    if s3:
        v = s3["variants"]
        out += ["### Stage 3: int8 PTQ with ONNX Runtime (CPU)", "", env_line(s3["env"]), "",
                f"Calibration: {s3['n_calib']} Imagenette-train images. Weights int8 symmetric, "
                "activations uint8 asymmetric, QDQ format. "
                f"Latency is batch 1, p50 of {v['fp32']['latency_1thread']['iters']} runs.", "",
                "| Variant | Top-1 | Δ vs fp32 | Agrees w/ fp32 | p50 1 thread | Speedup | p50 all threads | Speedup | Size | Kernels ORT actually runs |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in v.items():
            out.append(
                f"| `{name}` | {r['top1']:.2f}% | {r['top1_delta_pp']:+.2f} pp | "
                f"{f(r.get('top1_agreement_vs_fp32'), '{:.2f}%')} | "
                f"{r['latency_1thread']['p50_ms']:.2f} ms | {r['speedup_1thread']:.2f}x | "
                f"{r['latency_allthreads']['p50_ms']:.2f} ms | {r['speedup_allthreads']:.2f}x | "
                f"{r['size_bytes'] / 1e6:.1f} MB | {kernels(r['runtime_ops'])} |")
        wr = s3["weight_ranges"]
        w = wr["worst_layers"][0]
        out += ["", "**Why per-tensor hurts MobileNetV2 (measured on the BN-folded weights):** "
                f"the median ratio between the largest and smallest per-output-channel max|w| is "
                f"{wr['median_range_ratio_depthwise']:.1f}x in depthwise convs vs "
                f"{wr['median_range_ratio_pointwise']:.1f}x in pointwise convs. Worst layer "
                f"(`{w['node']}`, {'depthwise' if w['depthwise'] else 'pointwise'}): {w['range_ratio']:.0f}x, "
                f"so under one per-tensor scale its smallest channel gets {w['min_channel_int8_levels']:.1f} "
                f"of 127 int8 levels, and {w['channels_under_8_levels']} channels get fewer than 8.", ""]

    if s4:
        v = s4["variants"]
        out += ["### Stage 4a: ExecuTorch, portable kernels vs XNNPACK delegate (CPU)", "",
                env_line(s4["env"]), "",
                f"XNNPACK threadpool: {s4.get('threadpool_threads')} threads. Batch 1.", "",
                "| Program | Delegated / fallback ops | Top-1 (n) | Agrees w/ fp32 | p50 | vs portable | .pte size | vs own PyTorch ref: max / mean \\|Δlogit\\|, argmax agree |",
                "|---|---|---|---|---|---|---|---|"]
        for name, r in v.items():
            d = r["delegation"]
            fb = ", ".join(f"{k.removeprefix('aten_').removesuffix('_default')}×{c}"
                           for k, c in d["fallback_op_types"].items()) or "none"
            out.append(
                f"| `{name}` | {d['delegated_ops']} / {d['non_delegated_ops']} ({fb}) | "
                f"{r['top1']:.2f}% ({r['n']}) | {r['top1_agreement_vs_fp32']:.2f}% | "
                f"{r['latency']['p50_ms']:.2f} ms | {f(r.get('speedup_vs_portable'), '{:.1f}x')} | "
                f"{r['size_bytes'] / 1e6:.1f} MB | {r['lowering_vs_own_ref']['max_abs_diff']:.1e} / "
                f"{r['lowering_vs_own_ref']['mean_abs_diff']:.1e}, {r['lowering_vs_own_ref']['top1_agreement_pct']:.1f}% |")
        out.append("")

    if s5:
        out += [f"### Stage 5: on-device, Qualcomm AI Hub ({s5['device']})", "",
                "| Variant | Inference (on-device) | Layers by compute unit | On-device top-1 (n) | fp32 top-1, same images | Agrees w/ local fp32 | Jobs |",
                "|---|---|---|---|---|---|---|"]
        for name, r in s5["variants"].items():
            jobs = " ".join(f"[{k}]({u})" for k, u in s5["jobs"].get(name, {}).items())
            if "error" in r:
                out.append(f"| `{name}` | failed: {r['error'][:80]} | | | | | {jobs} |")
                continue
            units = ", ".join(f"{k} {c}" for k, c in r["compute_unit_layers"].items())
            dev = (f"{r['on_device_top1']:.2f}% ({r['on_device_n']})" if "on_device_top1" in r else "–")
            out.append(
                f"| `{name}` | {f(r['inference_ms'], '{:.3f} ms')} | {units} | {dev} | "
                f"{f(r.get('local_fp32_top1_same_subset'), '{:.2f}%')} | "
                f"{f(r.get('top1_agreement_vs_local_fp32'), '{:.2f}%')} | {jobs} |")
        off = {n: r.get("layers_off_npu") for n, r in s5["variants"].items() if r.get("layers_off_npu")}
        if off:
            out += ["", "Layers not placed on the NPU:", ""]
            for n, layers in off.items():
                out.append(f"- `{n}`: " + ", ".join(f"{l['type']} ({l['unit']})" for l in layers[:10]))
        out.append("")

    if len(out) <= 1:
        out.append("_No results yet. Run `make all`._")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="render smoke results (never touches README)")
    args = ap.parse_args()
    smoke = args.smoke or common.SMOKE

    md = render(smoke)
    name = "SMOKE_RESULTS.md" if smoke else "RESULTS.md"
    (common.RES_DIR / name).write_text(md + "\n")
    print(f"[report] wrote results/{name}")

    if smoke or args.no_readme:
        return
    if any((load(n) or {}).get("env", {}).get("smoke") for n in
           ("stage2a_onnx", "stage3_quantize", "stage4a_executorch", "stage5_aihub")):
        raise SystemExit("[report] a results file came from a smoke run; README not updated")
    readme = common.ROOT / "README.md"
    text = readme.read_text()
    if START in text and END in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END),
                      lambda _: f"{START}\n{md}\n{END}", text, flags=re.S)
        readme.write_text(text)
        print("[report] README.md results section updated")


if __name__ == "__main__":
    main()
