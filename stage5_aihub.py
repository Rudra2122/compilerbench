"""Stage 5 (v2): run MobileNetV2 on real Snapdragon hardware through Qualcomm AI Hub.

Everything before this stage runs on a laptop CPU. This stage compiles, quantizes,
profiles and runs inference on a hosted Snapdragon phone, so the numbers come
from the target: Hexagon NPU via QNN, with per-layer compute-unit placement.

Variants (all on the same device):
  tflite_fp32       TorchScript -> TFLite, AI Hub picks the delegate
  qnn_fp32          TorchScript -> QNN DLC (NPU executes fp32 graphs in fp16)
  qnn_int8          TorchScript -> ONNX -> AI Hub quantize job (w8a8, calibrated on
                    OUR 256 calibration images) -> QNN DLC, float I/O.
                    Profiled AND run on N eval images on-device -> on-target top-1.
  qnn_int8_qio      same int8 model compiled with --quantize_io (uint8 in/out),
                    which is how you would actually ship it. Profile only.

Setup (once):
    pip install qai-hub
    qai-hub configure --api_token <token from aihub.qualcomm.com -> Settings>
    python stage5_aihub.py --list-devices

    python stage5_aihub.py                                 # default device
    python stage5_aihub.py --device "Snapdragon 8 Elite QRD"
    python stage5_aihub.py --dry-run                       # print the plan only

Every job URL is saved to results/stage5_aihub.json, so each number links back to
the AI Hub job that produced it.
"""
import argparse
import traceback
from collections import Counter

import numpy as np
import torch

import common

DEFAULT_DEVICE = "Samsung Galaxy S24 (Family)"


def profile_summary(prof: dict):
    s = prof.get("execution_summary", {})
    layers = prof.get("execution_detail", [])
    units = Counter(l.get("compute_unit", "?") for l in layers)
    off_npu = [{"name": l.get("name"), "type": l.get("type"), "unit": l.get("compute_unit")}
               for l in layers if l.get("compute_unit") != "NPU"]
    t = s.get("estimated_inference_time")
    return {
        "inference_ms": round(t / 1000, 3) if isinstance(t, (int, float)) else None,
        "peak_memory_range_bytes": s.get("inference_memory_peak_range"),
        "first_load_ms": (s["first_load_time"] / 1000) if isinstance(s.get("first_load_time"), (int, float)) else None,
        "warm_load_ms": (s["warm_load_time"] / 1000) if isinstance(s.get("warm_load_time"), (int, float)) else None,
        "compute_unit_layers": dict(units),
        "layers_off_npu": off_npu[:25],
        "all_inference_times_us": s.get("all_inference_times"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--n-calib", type=int, default=common.N_CALIB_DEFAULT)
    ap.add_argument("--n-eval", type=int, default=200, help="images run on-device for accuracy")
    ap.add_argument("--variants", nargs="*", default=["tflite_fp32", "qnn_fp32", "qnn_int8", "qnn_int8_qio"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import qai_hub as hub

    if args.list_devices:
        for d in hub.get_devices():
            chip = next((a for a in d.attributes if a.startswith("chipset:")), "")
            print(f"{d.name:45s} {d.os:8s} {chip}")
        return

    if args.dry_run:
        print(f"device: {args.device}\nvariants: {args.variants}\n"
              f"calibration images uploaded: {args.n_calib}\non-device eval images: {args.n_eval}")
        return

    device = hub.Device(args.device)
    model = common.get_model()
    x = torch.from_numpy(common.example_input())
    traced = torch.jit.trace(model, x)
    specs = {common.INPUT_NAME: tuple(common.INPUT_SHAPE)}
    ref_logits = np.load(common.art("fp32_ref_logits.npy"))

    out = {"device": args.device, "variants": {}, "jobs": {}}

    def record(name, job_kind, job):
        out["jobs"].setdefault(name, {})[job_kind] = job.url
        print(f"  [{name}] {job_kind}: {job.url}")

    # ---- int8: compile to ONNX, then let AI Hub quantize it with OUR calibration set
    quant_model = None
    if any(v.startswith("qnn_int8") for v in args.variants):
        try:
            cj = hub.submit_compile_job(model=traced, device=device, input_specs=specs,
                                        options="--target_runtime onnx", name="cb_mnv2_to_onnx")
            record("qnn_int8", "compile_onnx", cj)
            calib = {common.INPUT_NAME: [xb for xb, _ in common.iter_batches("calib", 1, args.n_calib)]}
            qj = hub.submit_quantize_job(model=cj.get_target_model(), calibration_data=calib,
                                         weights_dtype=hub.QuantizeDtype.INT8,
                                         activations_dtype=hub.QuantizeDtype.INT8,
                                         name="cb_mnv2_quantize_w8a8")
            record("qnn_int8", "quantize", qj)
            quant_model = qj.get_target_model()
        except Exception:
            traceback.print_exc()

    plan = {
        "tflite_fp32": (traced, "--target_runtime tflite", True),
        "qnn_fp32": (traced, "--target_runtime qnn_dlc", True),
        "qnn_int8": (quant_model, "--target_runtime qnn_dlc", False),
        "qnn_int8_qio": (quant_model, "--target_runtime qnn_dlc --quantize_io", False),
    }

    eval_x, eval_y = [], []
    for xb, yb in common.iter_batches("eval", 1, args.n_eval):
        eval_x.append(xb)
        eval_y.append(int(yb[0]))
    eval_y = np.array(eval_y)

    for name in args.variants:
        src, opts, needs_specs = plan[name]
        if src is None:
            print(f"[{name}] skipped (quantize step failed)")
            continue
        print(f"[{name}] {opts}")
        try:
            cj = hub.submit_compile_job(model=src, device=device, options=opts, name=f"cb_mnv2_{name}",
                                        **({"input_specs": specs} if needs_specs else {}))
            record(name, "compile", cj)
            target = cj.get_target_model()
            if target is None:
                raise RuntimeError(f"compile failed, see {cj.url}")
            pj = hub.submit_profile_job(model=target, device=device, name=f"cb_mnv2_{name}_profile")
            record(name, "profile", pj)
            row = profile_summary(pj.download_profile())

            # on-device accuracy for the float-I/O models
            if name in ("qnn_int8", "qnn_fp32", "tflite_fp32") and args.n_eval:
                ij = hub.submit_inference_job(model=target, device=device,
                                              inputs={common.INPUT_NAME: eval_x},
                                              name=f"cb_mnv2_{name}_infer")
                record(name, "inference", ij)
                outputs = ij.download_output_data()
                logits = np.concatenate([np.asarray(o).reshape(1, -1)
                                         for o in next(iter(outputs.values()))])
                ref = ref_logits[: len(logits)]
                row.update({
                    "on_device_n": int(len(logits)),
                    "on_device_top1": float((logits.argmax(1) == eval_y).mean() * 100),
                    "local_fp32_top1_same_subset": float((ref.argmax(1) == eval_y).mean() * 100),
                    "top1_agreement_vs_local_fp32": float((logits.argmax(1) == ref.argmax(1)).mean() * 100),
                })
            out["variants"][name] = row
            print(f"  -> {row.get('inference_ms')} ms, units {row['compute_unit_layers']}"
                  + (f", on-device top1 {row['on_device_top1']:.2f}%" if "on_device_top1" in row else ""))
        except Exception as e:
            traceback.print_exc()
            out["variants"][name] = {"error": f"{type(e).__name__}: {e}"}

    common.save_result("stage5_aihub", out)


if __name__ == "__main__":
    main()
