"""Shared pieces for the v2 (MobileNetV2) pipeline.

Everything that more than one stage needs lives here so that every stage
measures the same model, on the same images, with the same timing method:

  * the model   : torchvision MobileNetV2, IMAGENET1K_V1 weights (71.878% top-1)
  * the data    : Imagenette (10 ImageNet classes), preprocessed once and cached
  * calibration : a fixed, class-balanced slice of the Imagenette *train* split
  * evaluation  : the Imagenette *val* split (never used for calibration)
  * latency     : warmup + N timed runs, reported as p50 / p90 / mean

Smoke mode (CB_SMOKE=1) swaps in random weights and random images so the whole
pipeline can be exercised without network access. Smoke results are tagged
and are NOT valid numbers; report.py refuses to put them in the README.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
RES_DIR = ROOT / "results"
for d in (DATA_DIR, ART_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

SMOKE = os.environ.get("CB_SMOKE") == "1"
SEED = 42
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "image"          # used for ONNX and AI Hub so names line up everywhere

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Imagenette folder (WordNet id) -> index in the 1000-class ImageNet head.
# The model still predicts over all 1000 classes; a prediction counts as correct
# only if the full 1000-way argmax lands on the right ImageNet index. That makes
# this a real (if 10-class) ImageNet top-1 measurement, not a 10-way re-head.
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

N_CALIB_DEFAULT = 256


# --------------------------------------------------------------------------- model
def get_model():
    """MobileNetV2 in eval mode. Pretrained unless in smoke mode."""
    import torch
    import torchvision

    torch.manual_seed(SEED)
    if SMOKE:
        model = torchvision.models.mobilenet_v2(weights=None)
    else:
        w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = torchvision.models.mobilenet_v2(weights=w)
    return model.eval()


def example_input():
    """Deterministic real-valued input for export / tracing (a real eval image when available)."""
    x_u8, _ = load_split("eval")
    return normalize(x_u8[:1])


# --------------------------------------------------------------------------- data
def _preprocess_pil(img):
    """torchvision's standard ImageNet eval transform, minus normalization.
    Resize short side to 256 (bilinear), center-crop 224. Returns uint8 CHW."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def normalize(x_u8: np.ndarray) -> np.ndarray:
    """uint8 NCHW -> normalized float32 NCHW (what the model actually consumes)."""
    return ((x_u8.astype(np.float32) / 255.0) - MEAN) / STD


def _cache_paths(split):
    tag = "smoke_" if SMOKE else ""
    return DATA_DIR / f"{tag}{split}_x.npy", DATA_DIR / f"{tag}{split}_y.npy"


IMG_EXT = {".jpg", ".jpeg", ".png"}


def _find_split_dir(root: Path, split: str) -> Path:
    """Locate <something>/<split>/<wnid>/ under root, whatever the archive's top folder is called."""
    wnid = next(iter(IMAGENETTE_TO_IMAGENET))
    for cand in [root / split, *sorted(root.rglob(split))]:
        if (cand / wnid).is_dir():
            return cand
    seen = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())[:30]
    raise SystemExit(f"[data] no '{split}/{wnid}/' folder found under {root}.\n"
                     f"       Folders found: {seen}\n"
                     f"       Delete {root}/imagenette2-320* and rerun to re-download.")


def _list_images(d: Path):
    # case-insensitive extension match; skip macOS AppleDouble '._*' files
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))


def build_cache(imagenette_root: Path, n_calib: int = N_CALIB_DEFAULT):
    """Decode + preprocess once, cache as uint8 so every stage reads identical pixels."""
    rng = np.random.default_rng(SEED)
    from PIL import Image

    train_dir = _find_split_dir(imagenette_root, "train")
    val_dir = _find_split_dir(imagenette_root, "val")
    print(f"[data] train: {train_dir}\n[data] val:   {val_dir}")

    # calibration: class-balanced sample of TRAIN
    per_class = int(np.ceil(n_calib / len(IMAGENETTE_TO_IMAGENET)))
    calib_files, calib_y = [], []
    for wnid, idx in sorted(IMAGENETTE_TO_IMAGENET.items()):
        files = _list_images(train_dir / wnid)
        if len(files) < per_class:
            raise SystemExit(f"[data] {train_dir / wnid} has {len(files)} images, need {per_class}")
        pick = rng.choice(len(files), size=per_class, replace=False)
        calib_files += [files[i] for i in sorted(pick)]
        calib_y += [idx] * per_class
    order = rng.permutation(len(calib_files))[:n_calib]
    calib_files = [calib_files[i] for i in order]
    calib_y = [calib_y[i] for i in order]

    # evaluation: ALL of VAL
    eval_files, eval_y = [], []
    for wnid, idx in sorted(IMAGENETTE_TO_IMAGENET.items()):
        files = _list_images(val_dir / wnid)
        if not files:
            raise SystemExit(f"[data] {val_dir / wnid} has no images")
        eval_files += files
        eval_y += [idx] * len(files)
    # Shuffle once (fixed seed) so any prefix of the eval set is a random, mixed-class
    # subset. Without this, --eval-limit 100 would be 100 images of class 0 (tench).
    perm = rng.permutation(len(eval_files))
    eval_files = [eval_files[i] for i in perm]
    eval_y = [eval_y[i] for i in perm]

    for split, files, ys in (("calib", calib_files, calib_y), ("eval", eval_files, eval_y)):
        xs = np.stack([_preprocess_pil(Image.open(f)) for f in files])
        xp, yp = _cache_paths(split)
        np.save(xp, xs)
        np.save(yp, np.array(ys, dtype=np.int64))
        print(f"[data] {split}: {xs.shape[0]} images -> {xp.name}")


def _build_smoke_cache():
    rng = np.random.default_rng(SEED)
    labels = np.array(list(IMAGENETTE_TO_IMAGENET.values()))
    for split, n in (("calib", 32), ("eval", 64)):
        xs = rng.integers(0, 256, size=(n, 3, 224, 224), dtype=np.uint8)
        ys = rng.choice(labels, size=n)
        xp, yp = _cache_paths(split)
        np.save(xp, xs)
        np.save(yp, ys)


def load_split(split: str, limit: int | None = None):
    xp, yp = _cache_paths(split)
    if not xp.exists():
        if SMOKE:
            _build_smoke_cache()
        else:
            raise SystemExit(f"{xp} missing. Run `python data_prep.py` first.")
    x, y = np.load(xp, mmap_mode="r"), np.load(yp)
    if limit:
        x, y = x[:limit], y[:limit]
    return x, y


def iter_batches(split: str, batch_size: int = 1, limit: int | None = None):
    x, y = load_split(split, limit)
    for i in range(0, len(x), batch_size):
        yield normalize(np.asarray(x[i:i + batch_size])), y[i:i + batch_size]


# --------------------------------------------------------------------------- metrics
def evaluate(run_fn, split="eval", limit=None, batch_size=1, ref_logits=None):
    """Top-1 on the eval split. run_fn: float32 NCHW ndarray -> logits ndarray.
    If ref_logits (fp32 logits in the same order) is given, also report how often
    the prediction agrees with fp32 and how far the logits moved."""
    preds, labels, logits_all = [], [], []
    for x, y in iter_batches(split, batch_size, limit):
        out = np.asarray(run_fn(x), dtype=np.float32)
        logits_all.append(out)
        preds.append(out.argmax(1))
        labels.append(y)
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    logits = np.concatenate(logits_all)
    res = {"n": int(len(labels)), "top1": float((preds == labels).mean() * 100)}
    if ref_logits is not None:
        ref = ref_logits[: len(logits)]
        res["top1_agreement_vs_fp32"] = float((preds == ref.argmax(1)).mean() * 100)
        d = np.abs(logits - ref)
        res["logit_max_abs_diff"] = float(d.max())
        res["logit_mean_abs_diff"] = float(d.mean())
    return res, logits


def bench(fn, x, warmup=20, iters=100):
    """Wall-clock latency of fn(x). Median is the headline number: it is robust to
    the occasional scheduler hiccup that skews a mean on a laptop."""
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return {
        "p50_ms": round(statistics.median(t), 3),
        "p90_ms": round(t[int(0.9 * (len(t) - 1))], 3),
        "mean_ms": round(statistics.fmean(t), 3),
        "min_ms": round(t[0], 3),
        "iters": iters,
    }


# --------------------------------------------------------------------------- bookkeeping
def env_info():
    info = {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "smoke": SMOKE,
    }
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("torch", "torchvision", "onnx", "onnxruntime", "executorch", "torchao", "qai-hub"):
        try:
            info[pkg.replace("-", "_")] = version(pkg)
        except PackageNotFoundError:
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            info["chip"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return info


def save_result(name: str, payload: dict):
    payload = {"env": env_info(), **payload}
    tag = "smoke_" if SMOKE else ""
    path = RES_DIR / f"{tag}{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[results] wrote {path.relative_to(ROOT)}")
    return path


def art(name: str) -> str:
    tag = "smoke_" if SMOKE else ""
    return str(ART_DIR / f"{tag}{name}")
