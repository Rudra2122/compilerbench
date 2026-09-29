"""Download Imagenette + MobileNetV2 weights, then build the fixed calib / eval caches.

Imagenette (fast.ai) is 10 easily-separable ImageNet classes. We use it because
full ImageNet val is 6.4 GB behind a license wall; Imagenette is ~330 MB and its
labels map 1:1 onto ImageNet indices, so the pretrained 1000-class head is scored
as-is. Absolute top-1 is therefore higher than on full ImageNet; the number that
matters here is the *delta* between fp32 and each quantized variant.

    python data_prep.py                 # default: 256 calibration images
    python data_prep.py --n-calib 512
    python data_prep.py --force         # wipe and re-download

The download goes to a .part file and is only renamed when complete; extraction
writes a marker only after every member is extracted. A half-finished download or
extraction is detected and redone instead of being silently used.
"""
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

import common

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
MARKER = ".extracted_ok"


def download(tgz: Path):
    part = tgz.with_suffix(".tgz.part")
    print(f"[data] downloading {URL}")
    with urllib.request.urlopen(URL) as r, open(part, "wb") as f:
        expected = int(r.headers.get("Content-Length", 0))
        shutil.copyfileobj(r, f, length=1 << 20)
    got = part.stat().st_size
    if expected and got != expected:
        part.unlink()
        raise SystemExit(f"[data] download truncated ({got} of {expected} bytes). Rerun.")
    part.rename(tgz)
    print(f"[data] downloaded {got / 1e6:.0f} MB")


def extract(tgz: Path, root: Path) -> Path:
    print("[data] extracting")
    try:
        with tarfile.open(tgz) as t:
            members = t.getmembers()          # reads the whole archive: fails fast if corrupt
            t.extractall(root, members=members)
    except (tarfile.TarError, EOFError, OSError) as e:
        tgz.unlink(missing_ok=True)
        raise SystemExit(f"[data] archive is corrupt ({e}); deleted it. Rerun to re-download.")
    top = root / members[0].name.split("/")[0]
    (top / MARKER).write_text("ok")
    return top


def check_complete(src: Path):
    """Every class must exist in both splits with a plausible number of images."""
    problems = []
    for split, min_n in (("train", 500), ("val", 200)):
        for wnid in common.IMAGENETTE_TO_IMAGENET:
            d = src / split / wnid
            n = len(common._list_images(d)) if d.is_dir() else 0
            if n < min_n:
                problems.append(f"{split}/{wnid}: {n} images")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=common.N_CALIB_DEFAULT)
    ap.add_argument("--root", type=Path, default=common.DATA_DIR)
    ap.add_argument("--force", action="store_true", help="delete and re-download the dataset")
    args = ap.parse_args()

    tgz = args.root / "imagenette2-320.tgz"
    src = args.root / "imagenette2-320"
    if args.force:
        shutil.rmtree(src, ignore_errors=True)
        tgz.unlink(missing_ok=True)

    if not (src / MARKER).exists():
        shutil.rmtree(src, ignore_errors=True)       # partial extraction from an earlier run
        if not tgz.exists():
            download(tgz)
        src = extract(tgz, args.root)

    problems = check_complete(src)
    if problems:
        present = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
        raise SystemExit("[data] dataset incomplete:\n  " + "\n  ".join(problems)
                         + f"\nClass folders actually in train/: {present}"
                         + "\nIf the folder names differ from IMAGENETTE_TO_IMAGENET in common.py, fix the table."
                         + "\nIf the folders are there but empty, run: python data_prep.py --force")

    common.build_cache(src, n_calib=args.n_calib)

    # warm the torchvision weight cache so later stages never hit the network
    common.get_model()
    print("[data] MobileNetV2 weights cached")


if __name__ == "__main__":
    main()
