"""Outpost Wildlife Fine-Tune Trainer -- closes the human-reinforcement loop.

Runs on Mike's GPU box (Carmody-PC / RTX 4090), NOT the Railway-hosted
telemetry server. Pulls confirmed human review corrections from
/api/review/export, builds a YOLO detection dataset from them, and
fine-tunes a checkpoint starting from the base model (yolo11x.pt by
default).

Until this script existed, /api/review/export was a dead-end endpoint --
nothing called it, nothing consumed its output, no training ever ran.
Mike caught this live (#side-project 2026-09-22): "not sure why you guys
built half a system." This is the missing half.

Usage:
  pip install ultralytics torch torchvision pillow requests
  python train_finetune.py --api-url https://outpost.brock.ventures --epochs 15

IMPORTANT -- this does NOT automatically deploy the resulting weights.
runner.py keeps using whatever --model it was started with. Point a
runner at the new checkpoint deliberately (--model models/wildlife_finetuned.pt)
once you've looked at the run's val metrics and decided it's actually
better than the base model -- fine-tuning on a handful of examples can
just as easily make things worse (catastrophic forgetting of the other
COCO classes the base model already handled fine).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "finetune_dataset"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_MODEL = MODELS_DIR / "wildlife_finetuned.pt"


def fetch_manifest(api_url: str) -> List[Dict[str, Any]]:
    """Pulls the accept+relabel review manifest from the live server."""
    url = f"{api_url.rstrip('/')}/api/review/export"
    resp = requests.get(url, params={"format": "yolo_manifest"}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("manifest", [])


def download_image(api_url: str, image_path: str, dest: Path) -> bool:
    """Downloads one snapshot image, resolving its relative URL against api_url."""
    if not image_path:
        return False
    url = image_path if image_path.startswith("http") else f"{api_url.rstrip('/')}{image_path}"
    try:
        resp = requests.get(url, timeout=15.0)
        if resp.status_code != 200 or not resp.content:
            return False
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"[!] Failed to download {url}: {exc}")
        return False


def bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Converts an absolute-pixel [x1, y1, x2, y2] bbox to YOLO's normalized
    (x_center, y_center, width, height), each in [0, 1]."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
    y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def build_dataset(manifest: List[Dict[str, Any]], api_url: str, dataset_dir: Path) -> Dict[str, Any]:
    """Downloads images, converts labels, and writes a YOLO-format dataset
    (images/{train,val}/, labels/{train,val}/, data.yaml) to dataset_dir."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Stable class ordering: alphabetical, so a re-run with the same label
    # set produces the same class indices (a shuffled order would silently
    # invalidate any weights trained under a previous ordering).
    labels_present = sorted({(m.get("label") or "").strip().lower() for m in manifest if m.get("label")})
    class_index = {label: i for i, label in enumerate(labels_present)}

    prepared = []
    skipped = 0
    for i, entry in enumerate(manifest):
        label = (entry.get("label") or "").strip().lower()
        bbox = entry.get("bbox")
        image_path = entry.get("image")
        if not label or not bbox or len(bbox) != 4 or not image_path:
            skipped += 1
            continue

        tmp_img = dataset_dir / f"_tmp_{i}.jpg"
        if not download_image(api_url, image_path, tmp_img):
            skipped += 1
            continue

        try:
            with Image.open(tmp_img) as im:
                img_w, img_h = im.size
        except Exception as exc:
            print(f"[!] Unreadable image {image_path}: {exc}")
            tmp_img.unlink(missing_ok=True)
            skipped += 1
            continue

        if img_w <= 0 or img_h <= 0:
            tmp_img.unlink(missing_ok=True)
            skipped += 1
            continue

        yolo_box = bbox_to_yolo(bbox, img_w, img_h)
        prepared.append((tmp_img, class_index[label], yolo_box))

    if not prepared:
        return {"count": 0, "classes": labels_present, "skipped": skipped}

    # 80/20 train/val split. With very few examples, ultralytics still
    # needs at least one image in val or training crashes at the first
    # validation pass -- guarantee that explicitly rather than trusting
    # the split math to produce one on its own for small counts.
    split_idx = max(1, int(len(prepared) * 0.8))
    if split_idx >= len(prepared):
        split_idx = len(prepared) - 1
    train_set, val_set = prepared[:split_idx], prepared[split_idx:]
    if not val_set:
        val_set = [train_set[-1]]

    for split_name, items in (("train", train_set), ("val", val_set)):
        for j, (tmp_img, cls_id, (xc, yc, w, h)) in enumerate(items):
            img_dest = dataset_dir / "images" / split_name / f"img_{j:04d}.jpg"
            label_dest = dataset_dir / "labels" / split_name / f"img_{j:04d}.txt"
            shutil.copy(tmp_img, img_dest)
            label_dest.write_text(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    for tmp_img, _, _ in prepared:
        tmp_img.unlink(missing_ok=True)

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n{}\n".format(
            str(dataset_dir.resolve()),
            "\n".join(f"  {i}: {name}" for name, i in class_index.items()),
        )
    )

    return {
        "count": len(prepared),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "classes": labels_present,
        "skipped": skipped,
        "data_yaml": str(data_yaml),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", type=str, default="https://outpost.brock.ventures")
    parser.add_argument("--base-model", type=str, default="yolo11x.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-examples", type=int, default=4,
                         help="Refuse to launch a training run with fewer than this many labeled examples total")
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_OUTPUT_MODEL))
    args = parser.parse_args()

    if YOLO is None:
        print("[!] ultralytics is not installed. pip install ultralytics torch torchvision")
        sys.exit(1)
    if Image is None:
        print("[!] Pillow is not installed. pip install pillow")
        sys.exit(1)

    print(f"[*] Fetching review manifest from {args.api_url} ...")
    manifest = fetch_manifest(args.api_url)
    print(f"[*] {len(manifest)} accept/relabel review(s) available.")

    stats = build_dataset(manifest, args.api_url, DATASET_DIR)
    print(f"[*] Dataset built: {stats}")

    if stats["count"] < args.min_examples:
        print(
            f"[!] Only {stats['count']} usable labeled example(s) (skipped {stats.get('skipped', 0)}) "
            f"-- below --min-examples={args.min_examples}. Refusing to train: a fine-tune run on this "
            f"little data wouldn't teach the model anything and would just burn GPU time and produce a "
            f"checkpoint that looks trained but isn't. Get more /review decisions in and re-run."
        )
        sys.exit(2)

    print(f"[*] Loading base model {args.base_model} ...")
    model = YOLO(args.base_model)

    print(f"[*] Starting fine-tune: {args.epochs} epochs, imgsz={args.imgsz}, device=0 (CUDA)")
    results = model.train(
        data=stats["data_yaml"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=0,
        project=str(BASE_DIR / "finetune_runs"),
        name=f"run_{int(time.time())}",
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"[!] Training finished but no weights found at {best_weights}")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_model)
    shutil.copy(best_weights, output_path)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "example_count": stats["count"],
        "train_count": stats.get("train_count"),
        "val_count": stats.get("val_count"),
        "classes": stats["classes"],
        "run_dir": str(run_dir),
        "output_model": str(output_path),
    }
    (output_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2))

    print(f"[*] Fine-tuned weights: {output_path}")
    print(f"[*] Metadata: {output_path.with_suffix('.json')}")
    print(
        "[*] NOT deployed automatically. Review the run's val metrics in "
        f"{run_dir} before pointing any runner at this checkpoint "
        f"(--model {output_path})."
    )


if __name__ == "__main__":
    main()
