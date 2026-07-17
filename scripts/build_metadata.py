"""
Build a metadata.csv template from a local folder of images.

Use this instead of download_dataset.py when you're supplying your own
image dataset (e.g. the 3200-image set) rather than pulling from
Fashionpedia/HuggingFace. It scans `data/images/`, assigns a stable
image_id to each file, and writes the empty enrichment columns that
enrich_metadata.py fills in next.

Usage:
    python scripts/build_metadata.py
    python scripts/build_metadata.py --images-dir data/images --out data/metadata.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from shared.config import load_config, resolve_path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_metadata_from_folder(images_dir: Path, metadata_path: Path) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(
            f"No images found under {images_dir}. "
            f"Supported extensions: {sorted(VALID_EXTENSIONS)}"
        )

    records = []
    for image_id, path in enumerate(image_files):
        rel_path = path.relative_to(images_dir).as_posix()
        records.append(
            {
                "image_id": image_id,
                "image_path": rel_path,
                "fashionpedia_category": "",   # optional, unused for custom datasets
                "clothing_type_hint": "",      # optional, unused for custom datasets
                "environment": "",
                "clothing_type": "",
                "color": "",
                "vibe": "",
            }
        )

    df = pd.DataFrame(records)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    print(f"Found {len(df)} images under {images_dir}")
    print(f"Saved metadata template to {metadata_path}")
    if len(df) < 500:
        print(f"Warning: assignment asks for 500-1000+ images, found only {len(df)}.")
    print("Next step: python scripts/enrich_metadata.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build metadata.csv from a local folder of images."
    )
    parser.add_argument("--images-dir", type=str, default=None, help="Path to image folder.")
    parser.add_argument("--out", type=str, default=None, help="Path to write metadata.csv.")
    args = parser.parse_args()

    config = load_config()
    images_dir = resolve_path(args.images_dir) if args.images_dir else resolve_path(config["dataset"]["images_dir"])
    metadata_path = resolve_path(args.out) if args.out else resolve_path(config["dataset"]["metadata_path"])

    build_metadata_from_folder(images_dir, metadata_path)


if __name__ == "__main__":
    main()
