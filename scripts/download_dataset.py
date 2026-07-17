"""
Download Fashionpedia images from Hugging Face and build a 500–1000 image dataset.

The assignment requires variation across environment, clothing type, and color.
Fashionpedia provides clothing categories/attributes; environment and vibe are
enriched later via CLIP zero-shot labeling in enrich_metadata.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from shared.config import load_config, resolve_path


FASHIONPEDIA_CATEGORY_MAP = {
    "shirt, blouse": "formal/casual top",
    "top, t-shirt, sweatshirt": "casual top",
    "sweater": "casual knitwear",
    "cardigan": "casual outer layer",
    "jacket": "outerwear jacket",
    "vest": "outerwear vest",
    "pants": "formal/casual pants",
    "shorts": "casual shorts",
    "skirt": "formal/casual skirt",
    "coat": "outerwear coat",
    "dress": "formal/casual dress",
    "jumpsuit": "formal/casual jumpsuit",
    "cape": "outerwear cape",
    "glasses": "accessory",
    "hat": "accessory",
    "headband, head covering, hair accessory": "accessory",
    "tie": "formal accessory",
    "glove": "accessory",
    "watch": "accessory",
    "belt": "accessory",
    "leg warmer": "accessory",
    "tights, stockings": "accessory",
    "sock": "accessory",
    "shoe": "footwear",
    "bag, wallet": "accessory",
    "scarf": "outerwear scarf",
    "umbrella": "accessory",
    "hood": "outerwear hood",
    "collar": "garment detail",
    "lapel": "formal detail",
    "sleeve": "garment detail",
    "pocket": "garment detail",
    "neckline": "garment detail",
}


def infer_clothing_bucket(category: str) -> str:
    lowered = category.lower()
    if any(word in lowered for word in ("jacket", "coat", "cape", "scarf", "hood", "outerwear")):
        return "outerwear jacket or coat"
    if any(word in lowered for word in ("blazer", "shirt", "blouse", "tie", "lapel", "dress", "jumpsuit")):
        return "formal blazer or button-down shirt"
    if any(word in lowered for word in ("t-shirt", "sweatshirt", "hoodie", "shorts", "top")):
        return "casual hoodie or t-shirt"
    return "casual hoodie or t-shirt"


def download_fashionpedia(target_size: int, images_dir: Path, metadata_path: Path, seed: int) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Fashionpedia from Hugging Face (detection-datasets/fashionpedia)...")
    dataset = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)

    # objects.category is an integer ClassLabel; resolve it to its string name
    # so FASHIONPEDIA_CATEGORY_MAP / infer_clothing_bucket keyword matching works.
    category_names = dataset.features["objects"].feature["category"].names

    rng = random.Random(seed)
    records: list[dict] = []
    seen_image_ids: set[int] = set()

    progress = tqdm(total=target_size, desc="Downloading images")
    for sample in dataset:
        image_id = int(sample["image_id"])
        if image_id in seen_image_ids:
            continue

        objects = sample.get("objects") or {}
        categories = objects.get("category") or []
        if not categories:
            continue

        primary_category = categories[0]
        if isinstance(primary_category, int):
            category_name = category_names[primary_category]
        else:
            category_name = str(primary_category)

        image = sample["image"]
        if not isinstance(image, Image.Image):
            continue

        filename = f"fashionpedia_{image_id:06d}.jpg"
        save_path = images_dir / filename
        image.convert("RGB").save(save_path, quality=92)

        mapped_category = FASHIONPEDIA_CATEGORY_MAP.get(category_name, category_name)
        records.append(
            {
                "image_id": image_id,
                "image_path": filename,
                "fashionpedia_category": mapped_category,
                "clothing_type_hint": infer_clothing_bucket(mapped_category),
                "environment": "",
                "clothing_type": "",
                "color": "",
                "vibe": "",
            }
        )
        seen_image_ids.add(image_id)
        progress.update(1)

        if len(records) >= target_size:
            break

    progress.close()

    if len(records) < target_size:
        raise RuntimeError(
            f"Only collected {len(records)} of {target_size} requested images. "
            "Check your internet connection and Hugging Face dataset availability."
        )

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    print(f"Saved {len(df)} images to {images_dir}")
    print(f"Saved metadata template to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fashionpedia subset for indexing.")
    parser.add_argument("--size", type=int, default=None, help="Number of images to download.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    config = load_config()
    target_size = args.size or config["dataset"]["target_size"]
    images_dir = resolve_path(config["dataset"]["images_dir"])
    metadata_path = resolve_path(config["dataset"]["metadata_path"])

    download_fashionpedia(
        target_size=target_size,
        images_dir=images_dir,
        metadata_path=metadata_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
