"""
Enrich dataset metadata with CLIP zero-shot labels for environment, clothing, color, and vibe.

This satisfies the assignment's multi-axis dataset requirement:
  - Environment: office, urban street, park, home
  - Clothing: formal, casual, outerwear
  - Color: diverse garment colors
  - Vibe: overall style context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from shared.clip_model import CLIPEncoder
from shared.config import load_config, resolve_path


ENVIRONMENT_LABELS = {
    "office interior workplace": "office interior",
    "urban street city outdoors": "urban street",
    "park nature outdoors": "park outdoors",
    "home interior living room": "home interior",
}

VIBE_LABELS = {
    "professional formal attire": "professional formal",
    "casual relaxed style": "casual relaxed",
    "trendy streetwear style": "trendy streetwear",
    "cozy comfortable style": "cozy comfortable",
}


def enrich_row(encoder: CLIPEncoder, image_path: Path, enrichment_cfg: dict, row: pd.Series) -> dict:
    environment_raw, _ = encoder.zero_shot_classify(image_path, enrichment_cfg["environments"])
    clothing_type, _ = encoder.zero_shot_classify(image_path, enrichment_cfg["clothing_types"])
    color_label, _ = encoder.zero_shot_classify(image_path, enrichment_cfg["colors"])
    vibe_raw, _ = encoder.zero_shot_classify(image_path, enrichment_cfg["vibes"])

    color = color_label.replace(" clothing", "")

    hint = row.get("clothing_type_hint", "")
    if hint and "outerwear" in str(hint):
        clothing_type = "outerwear jacket or coat"
    elif hint and "formal" in str(hint):
        clothing_type = "formal blazer or button-down shirt"

    return {
        "environment": ENVIRONMENT_LABELS.get(environment_raw, environment_raw),
        "clothing_type": clothing_type,
        "color": color,
        "vibe": VIBE_LABELS.get(vibe_raw, vibe_raw),
    }


def enrich_metadata(metadata_path: Path, images_dir: Path, enrichment_cfg: dict, model_name: str) -> None:
    df = pd.read_csv(metadata_path)
    encoder = CLIPEncoder(model_name=model_name)

    enriched_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching metadata"):
        image_path = images_dir / row["image_path"]
        if not image_path.exists():
            continue
        labels = enrich_row(encoder, image_path, enrichment_cfg, row)
        updated = row.to_dict()
        updated.update(labels)
        enriched_rows.append(updated)

    pd.DataFrame(enriched_rows).to_csv(metadata_path, index=False)
    print(f"Enriched metadata saved to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich metadata with CLIP zero-shot labels.")
    args = parser.parse_args()

    config = load_config()
    metadata_path = resolve_path(config["dataset"]["metadata_path"])
    images_dir = resolve_path(config["dataset"]["images_dir"])

    enrich_metadata(
        metadata_path=metadata_path,
        images_dir=images_dir,
        enrichment_cfg=config["enrichment"],
        model_name=config["clip_model"],
    )


if __name__ == "__main__":
    main()
