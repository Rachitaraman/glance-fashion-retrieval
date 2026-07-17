from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.clip_model import CLIPEncoder
from shared.config import load_config


def build_caption(row: pd.Series) -> str:
    """Compose a rich caption from structured metadata for optional hybrid indexing."""
    parts = [
        f"a person wearing {row.get('color', 'unknown')} {row.get('clothing_type', 'clothing')}",
        f"in a {row.get('environment', 'setting')}",
        f"with a {row.get('vibe', 'style')} look",
    ]
    if pd.notna(row.get("fashionpedia_category")):
        parts.append(f"({row['fashionpedia_category']})")
    return ", ".join(parts)


def extract_image_embeddings(
    metadata_csv: Path,
    images_dir: Path,
    model_name: str,
    batch_size: int,
) -> tuple[list[dict], np.ndarray]:
    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"No records found in {metadata_csv}")

    encoder = CLIPEncoder(model_name=model_name)
    records: list[dict] = []
    image_paths: list[Path] = []

    for _, row in df.iterrows():
        rel_path = row["image_path"]
        full_path = images_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not full_path.exists():
            full_path = images_dir / Path(rel_path).name
        if not full_path.exists():
            continue

        record = row.to_dict()
        record["absolute_path"] = str(full_path)
        record["caption"] = build_caption(row)
        records.append(record)
        image_paths.append(full_path)

    if not image_paths:
        raise FileNotFoundError(f"No valid images found under {images_dir}")

    print(f"Extracting CLIP embeddings for {len(image_paths)} images...")
    image_embeddings = encoder.encode_images(image_paths, batch_size=batch_size)

    # Hybrid index: average image + caption embeddings for richer multi-attribute search
    captions = [record["caption"] for record in records]
    print("Encoding metadata captions for hybrid retrieval...")
    caption_embeddings = encoder.encode_text(captions)
    combined = image_embeddings + caption_embeddings
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.clip(norms, 1e-12, None)

    return records, combined.astype(np.float32)
