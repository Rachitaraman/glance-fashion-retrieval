"""
indexer/build_index.py  (REPLACEMENT)

Fixes drawback #6 (caption text structurally couldn't represent
compositional queries): the previous version baked garment/color/env/vibe
labels into a single caption string, encoded that caption with CLIP, and
averaged it with the image embedding into ONE blended vector before
indexing (early fusion). That destroys the very structure just extracted
by enrich_metadata.py - once blended, you cannot later ask "does THIS
region match THIS queried attribute."

This version keeps two separate things:
  1. The raw CLIP image embedding, for broad semantic/style recall (FAISS).
  2. The structured per-region attribute JSON from enrich_metadata.py,
     stored as-is in a sidecar file for explicit attribute matching at
     query time (late fusion happens in retriever/search.py, not here).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.clip_model import ClipModel
from shared.config import load_config, resolve_path

EMBED_DIM = 512


def main():
    config = load_config()
    images_dir = resolve_path(config["dataset"]["images_dir"])
    enriched_path = images_dir.parent / "metadata_enriched.json"
    out_dir = resolve_path(config.get("index_store_dir", "index_store"))

    with open(enriched_path) as f:
        enriched = json.load(f)

    clip = ClipModel()
    index = faiss.IndexFlatIP(EMBED_DIM)
    id_to_record = []

    images = []
    for rec in enriched:
        img_path = images_dir / rec["filename"]
        images.append(Image.open(img_path).convert("RGB"))

    # Single batched encode for the whole dataset - not per-region here,
    # since this embedding is purely for broad CLIP recall; the detailed
    # per-region attribute data already lives in `enriched` from the
    # enrichment step and is stored alongside, not re-derived here.
    embeddings = clip.encode_images(images, batch_size=64)
    index.add(embeddings.astype(np.float32))

    for rec in enriched:
        id_to_record.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "vectors.faiss"))
    with open(out_dir / "records.json", "w") as f:
        json.dump(id_to_record, f, indent=2)

    print(f"Indexed {len(id_to_record)} images -> {out_dir}")


if __name__ == "__main__":
    main()
