from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from indexer.feature_extractor import extract_image_embeddings
from shared.config import load_config, resolve_path


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(
    records: list[dict],
    embeddings: np.ndarray,
    output_dir: Path,
    faiss_name: str,
    metadata_name: str,
    embeddings_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(output_dir / faiss_name))
    np.save(output_dir / embeddings_name, embeddings)

    serializable = []
    for idx, record in enumerate(records):
        serializable.append(
            {
                "index_id": idx,
                "image_path": record.get("image_path"),
                "absolute_path": record.get("absolute_path"),
                "environment": record.get("environment"),
                "clothing_type": record.get("clothing_type"),
                "color": record.get("color"),
                "vibe": record.get("vibe"),
                "fashionpedia_category": record.get("fashionpedia_category"),
                "caption": record.get("caption"),
            }
        )

    with (output_dir / metadata_name).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    print(f"Saved FAISS index with {len(records)} vectors to {output_dir}")


def run_indexer() -> None:
    config = load_config()
    dataset_cfg = config["dataset"]
    index_cfg = config["index"]

    metadata_csv = resolve_path(dataset_cfg["metadata_path"])
    images_dir = resolve_path(dataset_cfg["images_dir"])
    output_dir = resolve_path(index_cfg["output_dir"])

    records, embeddings = extract_image_embeddings(
        metadata_csv=metadata_csv,
        images_dir=images_dir,
        model_name=config["clip_model"],
        batch_size=index_cfg["batch_size"],
    )

    save_index(
        records=records,
        embeddings=embeddings,
        output_dir=output_dir,
        faiss_name=index_cfg["faiss_index_name"],
        metadata_name=index_cfg["metadata_name"],
        embeddings_name=index_cfg["embeddings_name"],
    )


if __name__ == "__main__":
    run_indexer()
