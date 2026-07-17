from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from shared.clip_model import CLIPEncoder
from shared.config import load_config, resolve_path


@dataclass
class SearchResult:
    rank: int
    score: float
    image_path: str
    absolute_path: str
    environment: str
    clothing_type: str
    color: str
    vibe: str
    caption: str


class FashionRetriever:
    """Retrieve top-k fashion images from a natural language query."""

    def __init__(
        self,
        index_dir: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        config = load_config()
        index_cfg = config["index"]

        self.index_dir = index_dir or resolve_path(index_cfg["output_dir"])
        self.model_name = model_name or config["clip_model"]
        self.encoder = CLIPEncoder(model_name=self.model_name)

        index_path = self.index_dir / index_cfg["faiss_index_name"]
        metadata_path = self.index_dir / index_cfg["metadata_name"]

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found in {self.index_dir}. Run `python run_index.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: list[dict] = json.load(handle)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query must be a non-empty string.")

        query_emb = self.encoder.encode_text(query)
        scores, indices = self.index.search(query_emb, top_k)

        results: list[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    image_path=meta.get("image_path", ""),
                    absolute_path=meta.get("absolute_path", ""),
                    environment=meta.get("environment", ""),
                    clothing_type=meta.get("clothing_type", ""),
                    color=meta.get("color", ""),
                    vibe=meta.get("vibe", ""),
                    caption=meta.get("caption", ""),
                )
            )
        return results

    def print_results(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = self.search(query, top_k=top_k)
        print(f'\nQuery: "{query}"')
        print("-" * 72)
        if not results:
            print("No results found.")
            return results

        for result in results:
            print(
                f"#{result.rank}  score={result.score:.4f}  "
                f"{result.color} | {result.clothing_type} | {result.environment} | {result.vibe}"
            )
            print(f"    image: {result.image_path}")
            print(f"    caption: {result.caption}")
        return results
