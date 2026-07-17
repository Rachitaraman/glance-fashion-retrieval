from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class CLIPEncoder:
    """Shared CLIP encoder for image and text embeddings."""

    def __init__(self, model_name: str = "clip-ViT-B-32", device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_images(self, image_paths: Iterable[str | Path], batch_size: int = 32) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        embeddings = self.model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(images) > batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_text(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def zero_shot_classify(
        self,
        image_path: str | Path,
        labels: list[str],
    ) -> tuple[str, float]:
        image_emb = self.encode_images([image_path])[0]
        label_embs = self.encode_text(labels)
        scores = label_embs @ image_emb
        best_idx = int(np.argmax(scores))
        return labels[best_idx], float(scores[best_idx])
