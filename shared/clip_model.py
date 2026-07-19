"""
shared/clip_model.py

Fixes drawback #1 and #2 from the analysis: an image's CLIP embedding is
identical no matter which label set (environment / garment / color /
vibe) you compare it against, and a fixed label set's text embedding
never changes across images. Both were previously recomputed redundantly
(image encoded once per axis = ~4x more image forward passes than
needed; label texts re-encoded on every row = thousands of redundant
text-encoder calls for a set of ~20-40 fixed strings).

This module encodes each image/region exactly once and caches every
label-set embedding exactly once at startup, then does classification
via plain matrix multiplication (cosine similarity) against the cached
label embeddings - no repeated CLIP forward passes for the same input.
"""

from __future__ import annotations
import functools
import numpy as np
import torch
import open_clip
from PIL import Image


class ClipModel:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[ClipModel] loaded {model_name}/{pretrained} on {self.device}")

    # ------------------------------------------------------------------
    # Image encoding - batched, single pass per image/crop.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Encode a list of PIL images in batches. Returns (N, D) L2-normalized
        array. This is the ONLY place image encoding happens - callers should
        never re-encode the same image/crop for a different label set.
        """
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 512))

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image])[0]

    # ------------------------------------------------------------------
    # Text encoding - cached per unique string set, computed once.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_texts(self, texts: tuple[str, ...]) -> np.ndarray:
        """Encode a tuple of text strings. Wrapped with lru_cache below so
        that a fixed label set (e.g. the 10 color names) is embedded
        exactly once for the entire run, not once per image.
        """
        tok = self.tokenizer(list(texts)).to(self.device)
        feats = self.model.encode_text(tok)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    @functools.lru_cache(maxsize=32)
    def _cached_label_embeddings(self, texts_key: tuple[str, ...]) -> np.ndarray:
        return self.encode_texts(texts_key)

    def get_label_embeddings(self, labels: list[str]) -> np.ndarray:
        """Public entrypoint - always goes through the cache. Call this
        once per label set at startup (see scripts/enrich_metadata.py),
        not inside a per-image loop.
        """
        return self._cached_label_embeddings(tuple(labels))

    @staticmethod
    def classify_multi_label(image_embedding: np.ndarray, label_embeddings: np.ndarray,
                              labels: list[str], threshold: float = 0.22,
                              top_k_fallback: int = 1) -> list[tuple[str, float]]:
        """Multi-label classification via cosine similarity + threshold,
        replacing single-winner argmax (drawback #4/#5). Returns ALL labels
        whose similarity clears `threshold`, not just the single best one -
        this is what allows an image to be tagged with BOTH "shirt" and
        "tie", or both "red" and "white", when both are actually present.

        If nothing clears the threshold (ambiguous/low-confidence image),
        fall back to the single top match rather than tagging nothing -
        keeps behavior sane on genuinely ambiguous images without forcing
        false-confident multi-labels elsewhere.
        """
        sims = image_embedding @ label_embeddings.T  # (n_labels,)
        order = np.argsort(-sims)
        above = [(labels[i], float(sims[i])) for i in order if sims[i] >= threshold]
        if above:
            return above
        return [(labels[order[0]], float(sims[order[0]]))][:top_k_fallback]
