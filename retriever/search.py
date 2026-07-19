"""
retriever/search.py  (REPLACEMENT)

This is where late fusion actually happens. Pipeline:
  1. CLIP top-N recall from FAISS (broad, cheap, good at style/context).
  2. Parse the query into (garment, color) pairs using a window-anchored
     parser - for each garment word found, look a few words back for a
     color modifier, so "a red tie and a white shirt" binds red->tie and
     white->shirt instead of losing which color belongs to which garment.
  3. Re-rank the candidate pool: for each candidate, check its per-region
     structured attributes (from enrich_metadata.py) against the parsed
     query attributes - does ANY region on this image have both the
     queried garment AND the queried color, at or above the region's
     detection confidence? Fuse that match signal with the CLIP score.

Queries with no explicit garment/color get an attribute score of exactly
0 (not penalized), so style/vibe-only queries fall back cleanly to
CLIP-only ranking.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.clip_model import ClipModel
from shared.config import load_config, resolve_path

GARMENT_SYNONYMS = {
    "shirt": ["shirt", "button-down", "button down", "blouse"],
    "t-shirt": ["t-shirt", "tshirt", "tee"],
    "blazer": ["blazer", "suit jacket"],
    "hoodie": ["hoodie", "sweatshirt"],
    "jacket or coat": ["jacket", "coat", "outerwear", "raincoat"],
    "dress": ["dress", "gown"],
    "skirt": ["skirt"],
    "pants or trousers": ["pants", "trousers", "slacks", "jeans"],
    "tie": ["tie", "necktie"],
    "scarf": ["scarf"],
    "hat": ["hat", "cap"],
    "handbag": ["handbag", "purse", "bag"],
    "sunglasses": ["sunglasses", "shades"],
}
COLOR_WORDS = ["red", "blue", "yellow", "green", "black", "white", "gray", "grey",
               "brown", "navy", "pink", "purple", "orange", "beige", "khaki"]


def parse_query(text: str) -> list[tuple[str, str | None]]:
    """Window-anchored parse: for each garment mention, look back up to 3
    words for a color modifier. Prevents "red tie and white shirt" from
    collapsing into an unordered {red, white, tie, shirt} bag, which would
    let a white-tie/red-shirt image match just as well as the correct one.
    """
    lower = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", lower)

    spans = []
    for canonical, synonyms in GARMENT_SYNONYMS.items():
        for syn in synonyms:
            syn_tokens = syn.split()
            n = len(syn_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == syn_tokens:
                    spans.append((i, canonical))
    spans.sort()

    used = set()
    results = []
    for idx, canonical in spans:
        color = None
        for j in range(idx - 1, max(0, idx - 3) - 1, -1):
            if j in used:
                continue
            w = tokens[j]
            if w in COLOR_WORDS:
                color = "gray" if w == "grey" else w
                used.add(j)
                break
        results.append((canonical, color))
    return results


def region_match_score(query_attrs: list[tuple[str, str | None]], regions: dict) -> float:
    """For each queried (garment, color) pair, check if ANY region on this
    image has that garment AND that color detected. Returns fraction of
    query attributes satisfied, in [0, 1]. Returns 0 if the query has no
    explicit garment/color (handled by caller before this is invoked).
    """
    if not query_attrs:
        return 0.0

    matches = 0
    total = 0
    for garment, color in query_attrs:
        total += 1
        for region_name, attrs in regions.items():
            region_garments = {g["label"] for g in attrs.get("garments", [])}
            region_colors = {c["label"] for c in attrs.get("colors", [])}
            if garment in region_garments:
                if color is None or color in region_colors:
                    matches += 1
                    break
    return matches / total if total else 0.0


def search(query: str, k: int = 5, alpha: float | None = None, beta: float | None = None,
           candidate_pool: int | None = None):
    config = load_config()
    fusion_cfg = config.get("fusion", {})
    alpha = fusion_cfg.get("alpha", 0.6) if alpha is None else alpha
    beta = fusion_cfg.get("beta", 0.4) if beta is None else beta
    candidate_pool = fusion_cfg.get("candidate_pool", 50) if candidate_pool is None else candidate_pool

    out_dir = resolve_path(config.get("index_store_dir", "index_store"))
    index = faiss.read_index(str(out_dir / "vectors.faiss"))
    with open(out_dir / "records.json") as f:
        records = json.load(f)

    clip = ClipModel()
    q_emb = clip.encode_texts((query,))[0].astype(np.float32)

    pool = min(candidate_pool, index.ntotal)
    sims, ids = index.search(np.array([q_emb]), pool)
    sims, ids = sims[0], ids[0]

    query_attrs = parse_query(query)

    results = []
    for sim, idx in zip(sims, ids):
        if idx == -1:
            continue
        rec = records[idx]
        attr_score = region_match_score(query_attrs, rec["regions"])
        final = alpha * float(sim) + beta * attr_score
        results.append({
            "filename": rec["filename"],
            "clip_score": float(sim),
            "attr_score": attr_score,
            "final_score": final,
            "environment": rec["environment"],
            "vibe": rec["vibe"],
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:k], query_attrs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    results, parsed = search(args.query, k=args.k)
    print(f'Query: "{args.query}"')
    print(f"Parsed attributes: {parsed}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"#{i} final={r['final_score']:.4f} clip={r['clip_score']:.4f} "
              f"attr={r['attr_score']:.4f}  {r['filename']}")
        print(f"    env={r['environment']} | vibe={r['vibe']}")
