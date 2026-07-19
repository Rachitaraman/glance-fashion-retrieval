"""
scripts/enrich_metadata.py  (REPLACEMENT)

Fixes, relative to the previous version:

1. Each image is encoded ONCE per region (global + neckline + upper_body +
   lower_body = 4 crops), not once per label-axis (was 4x redundant passes
   for the SAME whole image). The image content being encoded now differs
   per pass (different crops), so this compute is doing real work instead
   of redundant work.
2. All label-set text embeddings are computed ONCE at startup via
   ClipModel.get_label_embeddings() (cached), not re-encoded per image row.
3. Garment and color detection are per-REGION and MULTI-LABEL (threshold-
   based, see shared/clip_model.py), not single whole-image argmax. This
   is what allows one image to be tagged with tie=red AND shirt=white
   simultaneously, bound to their own regions - the actual fix for
   compositional queries like "a red tie and a white shirt."
4. Environment and vibe remain whole-image, single-label-ish (top match,
   since these are scene-level properties without a natural region split) -
   but still use the shared cached embeddings and single image encode.
5. Output is structured JSON per image (not a single flattened caption
   string), so the retriever can do explicit per-region attribute matching
   (late fusion) instead of relying on one blended embedding (early fusion)
   that can no longer be pulled apart at query time.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.clip_model import ClipModel
from shared.config import load_config, resolve_path


def crop_region(img: Image.Image, box: list[float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def main():
    config = load_config()
    garment_labels = config["garment_labels"]
    color_labels = config["color_labels"]
    environment_labels = config["environment_labels"]
    vibe_labels = config["vibe_labels"]
    regions = config["regions"]
    threshold = config.get("confidence_threshold", 0.22)

    images_dir = resolve_path(config["dataset"]["images_dir"])
    metadata_path = resolve_path(config["dataset"]["metadata_path"])

    clip = ClipModel()

    # Compute every label-set embedding EXACTLY ONCE, before the image loop.
    # This is the fix for drawback #2 (label sets re-encoded on every row).
    garment_emb = clip.get_label_embeddings(garment_labels)
    color_emb = clip.get_label_embeddings(color_labels)
    env_emb = clip.get_label_embeddings(environment_labels)
    vibe_emb = clip.get_label_embeddings(vibe_labels)

    df = pd.read_csv(metadata_path)

    enriched_rows = []
    for _, row in df.iterrows():
        # build_metadata.py writes an "image_path" column (relative to images_dir).
        rel_path = row["image_path"]
        img_path = images_dir / rel_path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"skip {img_path}: {e}")
            continue

        # --- Whole-image passes: environment + vibe (scene-level, no region split needed) ---
        global_emb = clip.encode_image(img)  # ONE encode for the whole image
        env_label = clip.classify_multi_label(global_emb, env_emb, environment_labels,
                                               threshold=threshold, top_k_fallback=1)[0][0]
        vibe_label = clip.classify_multi_label(global_emb, vibe_emb, vibe_labels,
                                                threshold=threshold, top_k_fallback=1)[0][0]

        # --- Region passes: garment + color, bound per-region, multi-label ---
        region_attributes = {}
        for region_name, box in regions.items():
            crop = crop_region(img, box)
            region_emb = clip.encode_image(crop)  # ONE encode per region crop

            garments = clip.classify_multi_label(region_emb, garment_emb, garment_labels,
                                                  threshold=threshold)
            colors = clip.classify_multi_label(region_emb, color_emb, color_labels,
                                                threshold=threshold, top_k_fallback=1)

            region_attributes[region_name] = {
                "garments": [{"label": g, "score": round(s, 4)} for g, s in garments],
                "colors": [{"label": c, "score": round(s, 4)} for c, s in colors],
            }

        enriched_rows.append({
            "filename": rel_path,
            "environment": env_label,
            "vibe": vibe_label,
            "regions": region_attributes,
        })

        if len(enriched_rows) % 200 == 0:
            print(f"enriched {len(enriched_rows)}/{len(df)}")

    out_path = resolve_path(config["dataset"]["images_dir"]).parent / "metadata_enriched.json"
    with open(out_path, "w") as f:
        json.dump(enriched_rows, f, indent=2)
    print(f"Wrote {len(enriched_rows)} enriched records -> {out_path}")


if __name__ == "__main__":
    main()
