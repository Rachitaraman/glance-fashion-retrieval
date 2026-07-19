# Glance ML Internship Assignment: Multimodal Fashion & Context Retrieval

An intelligent search engine that retrieves fashion images from a database using natural language queries. The system understands **what** someone is wearing (and where on their body), **where** they are, and the overall **vibe** of their attire — with garment/color attributes bound to image regions rather than blended into one vector, so compositional queries like *"a red tie and a white shirt"* don't lose track of which color belongs to which garment.

## Architecture

```
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
│  Your images     │────▶│  Region-based CLIP    │────▶│  Part A: Indexer │
│  (data/images/)  │     │  multi-label enrich   │     │  FAISS (image    │
└─────────────────┘     │  (neckline/upper/     │     │  embeddings only)│
                         │   lower/global crops) │     └────────┬─────────┘
                         └───────────┬───────────┘              │
                                     │ metadata_enriched.json    │
                                     ▼                           ▼
                              ┌───────────────────────────────────┐
                              │  Part B: Retriever                 │
                              │  1. CLIP top-N recall (FAISS)      │
                              │  2. Parse query → (garment,color)  │
                              │  3. Late-fuse: α·clip + β·attr_match│
                              └───────────────────────────────────┘
```

### Part A — Indexer (`indexer/`)
- **Feature extraction**: CLIP ViT-B-32 (`open_clip`, openai checkpoint) converts images into 512-d embeddings, used purely for broad semantic/style recall
- **Vector storage**: FAISS `IndexFlatIP` (cosine similarity via normalized vectors) in `index_store/vectors.faiss`
- Structured per-region attributes (from the enrichment step) are stored alongside as `index_store/records.json`, kept separate from the embedding rather than blended into it — this is what lets the retriever bind a queried color to a specific garment at search time

### Part B — Retriever (`retriever/`)
- Accepts natural language queries like `"a red tie and a white shirt in a formal setting"`
- **Query parsing**: window-anchored parser extracts `(garment, color)` pairs — for each garment word, looks back a few words for a color modifier, so "red tie and white shirt" binds `red→tie` and `white→shirt` instead of collapsing into an unordered bag of words
- **Late fusion**: CLIP similarity pulls a broad top-N candidate pool, then each candidate is re-ranked by `score = α·clip_similarity + β·attribute_match`, where `attribute_match` checks whether any of the image's detected regions has the queried garment *and* color together. Queries with no explicit garment/color get `attribute_match = 0` (not penalized), so style-only queries degrade gracefully to pure CLIP ranking.

## Metadata Enrichment (`scripts/enrich_metadata.py`)

Each image is encoded once per crop — a `global` pass for environment/vibe (scene-level properties), plus `neckline` / `upper_body` / `lower_body` region crops for garment + color (see `regions:` in `config.yaml`). Garment and color are **multi-label** (threshold-based, not single-winner argmax) *per region*, so one image can be tagged `neckline: {tie, red}` and `upper_body: {shirt, white}` simultaneously — this is what makes compositional queries representable at all, not just plausible-sounding.

All label-set text embeddings (garments, colors, environments, vibes) are computed **once** at startup and cached, and each image/crop is encoded **once** — not re-encoded per label axis — so enrichment scales roughly linearly with dataset size rather than with (images × axes).

## Dataset

- **Size**: 3,200 images (your own dataset — see Quick Start Option A below)
- **Axes of variation**:
  - **Environment**: office, urban street, park, home (CLIP zero-shot, whole-image)
  - **Garments**: shirt, t-shirt, blazer, hoodie, jacket/coat, dress, skirt, pants, tie, scarf, hat, handbag, sunglasses (CLIP zero-shot, multi-label, per-region)
  - **Color**: 14 color classes (CLIP zero-shot, multi-label, per-region)
  - **Vibe**: professional, casual, streetwear, cozy (CLIP zero-shot, whole-image)

## Quick Start

### 1. Setup

```bash
cd glance-fashion-retrieval
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A — use your own images (e.g. the 3200-image set):**

```bash
# 1. Copy/move your images into data/images/ (subfolders OK)
# 2. Build the metadata template by scanning that folder
python scripts/build_metadata.py

# 3. Enrich with region-based multi-label CLIP zero-shot detection
python scripts/enrich_metadata.py
```

**Option B — auto-download Fashionpedia:**

```bash
python scripts/download_dataset.py
python scripts/build_metadata.py
python scripts/enrich_metadata.py
```

### 3. Build Index (Part A)

```bash
python run_index.py
```

### 4. Search (Part B)

```bash
python run_search.py "a red tie and a white shirt in a formal setting"
python run_search.py "professional business attire inside a modern office" -k 10
python run_search.py "casual weekend outfit for a city walk" -k 5
```

## Project Structure

```
glance-fashion-retrieval/
├── indexer/
│   └── build_index.py        # Part A: CLIP embeddings -> FAISS index
├── retriever/
│   └── search.py             # Part B: query parsing + late-fusion re-ranking
├── scripts/
│   ├── download_dataset.py   # Optional: Fashionpedia auto-download
│   ├── build_metadata.py     # Scan data/images/ -> data/metadata.csv template
│   └── enrich_metadata.py    # Region-based multi-label CLIP zero-shot enrichment
├── shared/
│   ├── clip_model.py         # open_clip wrapper: batched encode + cached label embeddings
│   └── config.py
├── data/
│   ├── images/                    # Your images (gitignored)
│   ├── metadata.csv               # image_id, image_path (+ legacy unused columns)
│   └── metadata_enriched.json     # Per-image, per-region structured attributes (gitignored)
├── index_store/               # FAISS index + attribute records (gitignored)
│   ├── vectors.faiss
│   └── records.json
├── config.yaml                # Label vocab, region crops, fusion weights (α/β)
├── run_index.py
├── run_search.py
└── requirements.txt
```

## Example Queries

| Query | What it tests |
|-------|---------------|
| `"a red tie and a white shirt in a formal setting"` | Compositional: two garment+color pairs bound to distinct regions |
| `"professional business attire inside a modern office"` | Vibe + environment (no explicit garment/color — pure CLIP + scene tags) |
| `"casual weekend outfit for a city walk"` | Style inference — attribute score is neutral (0), CLIP ranking alone drives it |
| `"a person in a bright yellow raincoat"` | Single garment+color pair, straightforward attribute fusion |

## ML Design Notes

- **No filename keyword matching** — all retrieval is embedding + structured-attribute based
- **Late fusion, not early fusion** — an earlier version of this pipeline blended a generated caption into the image embedding before indexing (early fusion); that destroys the very structure needed to bind a color to a specific garment. This version keeps the CLIP embedding and the structured per-region attributes separate until query time, where they're explicitly fused per candidate.
- **Region heuristic, not real segmentation** — `neckline`/`upper_body`/`lower_body` are fixed fractional crops (see `config.yaml`), not a trained segmentation model. Works well on centered, full-body photos; less reliable on close crops or unusual poses. A trained garment segmentation model (e.g. SegFormer/U2Net fine-tuned on Fashionpedia) is the natural upgrade path — see the assignment write-up's future work section.
- **Simple vector DB** — FAISS `IndexFlatIP` chosen for zero-setup local indexing (per assignment guidance); the structured attribute matching is intentionally simple (threshold + set lookup) so effort stays on retrieval logic rather than infra.

## Requirements

- Python 3.10+
- ~2 GB disk space for images + index
- GPU optional (CPU works, slower for enrichment — each image is encoded 4x: once per region crop + once globally)

## License

Code: MIT.
