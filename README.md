# Glance ML Internship Assignment: Multimodal Fashion & Context Retrieval

An intelligent search engine that retrieves fashion images from a database using natural language queries. The system understands **what** someone is wearing, **where** they are, and the overall **vibe** of their attire.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Fashionpedia   │────▶│  CLIP Enrichment │────▶│  Part A: Indexer│
│  Images (800)   │     │  (env/color/vibe)│     │  FAISS Vector DB│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Part B: Retriever◀─────────────┘
                        │  NL Query → Top-k │
                        └──────────────────┘
```

### Part A — Indexer (`indexer/`)
- **Feature extraction**: CLIP (`clip-ViT-B-32`) converts images into 512-d embeddings
- **Hybrid indexing**: Combines image embeddings with auto-generated captions (color + clothing + environment + vibe) for richer multi-attribute search
- **Vector storage**: FAISS `IndexFlatIP` (cosine similarity via normalized vectors)

### Part B — Retriever (`retriever/`)
- Accepts natural language queries like `"person in blue blazer in office setting"`
- Encodes query with the same CLIP model
- Returns top-k images ranked by semantic similarity

## Dataset

- **Source**: [Fashionpedia](https://fashionpedia.github.io/) via Hugging Face (`detection-datasets/fashionpedia`)
- **Size**: 800 images (configurable, within the 500–1,000 requirement)
- **Axes of variation**:
  - **Environment**: office, urban street, park, home (CLIP zero-shot)
  - **Clothing**: formal, casual, outerwear (Fashionpedia categories + CLIP)
  - **Color**: 10 color classes (CLIP zero-shot)
  - **Vibe**: professional, casual, streetwear, cozy (CLIP zero-shot)

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

# 3. Enrich with CLIP zero-shot labels (environment/clothing/color/vibe)
python scripts/enrich_metadata.py
```

**Option B — auto-download Fashionpedia (~800 images):**

```bash
python scripts/download_dataset.py
python scripts/enrich_metadata.py
```

To use a smaller subset for testing:

```bash
python scripts/download_dataset.py --size 100
python scripts/enrich_metadata.py
```

### 3. Build Index (Part A)

```bash
python run_index.py
```

### 4. Search (Part B)

```bash
python run_search.py "person wearing blue formal blazer in office interior"
python run_search.py "casual red hoodie in urban street with streetwear vibe" -k 10
python run_search.py "outerwear coat in park outdoors" -k 5
```

## Project Structure

```
glance-fashion-retrieval/
├── indexer/                  # Part A: feature extraction + FAISS index
│   ├── feature_extractor.py
│   └── build_index.py
├── retriever/                # Part B: natural language search
│   └── search.py
├── scripts/
│   ├── download_dataset.py   # Fashionpedia download
│   └── enrich_metadata.py    # CLIP zero-shot labeling
├── shared/
│   ├── clip_model.py         # Shared CLIP encoder
│   └── config.py
├── data/
│   ├── images/               # Downloaded images (gitignored)
│   └── metadata.csv          # Image metadata
├── outputs/index/            # FAISS index + metadata (gitignored)
├── config.yaml
├── run_index.py
├── run_search.py
└── requirements.txt
```

## Example Queries

| Query | What it tests |
|-------|---------------|
| `"blue blazer in office setting"` | Color + clothing + environment |
| `"casual hoodie urban street"` | Clothing type + location |
| `"professional formal attire at home"` | Vibe + environment |
| `"red outerwear jacket in park"` | Color + outerwear + environment |

## ML Design Notes

- **No filename keyword matching** — all retrieval is embedding-based via CLIP
- **Multimodal alignment** — CLIP's shared embedding space enables text→image search
- **Hybrid embeddings** — averaging image + structured caption embeddings improves multi-attribute queries without complex engineering
- **Simple vector DB** — FAISS chosen for zero-setup local indexing (per assignment guidance)

## Requirements

- Python 3.10+
- ~2 GB disk space for images + index
- GPU optional (CPU works, slower for enrichment/indexing)

## License

Dataset: [Fashionpedia](https://fashionpedia.github.io/) (CVDF). Code: MIT.
