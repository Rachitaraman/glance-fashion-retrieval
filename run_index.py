#!/usr/bin/env python
"""Entry point for Part A: build the vector index + structured attribute records.

Requires scripts/enrich_metadata.py to have been run first (produces
data/metadata_enriched.json, which this reads).
"""

from indexer.build_index import main

if __name__ == "__main__":
    main()
