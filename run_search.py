#!/usr/bin/env python
"""Entry point for Part B: query the fashion retrieval engine.

Uses CLIP for broad top-N recall, then re-ranks with structured
(garment, color) attribute matching parsed from the query and bound to
the image's per-region attributes (see retriever/search.py).
"""

from __future__ import annotations

import argparse

from retriever.search import search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k fashion images from a natural language query."
    )
    parser.add_argument("query", type=str, help='Natural language query, e.g. "blue blazer in office"')
    parser.add_argument("-k", type=int, default=5, help="Number of results to return.")
    args = parser.parse_args()

    results, parsed = search(args.query, k=args.k)

    print(f'\nQuery: "{args.query}"')
    print(f"Parsed attributes: {parsed}")
    print("-" * 72)
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(
            f"#{i}  final={r['final_score']:.4f}  clip={r['clip_score']:.4f}  "
            f"attr={r['attr_score']:.4f}"
        )
        print(f"    image: {r['filename']}")
        print(f"    env={r['environment']} | vibe={r['vibe']}")


if __name__ == "__main__":
    main()
