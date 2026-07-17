#!/usr/bin/env python
"""Entry point for Part B: query the fashion retrieval engine."""

from __future__ import annotations

import argparse

from retriever.search import FashionRetriever
from shared.config import load_config

def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Retrieve top-k fashion images from a natural language query."
    )
    parser.add_argument("query", type=str, help='Natural language query, e.g. "blue blazer in office"')
    parser.add_argument(
        "-k",
        type=int,
        default=config["retrieval"]["default_top_k"],
        help="Number of results to return.",
    )
    args = parser.parse_args()

    retriever = FashionRetriever()
    retriever.print_results(args.query, top_k=args.k)


if __name__ == "__main__":
    main()
