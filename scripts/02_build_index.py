#!/usr/bin/env python3
"""Build TF-IDF retrieval index from processed pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.retrieve import ReplyRetriever, load_pairs_csv
from spotify_agent.paths import GOLDEN_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()

    pairs = load_pairs_csv()
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]
    held_ids = {row["customer_tweet_id"] for row in golden}
    held_text = {row["customer_text"].strip().casefold() for row in golden}
    held_authors = set(pairs.loc[pairs.customer_tweet_id.isin(held_ids), "customer_author_id"])
    # Hold out entire customers before fitting vocabulary or IDF, including exact duplicates.
    pairs = pairs.loc[~pairs.customer_author_id.isin(held_authors)
                      & ~pairs.customer_tweet_id.isin(held_ids)
                      & ~pairs.customer_text.str.strip().str.casefold().isin(held_text)]
    retriever = ReplyRetriever.build(pairs, max_features=args.max_features)
    path = retriever.save()
    print(f"Index built on {len(pairs)} pairs → {path}")


if __name__ == "__main__":
    main()
