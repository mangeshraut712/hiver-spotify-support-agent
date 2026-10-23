#!/usr/bin/env python3
"""Download / filter SpotifyCares pairs and write processed artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="SpotifyCares")
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twcs-path", default=None)
    args = parser.parse_args()

    pairs = prepare_dataset(
        brand=args.brand,
        max_pairs=args.max_pairs,
        seed=args.seed,
        twcs_path=args.twcs_path,
    )
    print(f"Prepared {len(pairs)} pairs for {args.brand}")
    print(pairs[["pair_id", "customer_text", "brand_reply_text"]].head(3).to_string())


if __name__ == "__main__":
    main()
