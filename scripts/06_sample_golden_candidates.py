#!/usr/bin/env python3
"""Stratified candidate sampling for the golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.intents import INTENT_KEYWORDS, keyword_intent
from spotify_agent.paths import GOLDEN_DIR, ensure_dirs
from spotify_agent.retrieve import load_pairs_csv


def assign_stratum(text: str) -> str:
    return keyword_intent(text)


def sample_stratified(
    pairs: pd.DataFrame,
    n: int = 220,
    seed: int = 42,
    per_stratum: int = 24,
) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["customer_text"] = pairs["customer_text"].fillna("").astype(str)
    pairs["brand_reply_text"] = pairs["brand_reply_text"].fillna("").astype(str)
    pairs = pairs[pairs["customer_text"].str.len().gt(5)]
    pairs["stratum"] = pairs["customer_text"].map(assign_stratum)
    rng = seed
    chunks = []
    for stratum, group in pairs.groupby("stratum"):
        take = min(per_stratum, len(group))
        chunks.append(group.sample(n=take, random_state=rng))
        rng += 1
    sampled = pd.concat(chunks, ignore_index=True)
    if len(sampled) < n:
        remaining = pairs[~pairs["pair_id"].isin(sampled["pair_id"])]
        need = min(n - len(sampled), len(remaining))
        if need:
            sampled = pd.concat(
                [sampled, remaining.sample(n=need, random_state=seed + 99)],
                ignore_index=True,
            )
    elif len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed).reset_index(drop=True)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-stratum", type=int, default=24)
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR / "candidates.jsonl")
    args = parser.parse_args()

    ensure_dirs()
    pairs = load_pairs_csv()
    sampled = sample_stratified(
        pairs, n=args.n, seed=args.seed, per_stratum=args.per_stratum
    )
    with args.out.open("w") as f:
        for _, row in sampled.iterrows():
            rec = {
                "id": row["pair_id"],
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "brand_reply_text": row["brand_reply_text"],
                "stratum": row["stratum"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sampled)} candidates → {args.out}")
    print(sampled["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
