"""Dataset download, SpotifyCares filtering, and customer→brand pairing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from spotify_agent.paths import PROCESSED_DIR, RAW_DIR, SAMPLE_DIR, ensure_dirs

BRAND = "SpotifyCares"
DEFAULT_SEED = 42
DEFAULT_MAX_PAIRS = 10_000


def find_twcs_csv(explicit: Optional[str] = None) -> Path:
    """Locate twcs.csv via env, explicit path, local cache, or kagglehub."""
    ensure_dirs()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    env_path = os.getenv("TWCS_CSV_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    cached = list(RAW_DIR.rglob("twcs.csv"))
    if cached:
        return cached[0]

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install requirements or set TWCS_CSV_PATH."
        ) from exc

    download_root = Path(
        kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    )
    # Mirror into data/raw for local discoverability.
    matches = list(download_root.rglob("twcs.csv"))
    if not matches:
        raise FileNotFoundError(f"twcs.csv not found under {download_root}")
    src = matches[0]
    dest = RAW_DIR / "twcs.csv"
    if not dest.exists():
        # Symlink when possible to avoid copying ~500MB.
        try:
            dest.symlink_to(src)
        except OSError:
            dest.write_bytes(src.read_bytes())
    return dest


def load_twcs(csv_path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    cols = usecols or [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    return pd.read_csv(csv_path, dtype=str, usecols=cols, low_memory=False)


def build_spotify_pairs(
    twcs: pd.DataFrame,
    brand: str = BRAND,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build customer inbound → first brand reply pairs for one support account."""
    df = twcs.copy()
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])

    brand_tweets = df[df["author_id"] == brand].copy()
    if brand_tweets.empty:
        raise ValueError(f"No tweets found for brand={brand}")

    # Brand replies that point at a customer tweet.
    brand_replies = brand_tweets.dropna(subset=["in_response_to_tweet_id"]).copy()
    brand_replies["in_response_to_tweet_id"] = brand_replies[
        "in_response_to_tweet_id"
    ].astype(str)

    customers = df[df["inbound"]].copy()
    customers = customers.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "text": "customer_text",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
        }
    )

    pairs = brand_replies.merge(
        customers[
            [
                "customer_tweet_id",
                "customer_text",
                "customer_author_id",
                "customer_created_at",
            ]
        ],
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    )

    pairs = pairs.rename(
        columns={
            "tweet_id": "brand_tweet_id",
            "text": "brand_reply_text",
            "created_at": "brand_created_at",
        }
    )

    # Keep one brand reply per customer tweet (earliest reply if timestamps parse).
    pairs["brand_created_at_parsed"] = pd.to_datetime(
        pairs["brand_created_at"], errors="coerce"
    )
    pairs = pairs.sort_values(
        ["customer_tweet_id", "brand_created_at_parsed"], kind="mergesort"
    )
    pairs = pairs.drop_duplicates(subset=["customer_tweet_id"], keep="first")

    pairs = pairs[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "brand_tweet_id",
            "brand_created_at",
            "brand_reply_text",
        ]
    ].dropna(subset=["customer_text", "brand_reply_text"])

    pairs["customer_text"] = pairs["customer_text"].fillna("").astype(str)
    pairs["brand_reply_text"] = pairs["brand_reply_text"].fillna("").astype(str)
    pairs = pairs[
        pairs["customer_text"].str.len().gt(5) & pairs["brand_reply_text"].str.len().gt(5)
    ]

    if len(pairs) > max_pairs:
        pairs = pairs.sample(n=max_pairs, random_state=seed)

    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = pairs.index.map(lambda i: f"sp_{i:05d}")
    return pairs


def save_pairs(pairs: pd.DataFrame) -> Path:
    ensure_dirs()
    csv_path = PROCESSED_DIR / "spotify_pairs.csv"
    pairs.to_csv(csv_path, index=False)
    try:
        pairs.to_parquet(PROCESSED_DIR / "spotify_pairs.parquet", index=False)
    except Exception:
        # parquet is optional if pyarrow/fastparquet missing
        pass
    return csv_path


def write_sample_csv(pairs: pd.DataFrame, n: int = 40) -> Path:
    ensure_dirs()
    sample = pairs.head(n)
    path = SAMPLE_DIR / "sample_spotify.csv"
    sample.to_csv(path, index=False)
    return path


def prepare_dataset(
    brand: str = BRAND,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = DEFAULT_SEED,
    twcs_path: Optional[str] = None,
) -> pd.DataFrame:
    csv_path = find_twcs_csv(twcs_path)
    twcs = load_twcs(csv_path)
    pairs = build_spotify_pairs(twcs, brand=brand, max_pairs=max_pairs, seed=seed)
    save_pairs(pairs)
    write_sample_csv(pairs)
    return pairs
