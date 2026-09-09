"""Path helpers rooted at the repo."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
EVAL_DIR = REPO_ROOT / "eval"
GOLDEN_DIR = EVAL_DIR / "golden"
RESULTS_DIR = REPO_ROOT / "results"
PROMPTS_DIR = REPO_ROOT / "prompts"
CACHE_DIR = RESULTS_DIR / "cache"

PAIRS_PATH = PROCESSED_DIR / "spotify_pairs.parquet"
PAIRS_CSV_PATH = PROCESSED_DIR / "spotify_pairs.csv"
INDEX_DIR = PROCESSED_DIR / "index"
GOLDEN_PATH = GOLDEN_DIR / "golden.jsonl"
HEADLINE_PATH = RESULTS_DIR / "headline.json"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, GOLDEN_DIR, RESULTS_DIR, CACHE_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
