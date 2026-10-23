"""TF-IDF retrieval over historical customer→brand pairs."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from spotify_agent.paths import INDEX_DIR, PROCESSED_DIR, ensure_dirs


@dataclass
class RetrievedExample:
    pair_id: str
    customer_text: str
    brand_reply_text: str
    score: float


class ReplyRetriever:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: Any,
        corpus: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.corpus = corpus.reset_index(drop=True)
        self.params = params or {}

    @classmethod
    def build(
        cls,
        pairs: pd.DataFrame,
        max_features: int = 30_000,
        min_df: int | None = None,
    ) -> ReplyRetriever:
        corpus = pairs.copy()
        corpus["customer_text"] = corpus["customer_text"].fillna("").astype(str)
        corpus["brand_reply_text"] = corpus["brand_reply_text"].fillna("").astype(str)
        corpus = corpus[
            corpus["customer_text"].str.len().gt(5) & corpus["brand_reply_text"].str.len().gt(5)
        ].reset_index(drop=True)
        if corpus.empty:
            raise ValueError("Cannot build a retrieval index from an empty corpus.")
        if min_df is None:
            min_df = 2 if len(corpus) >= 8 else 1
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=min_df,
            stop_words="english",
        )
        try:
            matrix = vectorizer.fit_transform(corpus["customer_text"])
        except ValueError:
            if min_df == 1:
                raise
            min_df = 1
            vectorizer.set_params(min_df=min_df)
            matrix = vectorizer.fit_transform(corpus["customer_text"])
        if matrix.shape[1] == 0:
            raise ValueError("Retrieval index has no terms after TF-IDF.")
        params = {
            "max_features": max_features,
            "min_df": min_df,
            "ngram_range": [1, 2],
            "stop_words": "english",
        }
        return cls(vectorizer=vectorizer, matrix=matrix, corpus=corpus, params=params)

    def save(self, directory: Path | None = None) -> Path:
        """Persist the corpus and vectorizer settings. The index is refit on load.

        Model files are JSON and CSV, not pickle, so loading an index cannot
        execute code from the index directory.
        """
        ensure_dirs()
        directory = directory or INDEX_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.corpus.to_csv(directory / "corpus.csv", index=False)
        meta = {"size": len(self.corpus), "vectorizer": "tfidf", "params": self.params}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> ReplyRetriever:
        directory = directory or INDEX_DIR
        meta_path = directory / "meta.json"
        corpus_path = directory / "corpus.csv"
        if not meta_path.is_file() or not corpus_path.is_file():
            raise FileNotFoundError(
                f"Missing retrieval index in {directory}. Run scripts/02_build_index.py."
            )
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid retrieval metadata in {meta_path}.")
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        corpus = pd.read_csv(corpus_path, dtype=str)
        min_df = int(params["min_df"]) if "min_df" in params else 2
        max_features = int(params.get("max_features", 30_000))
        return cls.build(corpus, max_features=max_features, min_df=min_df)

    def search(
        self,
        query: str,
        k: int = 5,
        exclude_pair_ids: Sequence[str] | None = None,
    ) -> list[RetrievedExample]:
        exclude = {str(item) for item in (exclude_pair_ids or [])}
        if k <= 0 or self.matrix.shape[0] == 0:
            return []
        query_text = "" if query is None else str(query)
        if not query_text.strip():
            return []
        transformed = self.vectorizer.transform([query_text])
        sims = cosine_similarity(transformed, self.matrix).ravel()
        order = sims.argsort()[::-1]
        results: list[RetrievedExample] = []
        for idx in order:
            position = int(idx)
            score = float(sims[position])
            if not math.isfinite(score) or score <= 0:
                continue
            row = self.corpus.iloc[position]
            pair_id = str(row.get("pair_id", position))
            customer_tweet_id = str(row.get("customer_tweet_id", ""))
            if pair_id in exclude or customer_tweet_id in exclude:
                continue
            results.append(
                RetrievedExample(
                    pair_id=pair_id,
                    customer_text=str(row["customer_text"]),
                    brand_reply_text=str(row["brand_reply_text"]),
                    score=score,
                )
            )
            if len(results) >= k:
                break
        return results


def load_pairs_csv() -> pd.DataFrame:
    path = PROCESSED_DIR / "spotify_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/01_prepare_data.py first.")
    return pd.read_csv(path, dtype=str)
