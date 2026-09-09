"""TF-IDF retrieval over historical customer→brand pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
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
        matrix,
        corpus: pd.DataFrame,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.corpus = corpus.reset_index(drop=True)

    @classmethod
    def build(cls, pairs: pd.DataFrame, max_features: int = 30_000) -> "ReplyRetriever":
        corpus = pairs.copy()
        corpus["customer_text"] = corpus["customer_text"].fillna("").astype(str)
        corpus["brand_reply_text"] = corpus["brand_reply_text"].fillna("").astype(str)
        corpus = corpus[
            corpus["customer_text"].str.len().gt(5)
            & corpus["brand_reply_text"].str.len().gt(5)
        ].reset_index(drop=True)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(corpus["customer_text"])
        return cls(vectorizer=vectorizer, matrix=matrix, corpus=corpus)

    def save(self, directory: Optional[Path] = None) -> Path:
        ensure_dirs()
        directory = directory or INDEX_DIR
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "tfidf.joblib")
        joblib.dump(self.matrix, directory / "matrix.joblib")
        self.corpus.to_csv(directory / "corpus.csv", index=False)
        meta = {"size": len(self.corpus), "vectorizer": "tfidf"}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "ReplyRetriever":
        directory = directory or INDEX_DIR
        vectorizer = joblib.load(directory / "tfidf.joblib")
        matrix = joblib.load(directory / "matrix.joblib")
        corpus = pd.read_csv(directory / "corpus.csv", dtype=str)
        return cls(vectorizer=vectorizer, matrix=matrix, corpus=corpus)

    def search(
        self,
        query: str,
        k: int = 5,
        exclude_pair_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievedExample]:
        exclude = set(exclude_pair_ids or [])
        if k <= 0:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1]
        results: List[RetrievedExample] = []
        for idx in order:
            row = self.corpus.iloc[int(idx)]
            pair_id = str(row.get("pair_id", idx))
            if pair_id in exclude or str(row.get("customer_tweet_id", "")) in exclude:
                continue
            if sims[int(idx)] <= 0:
                continue
            results.append(
                RetrievedExample(
                    pair_id=pair_id,
                    customer_text=str(row["customer_text"]),
                    brand_reply_text=str(row["brand_reply_text"]),
                    score=float(sims[int(idx)]),
                )
            )
            if len(results) >= k:
                break
        return results


def load_pairs_csv() -> pd.DataFrame:
    path = PROCESSED_DIR / "spotify_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/01_prepare_data.py first."
        )
    return pd.read_csv(path, dtype=str)
