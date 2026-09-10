import importlib.util
from pathlib import Path
import pandas as pd
import pytest
from spotify_agent.retrieve import ReplyRetriever
from eval import harness

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review", ROOT / "scripts/10_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_blank_and_duplicate_reviews_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        review.validate([{"id": "a"}], ["a"])
    with pytest.raises(ValueError, match="duplicates"):
        review.validate([{"id": "a"}, {"id": "a"}], ["a"])


def test_judge_does_not_manufacture_missing_scores(monkeypatch):
    monkeypatch.setattr(harness, "chat_json", lambda **kwargs: {"pass_fail": "pass"})
    monkeypatch.setattr(harness, "judge_model", lambda: "test")
    with pytest.raises(ValueError, match="Invalid judge"):
        harness.judge_reply("question", "reply", "history")


def test_retrieval_excludes_tweet_id_and_zero_similarity():
    data = pd.DataFrame([{"pair_id": f"p{i}", "customer_tweet_id": str(i),
                          "customer_text": "music playlist question", "brand_reply_text": "playlist instructions"}
                         for i in range(3)])
    retriever = ReplyRetriever.build(data)
    assert "p0" not in [r.pair_id for r in retriever.search("playlist", exclude_pair_ids=["0"])]
    assert retriever.search("qwertyunknown") == []
    assert retriever.search("playlist", k=0) == []
