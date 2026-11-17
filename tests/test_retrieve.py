"""Unit tests for ReplyRetriever and retrieval pipeline."""

import pandas as pd
import pytest

from spotify_agent.retrieve import ReplyRetriever


@pytest.fixture
def sample_pairs():
    return pd.DataFrame([
        {
            "pair_id": "pair_login",
            "customer_tweet_id": "tweet_1",
            "customer_text": "I cannot login to my Spotify premium account on my phone",
            "brand_reply_text": "Send us a DM with your account email address and we will help.",
        },
        {
            "pair_id": "pair_playback",
            "customer_tweet_id": "tweet_2",
            "customer_text": "Songs keep pausing and buffering when listening to music",
            "brand_reply_text": "Try clearing your cache and restarting your Spotify application.",
        },
        {
            "pair_id": "pair_family",
            "customer_tweet_id": "tweet_3",
            "customer_text": "How do I add my sibling to the family plan subscription?",
            "brand_reply_text": "Log into your account page online and send an invite link.",
        },
    ])


def test_retriever_build_and_search(sample_pairs):
    retriever = ReplyRetriever.build(sample_pairs, min_df=1)
    results = retriever.search("trouble login account", k=2)
    assert len(results) > 0
    assert results[0].pair_id == "pair_login"
    assert results[0].score > 0
    assert "Spotify premium account" in results[0].customer_text


def test_retriever_empty_and_guards(sample_pairs):
    retriever = ReplyRetriever.build(sample_pairs, min_df=1)

    # Empty and None queries
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search(None) == []

    # Non-positive k
    assert retriever.search("login", k=0) == []
    assert retriever.search("login", k=-1) == []

    # Empty dataframe rejects index building
    empty_df = pd.DataFrame(columns=["customer_text", "brand_reply_text"])
    with pytest.raises(ValueError) as exc:
        ReplyRetriever.build(empty_df)
    assert "empty corpus" in str(exc.value)


def test_retriever_exclusion(sample_pairs):
    retriever = ReplyRetriever.build(sample_pairs, min_df=1)

    # Exclude pair_id
    results_excluded = retriever.search("login account", k=2, exclude_pair_ids=["pair_login"])
    assert all(r.pair_id != "pair_login" for r in results_excluded)

    # Exclude by customer_tweet_id
    results_tweet_excluded = retriever.search("login account", k=2, exclude_pair_ids=["tweet_1"])
    assert all(r.pair_id != "pair_login" for r in results_tweet_excluded)


def test_retriever_save_and_load(sample_pairs, tmp_path):
    retriever = ReplyRetriever.build(sample_pairs, min_df=1)
    saved_dir = retriever.save(tmp_path)
    assert (saved_dir / "corpus.csv").exists()
    assert (saved_dir / "meta.json").exists()

    loaded = ReplyRetriever.load(saved_dir)
    loaded_results = loaded.search("songs buffering music", k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0].pair_id == "pair_playback"

    # Loading from non-existent directory raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ReplyRetriever.load(tmp_path / "non_existent")
