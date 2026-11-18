import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.baselines import TEMPLATE_REPLY, SimpleBaseline, TrivialBaseline
from spotify_agent.schemas import AgentOutput


@dataclass
class DummyHit:
    pair_id: str
    brand_reply_text: str
    score: float


class DummyRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.last_query = None
        self.last_exclude = None

    def search(self, query, k=1, exclude_pair_ids=None):
        self.last_query = query
        self.last_exclude = exclude_pair_ids
        return self._hits[:k]


def test_trivial_baseline():
    baseline = TrivialBaseline()
    out = baseline.run("Help with login")
    assert isinstance(out, AgentOutput)
    assert out.intent == "other"
    assert out.reply == TEMPLATE_REPLY
    assert out.decision == "escalate"
    assert out.confidence == 0.0
    assert out.retrieved_ids == []

    # None handling
    assert baseline.run(None).decision == "escalate"


def test_simple_baseline_safe_hit():
    retriever = DummyRetriever([
        DummyHit(pair_id="p-1", brand_reply_text="Clear your cache and restart the app.", score=0.85)
    ])
    baseline = SimpleBaseline(retriever)
    out = baseline.run("How do I fix playback issues?")

    assert out.reply == "Clear your cache and restart the app."
    assert out.retrieved_ids == ["p-1"]
    assert out.confidence == 0.85
    assert retriever.last_query == "How do I fix playback issues?"


def test_simple_baseline_unsafe_hit_falls_back():
    retriever = DummyRetriever([
        DummyHit(
            pair_id="p-2",
            brand_reply_text="Please send your password to verify your account",
            score=0.9,
        )
    ])
    baseline = SimpleBaseline(retriever)
    out = baseline.run("My music crashed")

    assert out.reply == TEMPLATE_REPLY
    assert out.retrieved_ids == ["p-2"]
    assert out.confidence == 0.1


def test_simple_baseline_no_hits():
    retriever = DummyRetriever([])
    baseline = SimpleBaseline(retriever)
    out = baseline.run("Something completely unknown")

    assert out.reply == TEMPLATE_REPLY
    assert out.retrieved_ids == []
    assert out.confidence == 0.1


def test_simple_baseline_escalate_heuristic():
    retriever = DummyRetriever([
        DummyHit(pair_id="p-3", brand_reply_text="Refund processed.", score=0.7)
    ])
    baseline = SimpleBaseline(retriever)
    out = baseline.run("I will sue you in small claims court for unauthorized charge!")

    assert out.decision == "escalate"
    assert "escalate" in out.reason.lower()


def test_simple_baseline_none_input():
    retriever = DummyRetriever([])
    baseline = SimpleBaseline(retriever)
    out = baseline.run(None)
    assert isinstance(out, AgentOutput)
    assert out.reply == TEMPLATE_REPLY
