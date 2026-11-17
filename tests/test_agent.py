"""Unit tests for SupportAgent, prompt generation, and fencing contracts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from spotify_agent.agent import SupportAgent, _format_examples, build_user_prompt
from spotify_agent.retrieve import RetrievedExample


def test_format_examples_empty_and_populated():
    """Empty example sequences yield '(no exemplars)' sentinel."""
    assert _format_examples([]) == "(no exemplars)"
    assert _format_examples(()) == "(no exemplars)"

    ex = RetrievedExample(
        pair_id="ex-123",
        customer_text="help my sound is gone",
        brand_reply_text="Restart the app and check output device.",
        score=0.88,
    )
    formatted = _format_examples([ex])
    assert "Example 1 (score=0.880, id=ex-123):" in formatted
    assert "Customer: help my sound is gone" in formatted
    assert "SpotifyCares: Restart the app and check output device." in formatted


def test_build_user_prompt_fencing_and_intent_locking():
    """Prompt must fence untrusted customer text and lock intent."""
    ex = RetrievedExample(
        pair_id="ex-1",
        customer_text="playlist won't load",
        brand_reply_text="Clear cache.",
        score=0.9,
    )
    prompt = build_user_prompt(
        customer_text="How do I make a collaborative playlist?",
        examples=[ex],
        locked_intent="feature_howto",
    )

    assert "LOCKED intent for this tweet (do not change): feature_howto" in prompt
    assert "<<<CUSTOMER>>>" in prompt
    assert "<<<END CUSTOMER>>>" in prompt
    assert "How do I make a collaborative playlist?" in prompt
    assert "<<<RETRIEVED>>>" in prompt
    assert "<<<END RETRIEVED>>>" in prompt
    assert 'Set "intent" exactly to "feature_howto".' in prompt


def test_build_user_prompt_handles_none_and_invalid_intent():
    """Prompt builder gracefully handles None customer text and unknown intent."""
    prompt = build_user_prompt(
        customer_text=None,
        examples=None,
        locked_intent="unknown_intent_type",
    )
    assert "LOCKED intent for this tweet (do not change): other" in prompt
    assert "<<<CUSTOMER>>>" in prompt
    assert "<<<END CUSTOMER>>>" in prompt
    assert "(no exemplars)" in prompt


class _StubRetriever:
    def __init__(self, examples: list[RetrievedExample] | None = None):
        self._examples = examples or []

    def search(self, text, k=5, exclude_pair_ids=None):
        return self._examples[:k]


def test_support_agent_compose_mode_sensitive_intent_escalation(monkeypatch):
    """Sensitive intents (billing, security) must unconditionally escalate."""
    monkeypatch.setenv("AGENT_REPLY_MODE", "compose")
    retriever = _StubRetriever()
    agent = SupportAgent(retriever=retriever, k=3)

    # Billing charge intent
    output = agent.run("I was charged $15.99 unexpectedly twice this month")
    assert output.intent == "billing_charge"
    assert output.decision == "escalate"
    assert output.confidence >= 0.65

    # Account security intent
    output_sec = agent.run("My Spotify account was compromised and password changed")
    assert output_sec.intent == "account_security"
    assert output_sec.decision == "escalate"


def test_support_agent_retrieval_mode(monkeypatch):
    """Retrieval mode copies top candidate reply when available and safe."""
    monkeypatch.setenv("AGENT_REPLY_MODE", "retrieval")
    top_ex = RetrievedExample(
        pair_id="ret-1",
        customer_text="how do I shuffle albums",
        brand_reply_text="Head to the album page and tap the green Shuffle button at the top!",
        score=0.92,
    )
    retriever = _StubRetriever([top_ex])
    agent = SupportAgent(retriever=retriever, k=1)

    output = agent.run("how to shuffle an album?")
    assert output.intent == "feature_howto"
    assert output.reply == "Head to the album page and tap the green Shuffle button at the top!"
    assert output.reason == "rules+retrieval_copy"
    assert output.decision == "auto"
