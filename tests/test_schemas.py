"""Unit tests for spotify_agent schemas."""

import pytest
from pydantic import ValidationError

from spotify_agent.intents import INTENTS
from spotify_agent.schemas import (
    AgentOutput,
    GoldenExample,
    JudgeScore,
    agent_output_to_dict,
    parse_agent_output,
)


def test_agent_output_valid():
    output = AgentOutput(
        intent="billing_charge",
        reply="Please check your subscription in Account Settings.",
        decision="auto",
        reason="Resolved via subscription guide.",
        confidence=0.92,
        retrieved_ids=["pair_1", "pair_2"],
    )
    assert output.intent == "billing_charge"
    assert output.decision == "auto"
    assert output.confidence == 0.92
    assert output.retrieved_ids == ["pair_1", "pair_2"]


def test_agent_output_intent_fallback():
    # Unknown intent automatically falls back to 'other'
    output = AgentOutput(
        intent="non_existent_intent",
        reply="Here is some general help.",
        decision="escalate",
        reason="Unknown intent.",
        confidence=0.5,
    )
    assert output.intent == "other"
    assert output.decision == "escalate"


def test_agent_output_confidence_bounds():
    # Confidence > 1.0 rejected
    with pytest.raises(ValidationError):
        AgentOutput(
            intent="other",
            reply="msg",
            decision="auto",
            reason="reason",
            confidence=1.05,
        )

    # Confidence < 0.0 rejected
    with pytest.raises(ValidationError):
        AgentOutput(
            intent="other",
            reply="msg",
            decision="auto",
            reason="reason",
            confidence=-0.1,
        )


def test_agent_output_serialization_roundtrip():
    original = AgentOutput(
        intent="login_access",
        reply="Reset your password at spotify.com/reset.",
        decision="auto",
        reason="Standard reset procedure.",
        confidence=0.88,
        retrieved_ids=["123"],
    )
    payload = agent_output_to_dict(original)
    assert isinstance(payload, dict)
    assert payload["intent"] == "login_access"

    restored = parse_agent_output(payload)
    assert restored == original


def test_golden_example_valid():
    sample_intent = next(iter(INTENTS))
    golden = GoldenExample(
        id="gold_1",
        customer_tweet_id="111222",
        customer_text="Why was I charged twice?",
        brand_reply_text="We can take a look. Send us a DM.",
        intent=sample_intent,
        escalate=True,
        stratum="billing",
        notes="Duplicate billing inquiry",
    )
    assert golden.id == "gold_1"
    assert golden.intent == sample_intent
    assert golden.escalate is True
    assert golden.stratum == "billing"


def test_golden_example_invalid_intent():
    # Unlike AgentOutput, GoldenExample strictly rejects unknown intents
    with pytest.raises(ValidationError) as exc_info:
        GoldenExample(
            id="gold_2",
            customer_tweet_id="333444",
            customer_text="Some text",
            brand_reply_text="Some reply",
            intent="invalid_intent_xyz",
            escalate=False,
        )
    assert "Unknown intent: invalid_intent_xyz" in str(exc_info.value)


def test_judge_score_valid_and_mean():
    score = JudgeScore(
        groundedness=5,
        brand_voice=4,
        helpfulness=5,
        safety=4,
        pass_fail="pass",
        rationale="Accurate and helpful guidance.",
    )
    assert score.groundedness == 5
    assert score.pass_fail == "pass"
    assert score.mean_score == 4.5


def test_judge_score_bounds():
    # Score out of 1-5 range rejected
    with pytest.raises(ValidationError):
        JudgeScore(
            groundedness=6,
            brand_voice=4,
            helpfulness=5,
            safety=4,
            pass_fail="pass",
        )

    with pytest.raises(ValidationError):
        JudgeScore(
            groundedness=0,
            brand_voice=4,
            helpfulness=5,
            safety=4,
            pass_fail="pass",
        )
