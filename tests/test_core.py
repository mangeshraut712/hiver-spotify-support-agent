"""Unit tests for intents, schemas, and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.baselines import TrivialBaseline
from spotify_agent.intents import keyword_intent, rule_based_intent, should_escalate_heuristic
from spotify_agent.schemas import AgentOutput, GoldenExample
from eval.harness import accuracy, binary_prf, cohens_kappa, macro_f1


def test_keyword_intent_billing():
    assert keyword_intent("I was charged twice this month!") == "billing_charge"


def test_keyword_intent_security():
    assert keyword_intent("My account was hacked please help") == "account_security"


def test_escalate_heuristic():
    assert should_escalate_heuristic("need a refund", "billing_charge") is True
    assert should_escalate_heuristic("how do I make a playlist", "feature_howto") is False


def test_agent_output_unknown_intent_maps_other():
    out = AgentOutput(
        intent="not_a_real_intent",
        reply="hi",
        decision="escalate",
        reason="x",
        confidence=0.5,
    )
    assert out.intent == "other"


def test_golden_roundtrip():
    g = GoldenExample(
        id="sp_00001",
        customer_tweet_id="1",
        customer_text="can't log in",
        brand_reply_text="Please DM us",
        intent="login_access",
        escalate=True,
    )
    assert g.intent == "login_access"


def test_metrics_helpers():
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5
    f1 = macro_f1(["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
    assert 0.0 <= f1 <= 1.0
    prf = binary_prf([True, False, True], [True, True, False])
    assert "f1" in prf
    assert cohens_kappa(["pass", "fail", "pass"], ["pass", "fail", "fail"]) <= 1.0


def test_trivial_baseline():
    out = TrivialBaseline().run("anything")
    assert out.intent == "other"
    assert out.decision == "escalate"


def test_rule_billing_beats_security_on_charges():
    intent, esc = rule_based_intent(
        "my debit card got hacked so i'm trying to fix my payment but charged 6 times"
    )
    assert intent == "billing_charge"
    assert esc is True


def test_rule_card_change_not_cancel():
    intent, _ = rule_based_intent(
        "how do I change my credit card info? We had to cancel my original one due to fraud"
    )
    assert intent == "billing_charge"


def test_rule_thanks_not_security():
    intent, esc = rule_based_intent(
        "That seems to have worked, thanks mate. Didn't know Spotify could get hacked ffs!"
    )
    assert intent == "other"
    assert esc is False
