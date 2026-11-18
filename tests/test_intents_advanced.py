"""Advanced unit tests for intent classification and escalation heuristics."""

from spotify_agent.intents import (
    INTENT_DESCRIPTIONS,
    INTENT_KEYWORDS,
    INTENTS,
    _contains_keyword,
    keyword_intent,
    rule_based_intent,
    should_escalate_heuristic,
)


def test_intent_taxonomy_consistency():
    """Verify intent taxonomy completeness across descriptions and keyword tables."""
    assert len(INTENTS) == 8
    assert set(INTENTS) == set(INTENT_DESCRIPTIONS.keys())
    assert set(INTENTS) == set(INTENT_KEYWORDS.keys())
    for intent in INTENTS:
        assert len(INTENT_DESCRIPTIONS[intent]) > 0


def test_contains_keyword_token_boundaries():
    """Verify _contains_keyword distinguishes full tokens and allowed plurals from substrings."""
    assert _contains_keyword("check my playlist please", "playlist")
    assert _contains_keyword("organizing my playlists", "playlist")
    assert _contains_keyword("request a refund now", "refund")
    assert _contains_keyword("tracking refunds", "refund")

    # Substrings must not trigger false positives
    assert not _contains_keyword("my phone charger is broken", "charge")
    assert not _contains_keyword("staying at the shack tonight", "hack")
    assert not _contains_keyword("listen to podcasts", "cast")


def test_card_cancellation_routes_to_billing():
    """Verify cancelling payment cards routes to billing_charge rather than cancel_premium."""
    cases = [
        "I need to cancel my credit card on this account",
        "Please cancel my debit card immediately",
        "How do I update or cancel my card on file",
    ]
    for text in cases:
        intent, esc = rule_based_intent(text)
        assert intent == "billing_charge"
        assert esc is True


def test_lockout_priority_over_cancellation():
    """Verify login lockout takes precedence over cancel signals to restore access first."""
    text = "I am locked out and cannot log in to cancel my subscription"
    intent, esc = rule_based_intent(text)
    assert intent == "login_access"
    assert esc is True


def test_empty_and_nan_fallback_to_other():
    """Verify null, empty, whitespace, and NaN strings resolve safely to other."""
    assert keyword_intent(None) == "other"
    assert keyword_intent("") == "other"
    assert keyword_intent("nan") == "other"
    assert keyword_intent("none") == "other"

    intent, esc = rule_based_intent("   ")
    assert intent == "other"
    assert esc is False


def test_escalation_heuristics():
    """Verify escalation heuristics accurately triage high-risk customer contacts."""
    # Critical security and billing intents must escalate
    assert should_escalate_heuristic("charged twice on receipt", "billing_charge") is True
    assert should_escalate_heuristic("account hacked someone else logged in", "account_security") is True
    assert should_escalate_heuristic("cannot reset password", "login_access") is True

    # Playback bug escalates only on recurring / persistent symptoms
    assert should_escalate_heuristic("app keeps crashing every time", "playback_bug") is True
    assert should_escalate_heuristic("how do I sort downloads", "playback_bug") is False

    # Other category triggers on legal/fraud keywords
    assert should_escalate_heuristic("contacting my lawyer about fraud", "other") is True
    assert should_escalate_heuristic("great app enjoying the music", "other") is False
