from __future__ import annotations

from spotify_agent.safety import redact_secrets


def test_redact_secrets_covers_slack_stripe_and_github_tokens() -> None:
    samples = {
        "whsec_" + "a" * 32: "whsec_",
        "rk_live_" + "a" * 24: "rk_live_",
        "https://hooks.slack.com/services/T00/B00/xxx": "hooks.slack.com",
        "github_pat_" + "A" * 22: "github_pat_",
        "xoxb-1234567890-abcdefghij": "xoxb-",
        "xapp-1-A-123-xyzxyzxy": "xapp-",
    }
    for raw, needle in samples.items():
        redacted = redact_secrets(f"token {raw} end")
        assert needle not in redacted, raw
        assert "[REDACTED]" in redacted


def test_redact_secrets_none_is_empty() -> None:
    assert redact_secrets(None) == ""
    assert "sk-abcdefghijk" not in redact_secrets("sk-abcdefghijk")
