from __future__ import annotations

from spotify_agent.safety import redact_secrets


def test_redact_secrets_covers_slack_stripe_and_github_tokens() -> None:
    samples = {
        "whsec_" + "a" * 32: "whsec_",
        "rk_live_" + "a" * 24: "rk_live_",
        "sk_live_" + "b" * 24: "sk_live_",
        "hf_" + "c" * 24: "hf_",
        "https://hooks.slack.com/services/T00/B00/xxx": "hooks.slack.com",
        "github_pat_" + "A" * 22: "github_pat_",
        "xoxb-1234567890-abcdefghij": "xoxb-",
        "xapp-1-A-123-xyzxyzxy": "xapp-",
        "AKIAIOSFODNN7EXAMPLE": "AKIA",
        "AIzaSyD-1234567890abcdefghijklmnopqrst": "AIzaSyD",
    }
    for raw, needle in samples.items():
        redacted = redact_secrets(f"token {raw} end")
        assert needle not in redacted, raw
        assert "[REDACTED]" in redacted


def test_redact_secrets_none_is_empty() -> None:
    assert redact_secrets(None) == ""
    assert "sk-abcdefghijk" not in redact_secrets("sk-abcdefghijk")


def test_redact_secrets_github_token_prefixes() -> None:
    """Verify GitHub token prefixes (ghp, gho, ghu, ghs, ghr) are redacted."""
    prefixes = ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
    for prefix in prefixes:
        token = f"{prefix}1234567890abcdef"
        text = f"Commit authorized by {token} in CI"
        redacted = redact_secrets(text)
        assert token not in redacted
        assert "[REDACTED]" in redacted


def test_redact_secrets_bearer_and_env_assignment() -> None:
    """Verify Bearer headers and explicit OPENAI_API_KEY assignments are scrubbed."""
    text1 = "Authorization: Bearer secret_access_token_abcdef12345"
    assert "secret_access_token_abcdef12345" not in redact_secrets(text1)
    assert "[REDACTED]" in redact_secrets(text1)

    text2 = "Set OPENAI_API_KEY = sk-proj-1234567890abcdef before starting"
    assert "sk-proj-1234567890abcdef" not in redact_secrets(text2)
    assert "[REDACTED]" in redact_secrets(text2)


def test_redact_secrets_aws_temporary_credentials() -> None:
    """Verify AWS STS temporary credentials starting with ASIA are redacted."""
    sts_key = "ASIAIOSFODNN7EXAMPLE"
    text = f"AWS session key: {sts_key}"
    redacted = redact_secrets(text)
    assert sts_key not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_env_var_fallback(monkeypatch) -> None:
    """Verify literal OPENAI_API_KEY environment variable value is scrubbed."""
    custom_secret = "my_custom_secret_key_9999"
    monkeypatch.setenv("OPENAI_API_KEY", custom_secret)
    text = f"Configured upstream secret {custom_secret} into client"
    redacted = redact_secrets(text)
    assert custom_secret not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_non_string_types() -> None:
    """Non-string inputs like integers or booleans are safely stringified."""
    assert redact_secrets(42) == "42"
    assert redact_secrets(True) == "True"

