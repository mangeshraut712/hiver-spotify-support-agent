"""Guards for untrusted tweets, model output, and secret material."""

from __future__ import annotations

import math
import os
import re
from typing import Any

# Customer text and historical tweets are data. Fence markers are rewritten
# inside the body so a tweet cannot close the fence early.
_FENCE_OPEN = "<<<"
_FENCE_CLOSE = ">>>"

_CREDENTIAL_REQUESTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byour password\b", re.I),
    re.compile(r"\byour (?:cvv|cvc|pin|ssn|passcode)\b", re.I),
    re.compile(r"\bfull card number\b", re.I),
    re.compile(
        r"\b(?:send|share|provide|enter|paste|give)\b"
        r"(?:\s+\w+){0,4}\s+"
        r"(?:your\s+|the\s+)?(?:password|cvv|cvc|ssn)\b",
        re.I,
    ),
    re.compile(r"\b(?:password|cvv|cvc|ssn)\s*[:=]\s*\S+", re.I),
)

_INSTRUCTION_OVERRIDE = re.compile(
    r"\bignore (?:all |any )?(?:previous|prior|above) instructions\b",
    re.I,
)

_CREDENTIAL_TEXT = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|OPENAI_API_KEY\s*=\s*\S+)"
)


def fence_untrusted(label: str, text: Any, *, limit: int = 6000) -> str:
    """Wrap untrusted text so it cannot break out of a prompt delimiter."""
    safe_label = re.sub(r"[^A-Z0-9_]", "", str(label).upper()) or "DATA"
    body = "" if text is None else str(text)
    body = body.replace("\x00", "")
    body = body.replace(_FENCE_OPEN, "‹‹‹").replace(_FENCE_CLOSE, "›››")
    if len(body) > limit:
        body = body[:limit] + "\n…[truncated]"
    return f"<<<{safe_label}>>>\n{body}\n<<<END {safe_label}>>>"


def redact_secrets(text: str) -> str:
    """Remove API keys and bearer tokens from text that may be logged or raised."""
    redacted = _CREDENTIAL_TEXT.sub("[REDACTED]", text)
    key = os.getenv("OPENAI_API_KEY")
    if key and len(key) >= 8 and key in redacted:
        redacted = redacted.replace(key, "[REDACTED]")
    return redacted


def asks_for_secret(text: str) -> bool:
    """True when a reply asks the customer to reveal credentials or card data."""
    if _CREDENTIAL_TEXT.search(text):
        return True
    return any(pattern.search(text) for pattern in _CREDENTIAL_REQUESTS)


def is_unsafe_support_text(text: str) -> bool:
    """True when historical or model text should not be shown to a customer."""
    if asks_for_secret(text):
        return True
    if _INSTRUCTION_OVERRIDE.search(text):
        return True
    return _FENCE_OPEN in text or _FENCE_CLOSE in text


def sanitize_model_reply(value: Any, *, limit: int = 600) -> str | None:
    """Return a customer-facing reply, or None when the draft is empty or unsafe."""
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or text.lower() in {"none", "null"}:
        return None
    if redact_secrets(text) != text or is_unsafe_support_text(text):
        return None
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def coerce_confidence(value: Any, default: float = 0.75) -> float:
    """Clamp a model confidence to [0, 1]. Values from 1 to 100 are percents."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if 1.0 < number <= 100.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def parse_loose_bool(value: Any, default: bool = False) -> bool:
    """Parse JSON booleans. The string ``\"false\"`` is False, not True."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default
