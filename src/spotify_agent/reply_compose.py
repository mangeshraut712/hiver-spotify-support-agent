"""Compose issue-aware SpotifyCares-style replies (not bare DM templates)."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from spotify_agent.retrieve import RetrievedExample

# Short, on-brand openers keyed by intent — always mention the issue class.
_TEMPLATES = {
    "account_security": (
        "Sorry you're seeing possible unauthorized access — that's not okay. "
        "Please DM us your account's email/username (no passwords) so we can investigate and secure it."
    ),
    "billing_charge": (
        "Sorry about the billing trouble. Please DM us your account email/username and a screenshot of the charge "
        "(hide full card digits) so we can look into this with you."
    ),
    "cancel_premium": (
        "We can help with cancelling/stopping Premium. Please DM us the account email/username you're trying to manage "
        "and we'll take a closer look."
    ),
    "family_duo_plan": (
        "Sorry you're hitting Family/Duo plan issues. Please DM us the plan manager's email/username and what error you see "
        "so we can check the invite/household side."
    ),
    "login_access": (
        "Sorry you can't get into your account. Please DM us the email/username you're using (don't send passwords) "
        "and the exact error message — we'll help from there."
    ),
    "playback_bug": (
        "Sorry playback/downloads are acting up. Mind DMing your device + OS + Spotify app version, "
        "and a quick note on what happens? We'll dig in."
    ),
    "feature_howto": (
        "Happy to help with that. "
    ),
    "other": (
        "Thanks for reaching out — we want to help. "
    ),
}

_HOWTO_HINTS = (
    "If you can share which device/app screen you're on (or a screenshot via DM), we can point you to the right steps."
)

_DM = " DM us here and we'll continue: https://t.co/ldFdZRiNAt"


def _has_useful_historical(ex: Optional[RetrievedExample]) -> bool:
    if not ex:
        return False
    t = ex.brand_reply_text.lower()
    # Prefer historical only if it asks for specifics beyond a naked DM
    return any(
        w in t
        for w in (
            "device",
            "os",
            "version",
            "screenshot",
            "username",
            "email",
            "which",
            "what happens",
            "operating system",
        )
    )


def compose_reply(
    intent: str,
    customer_text: str,
    examples: Sequence[RetrievedExample],
    decision: str,
) -> str:
    """Build a grounded, issue-aware reply."""
    base = _TEMPLATES.get(intent, _TEMPLATES["other"])
    top = examples[0] if examples else None

    if intent == "feature_howto":
        # For how-tos, try to keep historical tips if they look specific; else ask clarifying Q.
        if top and _has_useful_historical(top) and len(top.brand_reply_text) < 320:
            reply = top.brand_reply_text
        else:
            reply = base + _HOWTO_HINTS + _DM
        return _trim(reply)

    if intent == "other" and decision == "auto":
        if top and len(top.brand_reply_text) < 280:
            return _trim(top.brand_reply_text)
        return _trim(base + "Could you share a bit more detail" + _DM)

    # Sensitive / escalate intents: issue-aware template; optionally borrow a specific ask from history
    reply = base
    cust = customer_text.lower()
    if intent == "billing_charge":
        if re.search(r"change|update|card info|payment method", cust):
            reply = (
                "Happy to help update your payment method. Please DM us the account email/username "
                "and confirm which card/country you're trying to use — we'll walk you through it."
            )
        elif re.search(r"refund", cust):
            reply = (
                "Sorry about the refund delay. Please DM us your account email/username plus a screenshot "
                "of the charge (hide full card digits) so we can investigate the billing side with you."
            )
    if top and _has_useful_historical(top):
        # Merge one concrete ask from history if not already covered
        hist = top.brand_reply_text
        if "screenshot" in hist.lower() and "screenshot" not in reply.lower():
            reply += " A screenshot helps too."
        if "device" in hist.lower() and "device" not in reply.lower() and intent == "playback_bug":
            pass  # already asks device
    if "dm" not in reply.lower():
        reply += _DM
    return _trim(reply)


def _trim(text: str, limit: int = 380) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
