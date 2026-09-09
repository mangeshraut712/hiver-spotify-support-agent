"""Intent taxonomy for SpotifyCares Twitter support."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Locked after exploratory review of SpotifyCares customer messages.
INTENTS: Tuple[str, ...] = (
    "billing_charge",
    "cancel_premium",
    "login_access",
    "playback_bug",
    "family_duo_plan",
    "account_security",
    "feature_howto",
    "other",
)

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "billing_charge": "Unexpected charges, refunds, failed payments, pricing disputes.",
    "cancel_premium": "Cancel Premium, stop subscription, still charged after cancel.",
    "login_access": "Can't log in, password reset, wrong account, Facebook/Apple login.",
    "playback_bug": "App won't play, buffering, downloads failing, device/platform glitches.",
    "family_duo_plan": "Family/Duo invites, household verification, plan member issues.",
    "account_security": "Hacked account, unauthorized activity, stolen credentials.",
    "feature_howto": "How to use a feature, playlists, offline mode, settings tips.",
    "other": "Unclear, off-topic, praise/complaint without actionable ask, or multi-issue muddle.",
}

# Keyword hints used for stratified sampling and the simple baseline.
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "billing_charge": [
        "charge",
        "charged",
        "billing",
        "refund",
        "payment",
        "receipt",
        "money",
        "invoice",
        "price",
        "subscription fee",
    ],
    "cancel_premium": [
        "cancel",
        "cancelled",
        "canceled",
        "unsubscribe",
        "stop premium",
        "end subscription",
    ],
    "login_access": [
        "login",
        "log in",
        "password",
        "can't sign",
        "cannot sign",
        "locked out",
        "reset",
        "facebook",
        "apple id",
    ],
    "playback_bug": [
        "won't play",
        "cant play",
        "can't play",
        "not playing",
        "buffering",
        "crash",
        "freezing",
        "download",
        "offline",
        "glitch",
        "bug",
    ],
    "family_duo_plan": [
        "family",
        "duo",
        "household",
        "invite",
        "plan member",
        "premium family",
        "family plan",
        "family account",
        "family premium",
        "pff",
        "add my family",
        "family member",
    ],
    "account_security": [
        "hacked",
        "hack",
        "stolen",
        "unauthorized",
        "compromised",
        "someone else",
        "fraud",
    ],
    "feature_howto": [
        "how do i",
        "how to",
        "where can i",
        "playlist",
        "shuffle",
        "lyrics",
        "podcast",
        "settings",
    ],
    "other": [],
}

ESCALATE_KEYWORDS: List[str] = [
    "refund",
    "hacked",
    "stolen",
    "unauthorized",
    "charged twice",
    "lawyer",
    "fraud",
    "scam",
    "disabled",
    "banned",
    "still charged",
    "money back",
    "credit card",
    "csrf",
    "password",
]


def keyword_intent(text: str) -> str:
    """Return the first matching intent by keyword priority, else other."""
    lowered = str(text if text is not None else "").lower()
    if lowered in {"nan", "none"}:
        return "other"
    priority = [
        "account_security",
        "billing_charge",
        "cancel_premium",
        "family_duo_plan",
        "login_access",
        "playback_bug",
        "feature_howto",
    ]
    for intent in priority:
        for kw in INTENT_KEYWORDS[intent]:
            if kw in lowered:
                return intent
    return "other"


def rule_based_intent(text: str) -> tuple[str, bool]:
    """Stronger regex rules for the agent. Returns (intent, escalate_hint)."""
    t = str(text if text is not None else "").lower()
    original = str(text if text is not None else "")

    # Resolved / gratitude — even if the tweet casually mentions past hacking.
    if re.search(
        r"^(thanks|thank you|done thanks|sure! will do)|"
        r"thanks (mate|spotify|for (helping|now|replying))|"
        r"(that )?seems to have worked,? thanks|"
        r"you('re| are) (a )?lifesaver|all good now",
        t.strip(),
    ):
        if not re.search(r"\bstill\b|please help|can't|cannot|need (help|to)", t):
            return "other", False

    billing_signal = bool(
        re.search(
            r"charg(e|ed|ing)|refund|payment|debit card|credit card|billed|money back|"
            r"student (membership|discount|premium)|\$\d|£\d|rm\d|"
            r"pay my monthly bill|won'?t go through|payment.*(fail|wrong|approved)|"
            r"trying to pay|change my (credit |debit )?card|card info|payment method|"
            r"unauthorized charg|transaction",
            t,
        )
    )

    # Billing before security when money/payment is the actionable ask
    # (e.g. "card got hacked… charged 6 times", "refund of unauthorized charges").
    if billing_signal and re.search(
        r"charg|refund|payment|billed|transaction|card info|payment method|"
        r"trying to (fix|pay|update)|won'?t go through",
        t,
    ):
        esc = not bool(
            re.search(r"accept debit|when i become a student|does the month|annual premium", t)
        )
        return "billing_charge", esc

    if re.search(
        r"\bhacked\b|\bhacking\b|\bhacker\b|compromised|unauthorized|"
        r"devices i don't recognize|spotiamb|unknown (app|client|device)",
        t,
    ):
        return "account_security", True
    if re.search(r"someone (is |has |keeps )?(using|changed|listening)", t) and "account" in t:
        return "account_security", True
    if "playlist" in t and re.search(r"hacker|hacked", t):
        return "account_security", True

    # Playback before cancel (Portuguese cancelados ≠ unsubscribe).
    if re.search(
        r"won'?t play|not playing|does not want to play|crash|glitch|buffering|freezing|"
        r"undownload|download(s|ed|ing)?|"
        r"songs (delete|disappear|skip)|app crashes|keeps (on )?glitch|music stop|"
        r"offline|dispositivos|cancelados|baix",
        t,
    ) and re.search(
        r"download|play|crash|glitch|offline|buffer|skip|delet|clear|wipe|dispositivo|cancelad",
        t,
    ):
        if re.search(r"how (do|can|to).*(playlist|alphabet|filter|curator)", t):
            return "feature_howto", False
        esc = True
        if re.search(
            r"how (do|can|to)|where (is|are)|sort downloads|stop the .*download", t
        ) and not re.search(r"keep|every|still|gone|delet|crash|glitch|cancelad", t):
            esc = False
        return "playback_bug", esc

    if re.search(
        r"family plan|premium family|premium for family|\bpff\b|"
        r"family (account|invite|member|premium)|add (my )?family|household|\bduo\b|"
        r"spotify family|invitation to my|invite.*(family|sis|dad|mom|wife|member)|"
        r"error 3|address (verify|verification)|kicked (out|off).*(family|plan)|"
        r"welcoming me to .+family|premium for family",
        t,
    ):
        return "family_duo_plan", True
    if re.search(
        r"added my (dad|mom|wife|son|daughter|sister|brother).*(member|premium|family)", t
    ):
        return "family_duo_plan", True
    if "premium cancelled" in t and re.search(r"member|dad|family", t):
        return "family_duo_plan", True

    login_signal = bool(
        re.search(
            r"can'?t (log|sign) ?in|cannot (log|sign) ?in|password|locked out|csrf|"
            r"reset (my )?password|reset my pw|get my account back|unable to log|"
            r"won'?t let me (log|sign)|log ?in (with|using|via|through)|"
            r"trying to do is log|sign-?in|logged out|don'?t know (my )?(password|email)",
            t,
        )
    )
    cancel_signal = bool(
        re.search(
            r"\bcancel(led|ling|ed)?\b|\bunsubscribe\b|stop (auto-?renew|premium)|#cancel|"
            r"subscription canceled|suspend rather than cancel|"
            r"automatically renewed.*cancel|renewed itself after i cancel|"
            r"can'?t even cancel|help to cancel|want to cancel|need to cancel",
            t,
        )
        and "cancelad" not in t
    )

    # Lockout + cancel SOS: restore access first.
    if login_signal and cancel_signal and re.search(
        r"password|email|locked|log ?in|sign ?in|hack", t
    ):
        return "login_access", True

    # "Cancel my card" / card update is billing, not Premium cancel.
    if re.search(
        r"change my (credit |debit )?card|card info|update.*(card|payment)|"
        r"cancel my (original )?one.*(fraud|card)",
        t,
    ):
        return "billing_charge", True

    if cancel_signal:
        esc = not bool(
            re.search(
                r"if i cancel now|does it end|can we choose to unsubscribe|so after the 3 months",
                t,
            )
        )
        if re.search(r"subscription canceled|laziness", t):
            esc = True
        return "cancel_premium", esc

    if billing_signal:
        esc = not bool(
            re.search(r"accept debit|when i become a student|does the month|annual premium", t)
        )
        return "billing_charge", esc

    if login_signal:
        return "login_access", True
    if re.search(r"how (do|can|to)|where (do|can|is)|what('s| is) the (best )?way", t):
        return "feature_howto", False
    if re.search(r"playlist|shuffle|podcast|settings|filter|curator|alphabet", t) and (
        "?" in original or "please" in t or "can you" in t or "can we" in t
    ):
        return "feature_howto", False

    if "premium" in t and any(w in t for w in ("help", "assist", "problem", "issue", "dm me")):
        return "other", True
    if re.search(r"subscription|account", t) and any(
        w in t for w in ("help", "assist", "problem", "issue", "please")
    ):
        return "other", True

    return "other", False


def should_escalate_heuristic(text: str, intent: str) -> bool:
    """Conservative escalate heuristic for the simple baseline."""
    lowered = str(text if text is not None else "").lower()
    if intent in {
        "billing_charge",
        "account_security",
        "cancel_premium",
        "login_access",
        "family_duo_plan",
    }:
        return True
    if intent == "playback_bug":
        return bool(
            re.search(
                r"keep|every|still|gone|delet|crash|glitch|cancelad|again|week|day",
                lowered,
            )
        )
    return any(kw in lowered for kw in ESCALATE_KEYWORDS)
