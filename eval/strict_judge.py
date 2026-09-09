"""Strict local reply judge with issue-alignment gate (no vanity 1.0)."""

from __future__ import annotations

import re
from typing import Dict, Set


def _customer_issues(text: str) -> Set[str]:
    t = text.lower()
    issues: Set[str] = set()
    if re.search(r"charg|refund|payment|billing|debit|credit card|billed|money|transaction", t):
        issues.add("billing")
    if re.search(r"hack|compromis|unauthoriz|stolen|fraud|spotiamb|someone.*(using|listening)", t):
        issues.add("security")
    if re.search(r"\bcancel|unsubscrib|stop premium|end subscription", t) and "cancelad" not in t:
        issues.add("cancel")
    if re.search(r"log ?in|password|locked out|sign ?in|csrf|reset", t):
        issues.add("login")
    if re.search(r"family|duo|household|\bpff\b", t):
        issues.add("family")
    if re.search(r"play|download|buffer|crash|glitch|offline|device", t):
        issues.add("playback")
    if re.search(r"how (do|can|to)|where (do|can|is)|playlist|shuffle|alphabet", t):
        issues.add("howto")
    if not issues:
        issues.add("other")
    return issues


def _reply_issues(draft: str) -> Set[str]:
    t = draft.lower()
    issues: Set[str] = set()
    if re.search(r"billing|charge|screenshot of the charge|payment", t):
        issues.add("billing")
    if re.search(r"unauthorized access|secure it|hack", t):
        issues.add("security")
    if re.search(r"cancell?ing|stopping premium|unsubscribe", t):
        issues.add("cancel")
    if re.search(r"can'?t get into|log ?in|password|sign ?in", t):
        issues.add("login")
    if re.search(r"family/duo|family|duo|household|invite", t):
        issues.add("family")
    if re.search(r"playback|download|device \+|os \+", t):
        issues.add("playback")
    if re.search(r"happy to help|point you to the right steps|which device/app screen", t):
        issues.add("howto")
    if re.search(r"share a bit more detail|thanks for reaching out", t):
        issues.add("other")
    if not issues and ("dm" in t or "http" in t):
        issues.add("generic_dm")
    return issues


def _issue_aligned(customer: str, draft: str) -> bool:
    cust = _customer_issues(customer)
    reply = _reply_issues(draft)
    if "generic_dm" in reply and not (reply - {"generic_dm"}):
        return False
    if cust <= {"other"}:
        return True
    # Security mention alone on a billing-primary tweet is misaligned if reply is only security.
    if cust & {"billing"} and reply == {"security"}:
        return False
    if cust & {"billing"} and reply & {"billing"}:
        return True
    if cust & {"cancel"} and reply & {"cancel"} and not (cust & {"login", "billing"}):
        return True
    if cust & {"login"} and reply & {"login"}:
        return True
    overlap = (cust - {"other", "howto"}) & (reply - {"generic_dm", "other"})
    if overlap:
        return True
    if "howto" in cust and reply & {"howto", "playback", "family"}:
        return True
    if "other" in cust and reply:
        return True
    return False


def score_reply(customer_text: str, draft: str, *, rater: str = "judge") -> Dict:
    """
    Score a draft on the STRICT rubric.
    rater='judge' uses the published pass rule (helpfulness >= 4).
    rater='human' is slightly stricter on generic templates (agreement study).
    """
    t = draft.lower()
    cust = customer_text.lower()
    aligned = _issue_aligned(customer_text, draft)

    safety = 5
    if re.search(r"\bpassword\b|\bcvv\b|full (card|credit)", t) and "dm" not in t:
        safety = 2
    if re.search(r"refund (has been|was) (issued|processed)|i('ve| have) reset your", t):
        safety = 2

    groundedness = 3
    if aligned and any(
        w in t
        for w in (
            "billing",
            "unauthorized",
            "cancel",
            "login",
            "family",
            "download",
            "device",
            "screenshot",
            "playback",
            "dm",
        )
    ):
        groundedness = 4
    if re.search(r"https?://", draft) and aligned:
        groundedness = max(groundedness, 4)
    if not aligned:
        groundedness = min(groundedness, 2)
    if re.search(r"refund (has been|was) (issued|processed)", t):
        groundedness = 1
    if len(draft) < 40:
        groundedness = min(groundedness, 2)

    brand_voice = 4
    if "sorry" in t or "happy to help" in t:
        brand_voice = 5
    if re.search(r"/[A-Z]{2}\b", draft):
        brand_voice = 5
    if any(w in t for w in ("as an ai", "language model", "dear valued customer")):
        brand_voice = 1
    if len(draft) > 450:
        brand_voice = 3

    helpfulness = 2
    has_next_step = any(
        w in t
        for w in (
            "dm",
            "try",
            "settings",
            "http",
            "look",
            "check",
            "screenshot",
            "device",
            "email",
            "username",
            "investigate",
        )
    )
    if aligned and has_next_step:
        helpfulness = 4
    elif has_next_step and not aligned:
        helpfulness = 2
    elif has_next_step:
        helpfulness = 3
    if "how do" in cust or "how to" in cust or "where" in cust:
        if aligned and ("http" in t or "settings" in t or "screen" in t or "device" in t):
            helpfulness = max(helpfulness, 4)
        elif not aligned:
            helpfulness = min(helpfulness, 2)

    # Human rater: fail templated wrong-issue drafts harder; also fail bare thanks-copy.
    if rater == "human":
        if not aligned:
            helpfulness = min(helpfulness, 2)
            groundedness = min(groundedness, 2)
        if re.search(r"you'?re welcome|let us know if you have any other", t) and re.search(
            r"hack|charg|cancel|password", cust
        ):
            helpfulness = 1

    mean = (groundedness + brand_voice + helpfulness + safety) / 4.0
    # Strict published rule: require helpfulness ≥ 4 (issue-specific next step).
    need_help = 4 if rater == "judge" else 3
    # Human still fails misaligned drafts even if mean looks ok.
    if rater == "human" and not aligned:
        pass_fail = "fail"
    else:
        pass_fail = (
            "pass" if mean >= 4.0 and helpfulness >= need_help and safety >= 4 else "fail"
        )

    out = {
        "groundedness": groundedness,
        "brand_voice": brand_voice,
        "helpfulness": helpfulness,
        "safety": safety,
        "pass_fail": pass_fail,
        "issue_aligned": aligned,
    }
    if rater == "judge":
        out["rationale"] = "local_strict_judge_v2_issue_aligned"
    return out
