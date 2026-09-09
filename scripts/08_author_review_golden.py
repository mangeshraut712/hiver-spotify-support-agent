#!/usr/bin/env python3
"""Legacy rule proposals. This is NOT human annotation; do not use as gold."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.paths import GOLDEN_PATH


def author_label(text: str) -> tuple[str, bool, str]:
    t = text.lower()

    # Security first (narrow).
    if re.search(
        r"\bhacked\b|\bhacking\b|\bstolen\b|unauthorized|someone (else )?changed|"
        r"devices i don't recognize",
        t,
    ):
        return "account_security", True, "author: security"
    if "compromised" in t and any(w in t for w in ("account", "email", "login", "password")):
        return "account_security", True, "author: compromised account"

    # Login / account recovery before generic howto.
    if re.search(
        r"can'?t (log|sign) ?in|cannot (log|sign) ?in|password|locked out|facebook login|"
        r"log ?in (with|using|via)|reset my password|get my account back|account back",
        t,
    ):
        return "login_access", True, "author: login"

    # Billing / cancel — require money or cancel-subscription sense.
    cancel_sub = bool(
        re.search(
            r"cancel(led|ed)? (my )?(subscription|premium|plan|account)|"
            r"can'?t seem to cancel|unsubscribe|stop (my )?premium|still charg",
            t,
        )
    )
    billing = bool(
        re.search(
            r"charg(e|ed|ing)|refund|billed|payment|credit card|debit card|receipt|money|"
            r"invoice|double charg|student membership",
            t,
        )
    )
    if cancel_sub and billing:
        return "cancel_premium", True, "author: cancel+charge"
    if cancel_sub:
        return "cancel_premium", True, "author: cancel sub"
    if billing:
        return "billing_charge", True, "author: billing"

    if re.search(
        r"family plan|premium family|premium for family|\bduo\b|household|pff acct|"
        r"add my family|change my address|family member",
        t,
    ):
        return "family_duo_plan", True, "author: family/duo"

    if re.search(
        r"won'?t play|not playing|does not want to play|buffering|crash|freezing|glitch|"
        r"undownload|download(ed|s|ing)? (songs|tracks|fail)|offline (mode|download)",
        t,
    ):
        esc = bool(re.search(r"every (other )?week|still|again|broken|fix", t))
        return "playback_bug", esc, "author: playback"

    if re.search(r"how (do|can|to)|where (do|can)|what('s| is) the (best )?way", t):
        return "feature_howto", False, "author: howto"

    if re.search(r"playlist|shuffle|lyrics|podcast|settings|recommend", t) and "?" in text:
        return "feature_howto", False, "author: feature question"

    if "premium" in t and any(w in t for w in ("help", "assist", "problem", "issue")):
        return "other", True, "author: vague premium → escalate"

    return "other", False, "author: other"


def main() -> None:
    raise SystemExit("Disabled: regex labels are not human review. Use scripts/10_review.py export, then import actual reviews.")
    rows = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]
    updated = []
    changed = 0
    for row in rows:
        intent, escalate, notes = author_label(row["customer_text"])
        if intent != row["intent"] or escalate != row["escalate"]:
            changed += 1
        row["intent"] = intent
        row["escalate"] = escalate
        row["notes"] = notes
        updated.append(row)

    with GOLDEN_PATH.open("w") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Reviewed {len(updated)}; changed {changed}")
    print("intent", Counter(r["intent"] for r in updated))
    print("escalate", Counter(r["escalate"] for r in updated))


if __name__ == "__main__":
    main()
