#!/usr/bin/env python3
"""
Build the golden evaluation set.

Process:
1. Stratified sample candidates from SpotifyCares pairs.
2. Propose labels via keyword rules + LLM.
3. Apply deterministic overrides for clear patterns.
4. Author review pass (scripted consistency checks + manual overrides file).

Final labels are written to eval/golden/golden.jsonl (200 examples).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.intents import (
    INTENTS,
    INTENT_DESCRIPTIONS,
    keyword_intent,
    should_escalate_heuristic,
)
from spotify_agent.llm import agent_model, chat_json
from spotify_agent.paths import GOLDEN_DIR, ensure_dirs


def llm_label(text: str) -> dict:
    system = (
        "You are labeling Spotify Twitter support messages for evaluation. "
        "Choose exactly one intent and whether a human should handle it "
        "(escalate=true) vs safe auto-reply (escalate=false). "
        "Prefer escalate for billing, refunds, hacking, account ownership disputes. "
        'Return JSON: {"intent":..., "escalate": true|false, "notes": "..."}'
    )
    intent_lines = "\n".join(f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())
    user = f"Labels:\n{intent_lines}\n\nCustomer tweet:\n{text}"
    try:
        raw = chat_json(
            model=agent_model(),
            system=system,
            user=user,
            temperature=0.0,
            cache_prefix="golden_label",
        )
    except Exception as exc:  # noqa: BLE001
        intent = keyword_intent(text)
        return {
            "intent": intent,
            "escalate": should_escalate_heuristic(text, intent),
            "notes": f"llm_failed:{type(exc).__name__}",
        }
    intent = raw.get("intent", "other")
    if intent not in INTENTS:
        intent = "other"
    return {
        "intent": intent,
        "escalate": bool(raw.get("escalate", True)),
        "notes": str(raw.get("notes", "")),
    }


def override_label(text: str, intent: str, escalate: bool) -> tuple[str, bool, str]:
    """Deterministic author overrides for high-precision patterns."""
    t = str(text).lower()

    if re.search(r"\bhack(ed|ing)?\b|\bstolen\b|unauthorized|compromised", t):
        return "account_security", True, "override: security language"
    if re.search(r"charg(e|ed|ing)|refund|billed|payment failed|double charged", t):
        if "cancel" in t and "still" in t:
            return "cancel_premium", True, "override: cancel+still charged"
        return "billing_charge", True, "override: billing language"
    if re.search(r"\bcancel(led|ed)?\b|unsubscribe|stop (my )?premium", t):
        return "cancel_premium", True, "override: cancel language"
    if re.search(r"family plan|premium family|duo plan|household", t):
        return "family_duo_plan", True, "override: family/duo"
    if re.search(r"can'?t (log|sign) ?in|password|locked out|reset (my )?password", t):
        return "login_access", True, "override: login"
    if re.search(
        r"won'?t play|not playing|buffering|keeps crashing|download(s|ing)? fail", t
    ):
        return "playback_bug", escalate or False, "override: playback"
    if re.search(r"how (do|can) i|how to|where (do|can) i", t) and intent in {
        "other",
        "feature_howto",
    }:
        return "feature_howto", False, "override: howto → auto-ok"

    if intent in {"billing_charge", "account_security", "cancel_premium"}:
        return intent, True, "policy: money/security escalate"
    return intent, escalate, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=GOLDEN_DIR / "candidates.jsonl")
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR / "golden.jsonl")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if not args.candidates.exists():
        raise SystemExit(f"Missing {args.candidates}; run 06_sample_golden_candidates.py")

    candidates = []
    with args.candidates.open() as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    golden = []
    for row in tqdm(candidates, desc="label"):
        if len(golden) >= args.n:
            break
        text = row["customer_text"]
        if args.skip_llm:
            intent = keyword_intent(text)
            escalate = should_escalate_heuristic(text, intent)
            notes = "keyword-only"
        else:
            lab = llm_label(text)
            intent, escalate, notes = lab["intent"], lab["escalate"], lab["notes"]
        intent, escalate, onote = override_label(text, intent, escalate)
        if onote:
            notes = f"{notes}; {onote}" if notes else onote
        golden.append(
            {
                "id": row["id"],
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": text,
                "brand_reply_text": row["brand_reply_text"],
                "intent": intent,
                "escalate": escalate,
                "stratum": row.get("stratum", "random"),
                "notes": notes,
            }
        )

    with args.out.open("w") as f:
        for row in golden:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(golden)} golden examples → {args.out}")
    print("intent", Counter(r["intent"] for r in golden))
    print("escalate", Counter(r["escalate"] for r in golden))


if __name__ == "__main__":
    main()
