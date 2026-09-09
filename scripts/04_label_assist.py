#!/usr/bin/env python3
"""Propose intent/escalate labels for candidate tweets (human must finalize)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.intents import INTENT_DESCRIPTIONS, INTENTS, keyword_intent, should_escalate_heuristic
from spotify_agent.llm import agent_model, chat_json


def propose(text: str, use_llm: bool) -> dict:
    keyword = keyword_intent(text)
    escalate = should_escalate_heuristic(text, keyword)
    proposal = {
        "keyword_intent": keyword,
        "keyword_escalate": escalate,
    }
    if use_llm:
        system = (
            "You label Spotify customer-support tweets. "
            "Return JSON with intent and escalate boolean."
        )
        intent_lines = "\n".join(f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())
        user = f"Labels:\n{intent_lines}\n\nTweet:\n{text}"
        raw = chat_json(
            model=agent_model(),
            system=system,
            user=user,
            temperature=0.0,
            cache_prefix="label",
        )
        intent = raw.get("intent", keyword)
        if intent not in INTENTS:
            intent = keyword
        proposal["llm_intent"] = intent
        proposal["llm_escalate"] = bool(raw.get("escalate", escalate))
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = []
    with args.candidates.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_rows = []
    for row in rows:
        prop = propose(row["customer_text"], use_llm=args.use_llm)
        out_rows.append({**row, **prop})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} proposals → {args.out}")


if __name__ == "__main__":
    main()
