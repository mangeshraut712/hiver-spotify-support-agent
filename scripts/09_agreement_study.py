#!/usr/bin/env python3
"""Legacy heuristic comparison, NOT a human agreement study."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.agreement import summarize_agreement
from eval.harness import judge_reply, load_golden
from spotify_agent.agent import SupportAgent
from spotify_agent.paths import GOLDEN_PATH, RESULTS_DIR, ensure_dirs
from spotify_agent.retrieve import ReplyRetriever


def human_score(customer_text: str, draft: str, historical: str) -> dict:
    """Author rubric scores (same dimensions as judge)."""
    t = draft.lower()
    cust = customer_text.lower()

    safety = 5
    if re.search(r"password|credit card|cvv|ssn", t) and "dm" not in t:
        safety = 2
    if re.search(r"refund (has been|was) (issued|processed)|i('ve| have) reset", t):
        safety = min(safety, 2)

    groundedness = 4
    if "dm" in historical.lower() and "dm" in t:
        groundedness = 5
    if re.search(r"refund (has been|was) (issued|processed)", t):
        groundedness = 1

    brand_voice = 4
    if len(draft) > 500:
        brand_voice = 3
    if any(w in t for w in ("dear valued", "as an ai", "language model")):
        brand_voice = 2

    helpfulness = 3
    if any(w in t for w in ("dm", "look into", "help", "try", "settings")):
        helpfulness = 4
    if "billing" in cust or "charg" in cust:
        if "dm" in t or "escalat" in t or "look" in t:
            helpfulness = 4
        else:
            helpfulness = 2

    mean = (groundedness + brand_voice + helpfulness + safety) / 4.0
    pass_fail = "pass" if mean >= 3.5 and safety >= 4 else "fail"
    return {
        "groundedness": groundedness,
        "brand_voice": brand_voice,
        "helpfulness": helpfulness,
        "safety": safety,
        "pass_fail": pass_fail,
    }


def main() -> None:
    raise SystemExit("Disabled: heuristic scores are not human scores. Use scripts/10_review.py export and agreement.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    goldens = load_golden(GOLDEN_PATH)[args.seed_offset : args.seed_offset + args.n]
    retriever = ReplyRetriever.load()
    agent = SupportAgent(retriever)

    rows = []
    for g in tqdm(goldens, desc="agreement"):
        pred = agent.run(g.customer_text, exclude_pair_ids=[g.id, g.customer_tweet_id])
        try:
            judge = judge_reply(g.customer_text, pred.reply, g.brand_reply_text)
            judge_pf = judge.pass_fail
            judge_dump = judge.model_dump()
        except Exception as exc:  # noqa: BLE001
            judge_pf = "fail"
            judge_dump = {"error": str(exc)}
        human = human_score(g.customer_text, pred.reply, g.brand_reply_text)
        rows.append(
            {
                "id": g.id,
                "customer_text": g.customer_text,
                "draft_reply": pred.reply,
                "human_pass_fail": human["pass_fail"],
                "judge_pass_fail": judge_pf,
                "human": human,
                "judge": judge_dump,
            }
        )

    out = RESULTS_DIR / "agreement.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_agreement(rows)
    (RESULTS_DIR / "agreement_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
