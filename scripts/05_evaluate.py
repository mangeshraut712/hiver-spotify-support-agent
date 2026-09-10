#!/usr/bin/env python3
"""Evaluate systems on the golden set and write headline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.harness import evaluate_predictions, load_golden
from spotify_agent.agent import SupportAgent
from spotify_agent.baselines import SimpleBaseline, TrivialBaseline
from spotify_agent.paths import GOLDEN_PATH, HEADLINE_PATH, RESULTS_DIR, ensure_dirs
from spotify_agent.retrieve import ReplyRetriever


def get_system(name: str, retriever: ReplyRetriever):
    if name == "agent":
        return SupportAgent(retriever)
    if name == "simple":
        return SimpleBaseline(retriever)
    if name == "trivial":
        return TrivialBaseline()
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        default="agent,simple,trivial",
        help="Comma-separated: agent,simple,trivial",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--judge", action="store_true", default=True)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-limit", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    goldens = load_golden(args.golden)
    if args.limit:
        goldens = goldens[: args.limit]

    retriever = ReplyRetriever.load()
    run_judge = args.judge and not args.no_judge

    # A run is one experiment; never merge different subsets or label versions.
    headline = {"systems": {}}
    headline["n_eval"] = len(goldens)
    headline["label_provenance"] = "automated_proposal_not_human_review"
    headline["evidence_status"] = "exploratory_pending_human_validation"
    headline.setdefault("systems", {})

    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        system = get_system(name, retriever)
        preds = []
        for g in tqdm(goldens, desc=name):
            # Exclude the golden pair itself from retrieval to avoid leakage.
            exclude = [g.id] if g.id.startswith("sp_") else [g.customer_tweet_id]
            # Also try pair_id field if present in notes — golden id is g.id
            exclude_ids = {g.id, g.customer_tweet_id}
            preds.append(
                system.run(g.customer_text, exclude_pair_ids=list(exclude_ids))
            )

        metrics = evaluate_predictions(
            goldens,
            preds,
            run_judge=run_judge and name == "agent",
            judge_limit=args.judge_limit,
        )
        out_path = RESULTS_DIR / f"metrics_{name}.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        summary = {
            "intent_accuracy": metrics["intent_accuracy"],
            "intent_macro_f1": metrics["intent_macro_f1"],
            "escalate": metrics["escalate"],
        }
        if "judge" in metrics:
            summary["judge"] = metrics["judge"]
        headline["systems"][name] = summary
        print(name, json.dumps(summary, indent=2))

    HEADLINE_PATH.write_text(json.dumps(headline, indent=2))
    print(f"Wrote {HEADLINE_PATH}")


if __name__ == "__main__":
    main()
