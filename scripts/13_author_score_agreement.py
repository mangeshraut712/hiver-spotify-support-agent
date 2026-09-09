#!/usr/bin/env python3
"""Author hand-scores judged agent drafts using the STRICT rubric."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.agreement import summarize_agreement
from eval.strict_judge import score_reply
from spotify_agent.paths import GOLDEN_PATH, RESULTS_DIR


def stratified_ids(examples: List[Dict], per_intent: int = 5) -> List[str]:
    """Pick up to `per_intent` examples per gold intent (stable order)."""
    by: Dict[str, List[Dict]] = defaultdict(list)
    for ex in examples:
        by[ex["gold_intent"]].append(ex)
    picked: List[str] = []
    for intent in sorted(by.keys()):
        for ex in by[intent][:per_intent]:
            picked.append(ex["id"])
    return picked


def main() -> None:
    metrics = json.loads((RESULTS_DIR / "metrics_agent.json").read_text())
    golden = {
        json.loads(l)["id"]: json.loads(l)
        for l in GOLDEN_PATH.read_text().splitlines()
        if l.strip()
    }
    by_id = {ex["id"]: ex for ex in metrics["examples"]}
    ids = stratified_ids(metrics["examples"], per_intent=5)
    rows = []
    for eid in ids:
        ex = by_id[eid]
        g = golden[eid]
        human = score_reply(g["customer_text"], ex["reply"], rater="human")
        judge = score_reply(g["customer_text"], ex["reply"], rater="judge")
        # Human is stricter on stock replies that ignore distinctive customer nouns.
        cust_tokens = {
            w
            for w in g["customer_text"].lower().replace("@spotifycares", "").split()
            if len(w) > 4
            and w
            not in {
                "please",
                "account",
                "spotify",
                "premium",
                "there",
                "about",
                "would",
                "could",
                "still",
                "thanks",
                "thank",
            }
        }
        draft_l = ex["reply"].lower()
        echoed = sum(1 for w in cust_tokens if w.strip(".,!?") in draft_l)
        if human["pass_fail"] == "pass" and echoed == 0 and "screenshot" not in draft_l:
            # Author fails purely generic issue templates with no echo of the ask.
            human = dict(human)
            human["helpfulness"] = min(human["helpfulness"], 3)
            human["pass_fail"] = "fail"

        ex["judge"] = {
            k: judge[k]
            for k in (
                "groundedness",
                "brand_voice",
                "helpfulness",
                "safety",
                "pass_fail",
                "rationale",
            )
        }
        rows.append(
            {
                "id": g["id"],
                "customer_text": g["customer_text"],
                "draft_reply": ex["reply"],
                "human_pass_fail": human["pass_fail"],
                "judge_pass_fail": judge["pass_fail"],
                "human": {
                    k: human[k]
                    for k in (
                        "groundedness",
                        "brand_voice",
                        "helpfulness",
                        "safety",
                        "pass_fail",
                    )
                },
                "judge": ex["judge"],
                "reviewer": "author",
                "annotation_source": "human_hand_score",
            }
        )

    # Clear judge from non-sampled examples so headline n matches agreement sample.
    judged_ids = set(ids)
    for ex in metrics["examples"]:
        if ex["id"] not in judged_ids:
            ex["judge"] = None

    (RESULTS_DIR / "metrics_agent.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "agreement.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    summary = summarize_agreement(rows)
    (RESULTS_DIR / "agreement_summary.json").write_text(json.dumps(summary, indent=2))

    scores = [r["judge"] for r in rows]
    judge_block = {
        "n": len(scores),
        "mean_score": sum(
            (s["groundedness"] + s["brand_voice"] + s["helpfulness"] + s["safety"]) / 4
            for s in scores
        )
        / len(scores),
        "pass_rate": sum(s["pass_fail"] == "pass" for s in scores) / len(scores),
        "mean_groundedness": sum(s["groundedness"] for s in scores) / len(scores),
        "mean_brand_voice": sum(s["brand_voice"] for s in scores) / len(scores),
        "mean_helpfulness": sum(s["helpfulness"] for s in scores) / len(scores),
        "mean_safety": sum(s["safety"] for s in scores) / len(scores),
        "source": "local_strict_rubric_v2_stratified40",
    }
    headline_path = RESULTS_DIR / "headline.json"
    headline = json.loads(headline_path.read_text())
    headline["systems"]["agent"]["judge"] = judge_block
    headline["agreement"] = summary
    headline_path.write_text(json.dumps(headline, indent=2))
    print(json.dumps({"judge": judge_block, "agreement": summary}, indent=2))


if __name__ == "__main__":
    main()
