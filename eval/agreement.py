"""Human vs LLM-judge agreement utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from eval.harness import cohens_kappa


def load_agreement_file(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_agreement(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    human = [r["human_pass_fail"] for r in rows]
    judge = [r["judge_pass_fail"] for r in rows]
    agree = sum(h == j for h, j in zip(human, judge)) / len(rows) if rows else 0.0
    return {
        "n": len(rows),
        "pass_fail_agreement": agree,
        "cohens_kappa_pass_fail": cohens_kappa(human, judge),
        "human_pass_rate": sum(h == "pass" for h in human) / len(rows) if rows else 0.0,
        "judge_pass_rate": sum(j == "pass" for j in judge) / len(rows) if rows else 0.0,
    }
