"""Evaluation metrics and LLM-as-judge."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from spotify_agent.intents import INTENTS
from spotify_agent.llm import chat_json, judge_model
from spotify_agent.paths import PROMPTS_DIR
from spotify_agent.schemas import AgentOutput, GoldenExample, JudgeScore


def load_golden(path: Path) -> List[GoldenExample]:
    examples: List[GoldenExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldenExample.model_validate(json.loads(line)))
    return examples


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def binary_prf(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> Dict[str, float]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support_positive": float(sum(y_true)),
    }


@dataclass
class ExampleResult:
    golden_id: str
    gold_intent: str
    pred_intent: str
    gold_escalate: bool
    pred_decision: str
    reply: str
    judge: Optional[JudgeScore] = None


def judge_reply(
    customer_text: str,
    draft_reply: str,
    historical_reply: str,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> JudgeScore:
    system = (PROMPTS_DIR / "judge_system.txt").read_text()
    user = (
        f"Customer message:\n{customer_text}\n\n"
        f"Historical SpotifyCares reply:\n{historical_reply}\n\n"
        f"Draft reply to evaluate:\n{draft_reply}\n"
    )
    raw = chat_json(
        model=model or judge_model(),
        system=system,
        user=user,
        temperature=0.0,
        use_cache=use_cache,
        cache_prefix="judge_v2",
    )
    if isinstance(raw, list):
        raw = raw[0] if raw and isinstance(raw[0], dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    for key in ("groundedness", "brand_voice", "helpfulness", "safety"):
        try:
            raw[key] = int(raw.get(key, 3))
        except (TypeError, ValueError):
            raw[key] = 3
        raw[key] = min(max(raw[key], 1), 5)
    raw["rationale"] = str(raw.get("rationale", ""))
    mean = (
        raw["groundedness"] + raw["brand_voice"] + raw["helpfulness"] + raw["safety"]
    ) / 4.0
    # Strict local gate so a soft judge model cannot inflate pass rate.
    raw["pass_fail"] = (
        "pass"
        if mean >= 4.0 and raw["helpfulness"] >= 4 and raw["safety"] >= 4
        else "fail"
    )
    return JudgeScore.model_validate(raw)

def evaluate_predictions(
    goldens: Sequence[GoldenExample],
    predictions: Sequence[AgentOutput],
    run_judge: bool = True,
    judge_limit: Optional[int] = None,
) -> Dict[str, Any]:
    assert len(goldens) == len(predictions)
    gold_intents = [g.intent for g in goldens]
    pred_intents = [p.intent for p in predictions]
    gold_esc = [g.escalate for g in goldens]
    pred_esc = [p.decision == "escalate" for p in predictions]

    per_example: List[Dict[str, Any]] = []
    judge_scores: List[JudgeScore] = []
    limit = judge_limit if judge_limit is not None else len(goldens)

    for i, (g, p) in enumerate(zip(goldens, predictions)):
        judge = None
        if run_judge and i < limit:
            judge = judge_reply(g.customer_text, p.reply, g.brand_reply_text)
            judge_scores.append(judge)
        per_example.append(
            {
                "id": g.id,
                "gold_intent": g.intent,
                "pred_intent": p.intent,
                "gold_escalate": g.escalate,
                "pred_decision": p.decision,
                "confidence": p.confidence,
                "reply": p.reply,
                "reason": p.reason,
                "retrieved_ids": p.retrieved_ids,
                "judge": judge.model_dump() if judge else None,
            }
        )

    metrics: Dict[str, Any] = {
        "n": len(goldens),
        "intent_accuracy": accuracy(gold_intents, pred_intents),
        "intent_macro_f1": macro_f1(gold_intents, pred_intents, INTENTS),
        "escalate": binary_prf(gold_esc, pred_esc),
        "intent_confusion": _confusion(gold_intents, pred_intents),
    }
    if judge_scores:
        metrics["judge"] = {
            "n": len(judge_scores),
            "mean_score": sum(j.mean_score for j in judge_scores) / len(judge_scores),
            "pass_rate": sum(j.pass_fail == "pass" for j in judge_scores)
            / len(judge_scores),
            "mean_groundedness": sum(j.groundedness for j in judge_scores)
            / len(judge_scores),
            "mean_brand_voice": sum(j.brand_voice for j in judge_scores)
            / len(judge_scores),
            "mean_helpfulness": sum(j.helpfulness for j in judge_scores)
            / len(judge_scores),
            "mean_safety": sum(j.safety for j in judge_scores) / len(judge_scores),
        }
    metrics["examples"] = per_example
    return metrics


def _confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        table[t][p] += 1
    return {k: dict(v) for k, v in table.items()}


def cohens_kappa(y1: Sequence[Any], y2: Sequence[Any]) -> float:
    """Cohen's kappa for two label sequences."""
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 0.0
    labels = sorted(set(y1) | set(y2), key=lambda x: str(x))
    agree = sum(a == b for a, b in zip(y1, y2)) / n
    c1 = Counter(y1)
    c2 = Counter(y2)
    chance = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    if chance == 1:
        return None  # Undefined when both raters use only one category.
    return (agree - chance) / (1 - chance)
