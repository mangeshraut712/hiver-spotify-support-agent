"""Recompute archived metrics offline. Does not run an LLM or establish human gold."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from eval.harness import accuracy, macro_f1, binary_prf
from spotify_agent.intents import INTENTS

for system in ("trivial", "simple", "agent"):
    archived = json.loads((ROOT / f"results/metrics_{system}.json").read_text())
    rows = archived["examples"]
    truth = [r["gold_intent"] for r in rows]
    pred = [r["pred_intent"] for r in rows]
    result = dict(n=len(rows), intent_accuracy=accuracy(truth, pred),
                  intent_macro_f1=macro_f1(truth, pred, INTENTS),
                  escalate=binary_prf([r["gold_escalate"] for r in rows],
                                      [r["pred_decision"] == "escalate" for r in rows]))
    for key in ("intent_accuracy", "intent_macro_f1"):
        if abs(result[key] - archived[key]) > 1e-12:
            raise ValueError(f"Archived {system} {key} does not match predictions")
    if abs(result["escalate"]["f1"] - archived["escalate"]["f1"]) > 1e-12:
        raise ValueError("Escalation F1 mismatch")
    print(json.dumps({"system": system, "evidence": "archived_weak_labels_not_human_gold", **result}))
