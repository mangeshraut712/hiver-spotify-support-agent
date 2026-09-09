"""Export blank human worksheets and validate actual reviews without inventing labels."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from eval.harness import cohens_kappa
from spotify_agent.intents import INTENTS


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def export():
    directory = ROOT / "eval/review"
    directory.mkdir(exist_ok=True)
    golden = read_rows(ROOT / "eval/golden/golden.jsonl")
    metrics = json.loads((ROOT / "results/metrics_agent.json").read_text())
    by_id = {r["id"]: r for r in metrics["examples"]}
    sheets = {
        "labels.csv": [dict(id=g["id"], customer_text=g["customer_text"],
                            intent="", escalate="", reviewer="", reviewed_at="", notes="") for g in golden],
        "replies.csv": [dict(id=g["id"], customer_text=g["customer_text"],
                             historical_reply=g["brand_reply_text"], draft=by_id[g["id"]]["reply"],
                             draft_sha256=digest(by_id[g["id"]]["reply"]),
                             groundedness="", brand_voice="", helpfulness="", safety="",
                             reviewer="", reviewed_at="", notes="")
                        for g in golden if by_id[g["id"]].get("judge")],
    }
    for name, rows in sheets.items():
        path = directory / name
        if path.exists():
            print(f"Preserved existing {path}")
            continue
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} blank reviews to {path}")


def validate(rows, expected):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Review IDs must match the expected set exactly, without duplicates")
    for row in rows:
        if any(not row.get(key, "").strip() for key in ("reviewer", "reviewed_at", "notes")):
            raise ValueError(f"Incomplete human review: {row['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "labels", "agreement"])
    args = parser.parse_args()
    if args.action == "export":
        return export()
    golden_path = ROOT / "eval/golden/golden.jsonl"
    golden = read_rows(golden_path)
    if args.action == "labels":
        rows = list(csv.DictReader((ROOT / "eval/review/labels.csv").open()))
        validate(rows, [g["id"] for g in golden])
        by_id = {r["id"]: r for r in rows}
        for g in golden:
            r = by_id[g["id"]]
            if r["intent"] not in INTENTS or r["escalate"].lower() not in ("true", "false"):
                raise ValueError(f"Invalid label: {g['id']}")
            if r["customer_text"] != g["customer_text"]:
                raise ValueError("Customer text changed during review")
            g.update(intent=r["intent"], escalate=r["escalate"].lower() == "true",
                     notes=r["notes"], annotation_source="human", reviewer=r["reviewer"], reviewed_at=r["reviewed_at"])
        golden_path.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in golden))
        print("Imported actual human labels; rerun evaluation before claiming scores")
    else:
        metrics = json.loads((ROOT / "results/metrics_agent.json").read_text())
        predictions = {r["id"]: r for r in metrics["examples"] if r.get("judge")}
        rows = list(csv.DictReader((ROOT / "eval/review/replies.csv").open()))
        validate(rows, predictions)
        human, judge = [], []
        for r in rows:
            p = predictions[r["id"]]
            if r["draft_sha256"] != digest(p["reply"]) or r["draft"] != p["reply"]:
                raise ValueError("Draft changed since human review; export a new worksheet")
            values = [int(r[key]) for key in ("groundedness", "brand_voice", "helpfulness", "safety")]
            if any(v < 1 or v > 5 for v in values):
                raise ValueError("Scores must be integers 1–5")
            human.append("pass" if sum(values) / 4 >= 3.5 and values[3] >= 4 else "fail")
            judge.append(p["judge"]["pass_fail"])
        result = dict(provenance="human_review_import", n=len(rows),
                      agreement=sum(a == b for a, b in zip(human, judge)) / len(rows),
                      kappa=cohens_kappa(human, judge))
        (ROOT / "results/human_agreement.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
