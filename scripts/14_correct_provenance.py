"""Correct disclaimed human provenance while preserving prior artifacts for audit."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / "results/archive_before_provenance_correction"
    archive.mkdir(exist_ok=True)
    paths = [ROOT / "eval/golden/golden.jsonl", ROOT / "eval/golden/hand_review_workbook.jsonl",
             *ROOT.joinpath("results").glob("*.json"), ROOT / "results/agreement.jsonl"]
    for path in paths:
        if not path.exists():
            continue
        backup = archive / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                row["annotation_source"] = "automated_proposal_not_human_review"
                row.pop("reviewer", None)
                row.pop("reviewed_at", None)
                if "human_pass_fail" in row:
                    row["heuristic_a_pass_fail"] = row.pop("human_pass_fail")
                    row["heuristic_a"] = row.pop("human", {})
                    row["heuristic_b_pass_fail"] = row.pop("judge_pass_fail")
                    row["heuristic_b"] = row.pop("judge", {})
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        else:
            value = json.loads(path.read_text())
            value["evidence_status"] = "exploratory_automated_labels_not_submission_validation"
            if path.name == "agreement_summary.json":
                value = {"status": "invalidated", "reason": "Both raters were heuristic code; no human–LLM agreement exists."}
            if path.name == "headline.json":
                value["label_provenance"] = "automated_proposal_not_human_review"
                value.pop("agreement", None)
                for system in value.get("systems", {}).values():
                    if "judge" in system:
                        system["heuristic_reply_check"] = system.pop("judge")
            if path.name.startswith("metrics_"):
                if "judge" in value:
                    value["heuristic_reply_check"] = value.pop("judge")
                for row in value.get("examples", []):
                    if row.get("judge"):
                        row["heuristic_reply_check"] = row.pop("judge")
                    row["judge"] = None
            path.write_text(json.dumps(value, indent=2))
    (archive / "README.md").write_text("# Invalidated historical evidence\n\nThese files preserve previous claims for audit only. The user confirmed no human annotation was performed. Neither the human labels nor human–judge agreement claims are valid. Do not cite these as submission results.\n")


if __name__ == "__main__":
    main()
