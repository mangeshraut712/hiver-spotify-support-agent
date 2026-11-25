"""Tests for the human-review worksheet guardrails in scripts/10_review.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _load_review_module():
    spec = importlib.util.spec_from_file_location(
        "review_worksheet", ROOT / "scripts" / "10_review.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review = _load_review_module()


def test_csv_safe_neutralises_every_formula_prefix():
    """Every spreadsheet formula trigger must gain a leading quote on export."""
    payloads = [
        "=cmd|' /C calc'!A0",
        "+1+1",
        "-2+3",
        "@SUM(A1:A2)",
        "\t=1+1",
        "\r=1+1",
    ]
    for text in payloads:
        assert review.csv_safe(text) == "'" + text


def test_csv_safe_csv_restore_round_trip():
    """csv_restore must exactly invert csv_safe, including apostrophe-led text."""
    texts = [
        "=cmd|' /C calc'!A0",
        "@SpotifyCares my app keeps crashing",
        "-5 stars, my playlists are gone",
        "'- still broken after the update",
        "'@name same problem here",
        "'twas fine yesterday",
        "''double",
        "'",
        "plain tweet text",
        "",
    ]
    for text in texts:
        assert review.csv_restore(review.csv_safe(text)) == text


def test_csv_restore_leaves_unescaped_values_untouched():
    assert review.csv_restore(None) == ""
    assert review.csv_restore("ordinary text") == "ordinary text"
    assert review.csv_restore("'") == "'"


def test_validate_rejects_incomplete_or_mismatched_reviews():
    row = {"id": "sp_1", "reviewer": "ada", "reviewed_at": "2026-01-01", "notes": "ok"}
    review.validate([row], ["sp_1"])
    with pytest.raises(ValueError):
        review.validate([row, dict(row)], ["sp_1", "sp_2"])  # duplicate id
    with pytest.raises(ValueError):
        review.validate([dict(row, reviewer="")], ["sp_1"])  # incomplete review
    with pytest.raises(ValueError):
        review.validate([dict(row, id="sp_9")], ["sp_1"])  # unknown id


def test_score_accepts_only_integers_1_to_5():
    assert review._score("4") == 4
    assert review._score(5) == 5
    for bad in ("0", "6", "3.5", "", None, True):
        with pytest.raises(ValueError):
            review._score(bad)


def test_legacy_worksheet_cells_still_match_their_original_text():
    """Worksheets exported before the escape fix must keep importing."""
    for text in ["''still broken", "'twas fine", "@SpotifyCares help", "plain"]:
        legacy_cell = review._legacy_csv_safe(text)
        assert review.csv_matches(legacy_cell, text)
        assert review.csv_matches(review.csv_safe(text), text)


def test_edited_text_is_still_rejected():
    assert not review.csv_matches("''still fixed", "''still broken")
    assert not review.csv_matches("plain edited", "plain")
    assert not review.csv_matches(None, "plain")
