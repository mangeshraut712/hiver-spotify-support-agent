"""Unit test contracts for evaluation metrics, confusion matrices, and judge score parsing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from eval.harness import (
    _confusion,
    _require_score,
    accuracy,
    binary_prf,
    cohens_kappa,
    macro_f1,
)


def test_accuracy_contracts():
    assert accuracy([], []) == 0.0
    assert accuracy(["cat", "dog"], ["cat", "dog"]) == 1.0
    assert accuracy(["cat", "dog"], ["dog", "dog"]) == 0.5
    with pytest.raises(ValueError, match="Expected equal-length"):
        accuracy(["cat"], ["cat", "dog"])


def test_macro_f1_boundaries():
    labels = ["positive", "negative"]
    assert macro_f1(["positive", "negative"], ["positive", "negative"], labels) == 1.0
    assert macro_f1(["positive", "positive"], ["negative", "negative"], labels) == 0.0
    assert macro_f1([], [], labels) == 0.0
    assert macro_f1(["positive"], ["positive"], []) == 0.0


def test_binary_prf_metrics():
    # All true
    res_true = binary_prf([True, True], [True, True])
    assert res_true["accuracy"] == 1.0
    assert res_true["precision"] == 1.0
    assert res_true["recall"] == 1.0
    assert res_true["f1"] == 1.0

    # All false
    res_false = binary_prf([False, False], [False, False])
    assert res_false["accuracy"] == 1.0
    assert res_false["precision"] == 0.0
    assert res_false["recall"] == 0.0
    assert res_false["f1"] == 0.0

    # Mixed
    res_mixed = binary_prf([True, False, True, False], [True, True, False, False])
    assert res_mixed["accuracy"] == 0.5
    assert 0.0 <= res_mixed["f1"] <= 1.0


def test_cohens_kappa_contracts():
    assert cohens_kappa([], []) == 0.0
    assert cohens_kappa(["same", "same"], ["same", "same"]) == 1.0
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    with pytest.raises(ValueError, match="Expected equal-length"):
        cohens_kappa(["a"], ["a", "b"])


def test_require_score_validation():
    # Valid ints
    assert _require_score(1, "safety") == 1
    assert _require_score(5, "helpfulness") == 5
    assert _require_score(3.0, "groundedness") == 3

    # Valid string ints
    assert _require_score("4", "brand_voice") == 4
    assert _require_score(" 2 ", "safety") == 2

    # Reject booleans
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(True, "safety")
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(False, "safety")

    # Reject floats with decimals
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(4.5, "helpfulness")

    # Reject out of bounds
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(0, "safety")
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(6, "safety")
    with pytest.raises(ValueError, match="Invalid judge dimension"):
        _require_score(None, "safety")


def test_confusion_matrix_structure():
    y_true = ["billing", "billing", "playback"]
    y_pred = ["billing", "other", "playback"]
    cm = _confusion(y_true, y_pred)
    assert cm["billing"]["billing"] == 1
    assert cm["billing"]["other"] == 1
    assert cm["playback"]["playback"] == 1
