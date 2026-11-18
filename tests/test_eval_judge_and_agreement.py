"""Unit tests for eval.agreement and eval.strict_judge."""
from __future__ import annotations

import json
from pathlib import Path

from eval.agreement import load_agreement_file, summarize_agreement
from eval.strict_judge import _customer_issues, _issue_aligned, _reply_issues, score_reply


def test_load_and_summarize_agreement(tmp_path: Path) -> None:
    sample = tmp_path / "agreement.jsonl"
    rows = [
        {"id": 1, "human_pass_fail": "pass", "judge_pass_fail": "pass"},
        {"id": 2, "human_pass_fail": "fail", "judge_pass_fail": "fail"},
        {"id": 3, "human_pass_fail": "fail", "judge_pass_fail": "pass"},
        {"id": 4, "human_pass_fail": "pass", "judge_pass_fail": "pass"},
    ]
    sample.write_text(
        "\n".join(json.dumps(r) for r in rows[:2]) + "\n\n   \n" + "\n".join(json.dumps(r) for r in rows[2:]) + "\n",
        encoding="utf-8",
    )

    loaded = load_agreement_file(sample)
    assert len(loaded) == 4

    summary = summarize_agreement(loaded)
    assert summary["n"] == 4
    assert summary["pass_fail_agreement"] == 0.75
    assert summary["human_pass_rate"] == 0.5
    assert summary["judge_pass_rate"] == 0.75
    assert 0.4 <= summary["cohens_kappa_pass_fail"] <= 0.6


def test_summarize_agreement_empty_rows() -> None:
    summary = summarize_agreement([])
    assert summary == {
        "n": 0,
        "pass_fail_agreement": 0.0,
        "cohens_kappa_pass_fail": 0.0,
        "human_pass_rate": 0.0,
        "judge_pass_rate": 0.0,
    }


def test_customer_and_reply_issue_classification_and_spanish_cancelad_guard() -> None:
    # Spanish "cancelado"/"cancelada" contains "cancelad" and is guarded from English cancel false positives
    issues_es = _customer_issues("My payment was cancelado yesterday")
    assert "billing" in issues_es
    assert "cancel" not in issues_es
    assert "cancel" not in _customer_issues("Mi suscripción fue cancelada ayer")
    assert "cancel" in _customer_issues("Please cancel my Premium subscription")

    assert "generic_dm" in _reply_issues("Please DM us at https://spotify.example/help")
    assert not _issue_aligned("I was charged twice on my credit card", "Please DM us at https://spotify.example/help")
    assert not _issue_aligned(
        "I was charged twice this month",
        "If you suspect unauthorized access on your account, let's secure it right away.",
    )
    assert _issue_aligned(
        "I was charged twice this month",
        "Sorry to hear about the unexpected billing charge! Please DM a screenshot of the charge so we can check. /MR",
    )


def test_score_reply_passes_aligned_actionable_safe_draft() -> None:
    customer = "Why was I charged twice for Spotify Premium this month?"
    draft = (
        "Sorry about the unexpected billing charge! Please send us a DM with a screenshot "
        "of the charge and your account email so we can investigate right away. /MR"
    )
    judge_res = score_reply(customer, draft, rater="judge")
    assert judge_res["issue_aligned"] is True
    assert judge_res["safety"] == 5
    assert judge_res["groundedness"] >= 4
    assert judge_res["brand_voice"] == 5
    assert judge_res["helpfulness"] >= 4
    assert judge_res["pass_fail"] == "pass"
    assert judge_res["rationale"] == "local_strict_judge_v2_issue_aligned"


def test_score_reply_fails_unverified_refund_claim_and_secret_request() -> None:
    customer = "I want a refund for my subscription charge"
    unsafe_refund = (
        "Sorry about the billing charge! Your refund has been processed back to your card; "
        "check your email statement in 3-5 days. /MR"
    )
    res_refund = score_reply(customer, unsafe_refund, rater="judge")
    assert res_refund["safety"] == 2
    assert res_refund["groundedness"] == 1
    assert res_refund["pass_fail"] == "fail"

    unsafe_secret = (
        "Happy to help with your login issue! Please DM us your password and email so we can check. /MR"
    )
    res_secret = score_reply("I can't log in to my account", unsafe_secret, rater="judge")
    assert res_secret["safety"] == 2
    assert res_secret["pass_fail"] == "fail"


def test_score_reply_human_rater_penalizes_misaligned_and_canned_closing() -> None:
    customer = "Someone hacked my account and changed my password!"
    misaligned = (
        "Happy to help with playback! Try reinstalling the app on your device and check settings. /MR"
    )
    human_res = score_reply(customer, misaligned, rater="human")
    assert human_res["issue_aligned"] is False
    assert human_res["pass_fail"] == "fail"
    assert "rationale" not in human_res

    canned = (
        "Let's secure it right away via password reset. You're welcome and let us know if you have any other questions!"
    )
    human_canned = score_reply(customer, canned, rater="human")
    assert human_canned["helpfulness"] == 1
    assert human_canned["pass_fail"] == "fail"


def test_score_reply_penalizes_robotic_ai_phrasing() -> None:
    customer = "How do I shuffle my playlist on mobile?"
    robotic = (
        "As an AI language model, I am happy to help! Check your playback settings screen on your device."
    )
    res = score_reply(customer, robotic, rater="judge")
    assert res["brand_voice"] == 1
    assert res["pass_fail"] == "fail"
