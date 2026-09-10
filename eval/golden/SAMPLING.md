# Evaluation sampling and provenance

The current 200 examples come from a 10,000-pair SpotifyCares subsample (seed 42) of thoughtvector/Customer Support on Twitter. `06_sample_golden_candidates.py` stratifies by keyword proxy intent, then fills the candidate pool. This overrepresents rare categories relative to traffic.

**These are automated label proposals, not a hand-labelled golden set.** Scripts 07/08/12 generated rules and per-ID overrides. The user confirmed that they performed no review. `annotation_source` has been corrected and reviewer claims removed. Prior files remain in the invalidated evidence archive.

The data has been used during development; even a later human review will not turn it into an untouched test set. Freeze a fresh customer-disjoint holdout for final performance claims.

To complete the assignment, a person should independently review each message against the taxonomy and escalation guideline, record their decisions and rationales in `eval/review/labels.csv`, and import with `python scripts/10_review.py labels`. Do not accept proposals automatically. Ambiguous, multilingual and mid-thread messages need explicit uncertainty notes. The importer cannot prove who typed a CSV; genuine human participation is still required.
