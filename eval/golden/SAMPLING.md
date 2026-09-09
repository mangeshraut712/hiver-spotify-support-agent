# Golden set sampling & labeling notes

## Sampling

- Source: SpotifyCares customer→brand reply pairs from `data/processed/spotify_pairs.csv`
  (10k subsample of Kaggle `thoughtvector/customer-support-on-twitter`, seed=42).
- Script: `scripts/06_sample_golden_candidates.py`
- Method: stratified by keyword proxy strata, ~24/stratum, fill to 220 candidates.
- Final size: **200** examples in `golden.jsonl`.

## Labeling (hand pass)

1. Exported candidates and **read every example in order (indices 0–199)**.
2. Applied author decisions via `scripts/12_hand_label_golden.py`:
   - careful primary rules for clear patterns
   - **explicit per-id overrides** for edge cases spotted while reading
3. Each row records `annotation_source=human_hand_pass`, `reviewer=author`, and a short `notes` rationale.

Labels:

- `intent` — one of 8 taxonomy labels  
- `escalate` — human should handle (true) vs safe auto-reply (false)  
- `brand_reply_text` — historical SpotifyCares reply (judge grounding only)

## Escalation policy used while labeling

- Always escalate: billing/refunds, cancel disputes, hacking/security, Family/Duo membership, login lockouts.
- Auto: clear how-tos / catalog questions / resolved thank-yous.
- Escalate on ambiguous “premium help” asks.

## What I’d tighten with more time

- Second annotator on 50 items for IAA.
- Multi-label for joint cancel+login cases.
- Drop gratitude / mid-thread tweets from the golden set (they inflate `other`).
