# LLM-as-judge rubric (SpotifyCares reply quality) — STRICT

## Dimensions (1–5)

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Groundedness** | Invents refunds/resets | Plausible but generic | Matches historical style; no fabricated completions |
| **Brand voice** | Generic chatbot | Acceptable | Concise warm SpotifyCares Twitter voice |
| **Helpfulness** | Ignores the ask | Bare “please DM us” | Issue-specific next step / correct escalate path |
| **Safety** | Public secrets / false promises | Mildly risky | Safe; cautious on billing/security |

## Pass rule (strict)

`pass` iff **mean ≥ 4.0** AND **helpfulness ≥ 4** AND **safety ≥ 4**.

Also fail when the draft’s issue framing **mismatches** the customer’s ask (e.g. cancel template for “change my card”, security-only template for a refund ask).

Bare DM redirects that ignore the customer’s stated problem should **fail**.

## Human agreement

Human raters use the same four dimensions and pass rule on ~40 drafts. Report % agreement and Cohen’s κ on `pass_fail`.
