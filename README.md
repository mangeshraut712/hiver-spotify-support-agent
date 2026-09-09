# SpotifyCares AI Support Agent (Hiver SDE Intern Take-Home)

Turn messy Twitter support threads into a **working agent** and **prove** it works.

**Brand:** SpotifyCares  
**Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (subsample)  
**Stack:** Python · TF-IDF retrieval · OpenAI-compatible LLM (OpenAI or OpenRouter)

---

## Quick start (<15 minutes after dataset download)

```bash
# Prefer Python 3.10–3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
cp .env.example .env   # set OPENAI_API_KEY (and optional OPENAI_BASE_URL)

python scripts/01_prepare_data.py --brand SpotifyCares --max-pairs 10000
python scripts/02_build_index.py
python scripts/05_evaluate.py --systems trivial,simple --no-judge
AGENT_REPLY_MODE=compose python scripts/05_evaluate.py --systems agent --no-judge
python scripts/13_author_score_agreement.py   # stratified 40 + human/judge agreement
```

Headline numbers: [`results/headline.json`](results/headline.json)  
Golden set (200): [`eval/golden/golden.jsonl`](eval/golden/golden.jsonl) · [`eval/golden/SAMPLING.md`](eval/golden/SAMPLING.md)  
Decision log: [`DECISION_LOG.md`](DECISION_LOG.md)

```bash
python scripts/03_run_agent.py --system agent --text "I was charged twice for Premium"
AGENT_REPLY_MODE=retrieval python scripts/05_evaluate.py --systems trivial,simple,agent --no-judge
python scripts/13_author_score_agreement.py   # if metrics include judged drafts
pytest -q
```

### LLM providers

| Setup | `.env` |
|---|---|
| OpenAI | `OPENAI_API_KEY=...` · `OPENAI_AGENT_MODEL=gpt-4o-mini` · `OPENAI_JUDGE_MODEL=gpt-4o` |
| OpenRouter | `OPENAI_BASE_URL=https://openrouter.ai/api/v1` · e.g. `google/gemma-4-26b-a4b-it:free` |

Kaggle: `kagglehub` on first run, or set `TWCS_CSV_PATH` to a local `twcs.csv`.

---

## Report

### Problem framing — what “good” means

For SpotifyCares Twitter support, a good agent:

1. Routes each inbound tweet to one of **8 intents** a human would use for triage.
2. Drafts a **short, on-brand reply grounded** in historical SpotifyCares resolutions (often: ask for a DM, gather device/account identifiers, never invent completed refunds/resets).
3. **Escalates conservatively** on billing, cancel disputes, security, login lockouts, and Family/Duo membership; auto-handles only clear how-tos with retrieval support.

**Not built:** multi-turn state machines, live posting, refund tooling, fine-tuned classifiers, full 3M-tweet training, Banking77 transfer (domain mismatch).

### System

```
inbound tweet → TF-IDF retrieve top-k pairs
             → rule-based intent + escalate policy
             → compose issue-aware reply (default) | retrieval copy | LLM draft
```

| System | Behavior |
|---|---|
| **Trivial** | Always `other` + fixed DM template + always escalate |
| **Simple** | Keyword intent + copy nearest historical reply + escalate heuristics |
| **Agent** | Stronger rule-based intent + issue-aware composed reply + sensitive escalate policy |

### Results (golden n=200)

From a local reproduce run (`results/headline.json`, `AGENT_REPLY_MODE=compose`):

| System | Intent acc. | Intent macro-F1 | Escalate acc. | Escalate F1 | Judge pass rate (n=40 stratified) |
|---|---:|---:|---:|---:|---:|
| Trivial | 0.245 | 0.049 | 0.600 | 0.750 | — |
| Simple | 0.710 | 0.704 | 0.780 | 0.827 | — |
| Agent | **0.820** | **0.813** | **0.825** | **0.859** | **0.70** |

Human–judge agreement on the same 40 drafts: **90%** pass/fail agreement, **Cohen’s κ ≈ 0.78** (`results/agreement_summary.json`). Human pass rate 0.60 vs judge 0.70 — author is slightly stricter on generic templates.

**Takeaway:** after a full hand pass on gold labels, the hybrid agent **beats the keyword baseline on intent** (0.82 vs 0.71) **and escalate F1** (0.86 vs 0.83). Issue-aware composed replies raise the strict judge pass rate to **0.70** (stratified 40) without a vanity 1.0 — bare DM copies and misaligned issue framing still fail.

### Failure analysis — top 5 modes (real examples)

1. **Noisy gold / thank-you tweets labeled billing**  
   Gold `billing_charge` for *“Aha, the new window did the trick! So you will still get my money…”* — agent correctly treats it as resolved chit-chat (`other`/`auto`).  
   *Hypothesis:* author keyword `money` over-fires on gratitude tweets.

2. **Portuguese multi-device download complaint → playback escalate**  
   Gold `other`, pred `playback_bug`/`escalate` for a downloads/device-limit complaint.  
   *Hypothesis:* agent is operationally reasonable; coarse gold `other` under-specifies.

3. **Login+cancel SOS still ambiguous**  
   Some lockout+cancel tweets still collapse to a single label; policy prefers login when both cues fire, but gold can disagree.  
   *Hypothesis:* single-label taxonomy can’t hold joint intents.

4. **Family playlist transfer scored as howto**  
   *“can my daughter transfer her huge music collection… to our Premium family account?”* → `feature_howto`/`auto` vs gold `family_duo_plan`/`escalate`.  
   *Hypothesis:* “how do I transfer playlists” looks like howto; membership risk wants escalate.

5. **Strict judge / human fail generic or misaligned drafts**  
   Stratified judge pass **0.70**; human fails an extra ~10% of stock templates that never echo the customer’s specific ask.  
   *Hypothesis:* issue-class templates help, but specificity still matters under the strict rubric.

### What is misleading about my headline number?

- **Golden author = system author.** Labels are a full hand pass with per-id notes, but still one annotator (no IAA yet).
- **Stratified eval ≠ production mix.** Rare intents are oversampled vs raw Twitter traffic; judge sample is also stratified (5/intent).
- **Agent intent gains come from stronger rules + compose, not a giant LLM.** `AGENT_REPLY_MODE=compose` is the offline default; LLM drafting is optional when APIs allow.
- **Escalate F1 can still be gamed** by escalating almost everything (trivial recall=1.0). Prefer precision + accuracy for trust.
- **Judge pass rate 0.70 is intentional mid-band.** Mean≥4 and helpfulness≥4 plus issue-alignment; not a soft 1.0.
- **2017 exemplars** can be stale for 2026 product flows.

### What I’d do with one more week

1. Double-annotate 50 golden items for IAA on intent + escalate.  
2. Swap TF-IDF for local embeddings; report retrieval hit@k.  
3. Calibrate escalate separately (precision–recall curve) from reply generation.  
4. Cluster misses and add targeted few-shots / policy fixes.  
5. Tiny review UI for humans to accept/edit drafts before send.

### Live interview — ready answers

- **How I labeled:** full hand pass over all 200 (`scripts/12_hand_label_golden.py` + per-id overrides); notes in `eval/golden/SAMPLING.md`. One annotator — I’d add a second rater for IAA next.
- **Why hybrid beats simple:** stronger rules (billing-before-security, playback-before-Portuguese-cancelados, lockout+cancel → login) + issue-aware compose, not keyword+copy.
- **Why judge isn’t 1.0:** strict rubric (mean≥4, helpfulness≥4, issue-aligned) on a **stratified** 40; bare DM / wrong-issue drafts fail.
- **Free-tier reality:** OpenRouter rate limits → default `AGENT_REPLY_MODE=compose` for offline prove; LLM draft remains optional.

---

## Repo map

```
src/spotify_agent/   data prep, intents, retrieve, agent, baselines
scripts/             01 prepare → 02 index → 03 run → 05 evaluate → 09 agreement
eval/golden/         200 labeled examples + sampling notes
prompts/             agent + judge system prompts
results/             headline metrics + agreement (cache gitignored)
tests/               unit tests
```

## Cite / borrow

- Dataset: Stuart Axelbrooke / thoughtvector — *Customer Support on Twitter* (Kaggle).  
- sklearn `TfidfVectorizer`; OpenAI-compatible Chat Completions JSON mode.  
- Intent themes informed by public Spotify Help Center categories (billing, login, Premium, Family) — not used as a scraped corpus.

## License note

Dataset remains under Kaggle terms. This repo ships code, a tiny sample CSV, golden labels, and metrics — not the full `twcs.csv`.
