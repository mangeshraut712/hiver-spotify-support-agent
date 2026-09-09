# Submission packet — Hiver SDE Intern take-home

Use this with the Notion form:
https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f?pvs=105

## Form fields (copy/paste)

| Field | Value |
|---|---|
| **Public repo URL** | https://github.com/mangeshraut712/hiver-spotify-support-agent |
| **Brand** | SpotifyCares |
| **Report** | See README.md “Report” section (+ DECISION_LOG.md) |
| **Headline intent (agent)** | 0.82 (simple 0.71, trivial 0.245) |
| **Headline escalate F1 (agent)** | 0.859 (simple 0.827, trivial 0.750) |
| **Judge pass rate** | 0.70 on stratified n=40 |
| **Human–judge agreement** | 90% pass/fail · Cohen’s κ ≈ 0.78 |
| **Golden set** | 200 hand-labelled examples · `eval/golden/` |
| **Reproduce** | README quick start · `AGENT_REPLY_MODE=compose` · &lt;15 min after data download |

## Short summary for the form (if there is a free-text box)

Built a SpotifyCares Twitter support agent: TF-IDF retrieval + stronger rule-based intent + issue-aware composed replies + escalate policy. Proven on a hand-labelled golden set of 200 with trivial/simple baselines. Agent beats simple on intent accuracy (0.82 vs 0.71) and escalate F1 (0.859 vs 0.827). Strict local judge (issue-aligned, helpfulness≥4) scores 0.70 pass on a stratified 40; human agreement 90% (κ≈0.78). Full report and decision log are in the repo README / DECISION_LOG.md.

## Deliverable checklist

- [x] Runnable pipeline + README &lt;15 min reproduce path
- [x] Golden set 150–250 (200) + sampling/label notes
- [x] Eval harness + trivial & simple baselines
- [x] LLM/local judge rubric + human agreement evidence
- [x] Report sections (framing, results, failures, misleading, next week)
- [x] Decision log
- [x] Public GitHub repo
- [x] Unit tests passing (10)
- [x] Notion form submitted (2026-09-09) — confirmation: “Your response has been submitted.”
  - Github: https://github.com/mangeshraut712/hiver-spotify-support-agent
  - Email: mbr63@drexel.edu
  - LinkedIn: https://www.linkedin.com/in/mangeshraut71298/
  - Phone: +91 7276819090
  - Form: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f

## Cite

Dataset: thoughtvector / Customer Support on Twitter (Kaggle).  
sklearn TF-IDF; OpenAI-compatible Chat Completions (OpenRouter optional).
