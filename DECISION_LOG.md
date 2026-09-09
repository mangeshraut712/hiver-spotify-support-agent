# Decision log

Non-obvious choices made while building the SpotifyCares support agent.

1. **Brand = SpotifyCares** — Smaller than Amazon/Apple, coherent product domain, enough volume (~43k brand tweets) for retrieval without drowning labeling capacity.
2. **Skip Banking77** — Banking intents do not transfer to Spotify support; using them would inflate intent metrics without helping reply quality.
3. **Subsample 10k pairs (seed 42)** — Full CSV is unnecessary for retrieval quality; keeps reproduce time under 15 minutes after download.
4. **TF-IDF retrieval over embeddings** — No extra embedding model/API; fast, local, good enough for Twitter-length text; easier to explain live.
5. **Retrieve then LLM-generate (no fine-tune)** — Fine-tuning would overfit 2017 style and block a 15-minute reproduce path.
6. **Eight intents, not 20+** — Coarse labels match routing decisions humans make; finer labels collapse into `other` under Twitter noise.
7. **Conservative escalate policy** — False escalate preferred over false auto-handle for billing/security/cancel/login.
8. **Single structured JSON LLM call** — Intent + reply + decision together so the draft is consistent with the routing choice.
9. **Exclude golden pair ids from retrieval at eval time** — Prevents trivial nearest-neighbor leakage of the historical reply for the same tweet.
10. **Stratified golden sampling** — Raw traffic is mostly `other`; stratification ensures rare intents appear in the 200-set.
11. **Author labels via reviewed rules (not the evaluated agent)** — LLM was optional for proposals; final labels come from author review script + spot checks.
12. **OpenAI-compatible client + env base URL** — Same code path for OpenAI or OpenRouter; free OpenRouter models used when paid credits were unavailable.
13. **Judge only a subset (40) in default headline runs** — Cost/latency control; intent/escalate still scored on all 200.
14. **Baselines share `AgentOutput` schema** — Fair comparison without metric plumbing differences.
15. **Cache LLM JSON by prompt hash** — Makes re-runs cheap and headline numbers stable across reproduce attempts.
16. **Policy layer forces escalate on sensitive intents** — Model drafts can be soft; routing policy should not be.
17. **Use OpenAI-compatible free models when paid credits unavailable** — document model IDs; support `AGENT_REPLY_MODE=retrieval` for offline metric reproduce.
18. **Hand-pass all 200 gold labels** with per-id overrides after reading every example (`scripts/12_hand_label_golden.py`).
19. **Hybrid agent: rule-based intent (stronger than keyword baseline) + composed issue-aware reply** — intent + escalate are the routing KPIs; compose beats bare DM copy under the strict judge.
20. **Strict judge (mean≥4, helpfulness≥4, issue-aligned)** — reject vanity 1.0; judge on stratified 40 (5/intent) so the sample isn’t security/billing skewed.
21. **Billing before security when payment/refund is the ask** — “card got hacked / charged 6×” and “refund of unauthorized charges” route to billing.
22. **Default `AGENT_REPLY_MODE=compose`** — free-tier LLM rate limits made full LLM drafting unreliable for offline prove; compose is reproducible and on-brand.