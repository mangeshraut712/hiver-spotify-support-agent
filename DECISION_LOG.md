# Decision log

1. Keep SpotifyCares as the brand so data preparation and comparisons remain scoped.
2. Use a 10,000-pair subsample with seed 42 to bound runtime.
3. Keep eight routing intents; expose mixed-intent limitations in the report.
4. Exclude evaluation customers before fitting TF-IDF to prevent customer overlap.
5. Exclude exact evaluation text duplicates as an additional retrieval safeguard.
6. Retain trivial and simple baselines under the same output schema.
7. Call default rules-plus-compose behavior deterministic; reserve LLM claims for actual API runs.
8. Keep sensitive-intent escalation as a policy separate from generation.
9. Preserve original evidence in an invalidated archive so corrections are auditable.
10. Mark existing labels automated because the user confirmed no human review occurred.
11. Disable scripts that manufactured human provenance rather than relabeling their results as genuine reviews.
12. Reject malformed LLM judge dimensions instead of silently filling scores.
13. Replace experiment summaries per run to avoid combining incompatible subsets.
14. Provide offline metric replay and blank human worksheets so evidence can be checked without provider calls.
15. Keep submission readiness blocked until actual human labels and human–LLM agreement exist; do not create a duplicate form submission.
