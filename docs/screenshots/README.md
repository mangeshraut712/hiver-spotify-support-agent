# Screenshots

Captured 21 September 2026 from this repository. Neither image is a mock of unrun software.

| File | What it shows | Secrets required |
| --- | --- | --- |
| `agent-compose.png` | `AGENT_REPLY_MODE=compose` CLI on “I was charged twice for Premium”, after building a TF-IDF index from the 40 bundled `data/sample/sample_spotify.csv` pairs. | None |
| `offline-replay.png` | `python scripts/11_replay_results.py` plus `pytest -q` (13 passed). Replay JSON is abbreviated in the image; full floats are in `results/metrics_*.json`. | None |

The compose-mode screenshot uses the sample index, not the full ~9.7k-pair retrieval corpus (that corpus is gitignored and needs a local Kaggle CSV).
