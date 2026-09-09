#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  # Prefer existing OpenAI key from openclaw env if present
  if [[ -f "$HOME/.openclaw/.env" ]]; then
    KEY=$(grep -E '^OPENAI_API_KEY=' "$HOME/.openclaw/.env" | head -1 | cut -d= -f2-)
    if [[ -n "$KEY" ]]; then
      python3 - <<PY
from pathlib import Path
p = Path("$ROOT/.env")
text = p.read_text()
key = """$KEY"""
if "OPENAI_API_KEY=" in text:
    lines = []
    for line in text.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            lines.append("OPENAI_API_KEY=" + key)
        else:
            lines.append(line)
    p.write_text("\\n".join(lines) + "\\n")
PY
    fi
  fi
fi
echo "env ready"
