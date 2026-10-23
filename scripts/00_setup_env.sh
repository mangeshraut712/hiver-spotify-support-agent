#!/usr/bin/env bash
# Create a local .env from the example. Never copy API keys from other tools.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
fi
echo "env ready: put OPENAI_API_KEY in .env yourself; this script does not copy secrets"
