#!/usr/bin/env bash
# One-shot launcher: venv + deps + Ollama model + app
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
MODEL="${AICB_MODEL:-llama3.2}"
if command -v ollama >/dev/null; then
  (curl -s localhost:11434/api/tags >/dev/null || (ollama serve >/dev/null 2>&1 &) ; sleep 3)
  ollama pull "$MODEL" || true
else
  echo "Ollama not installed: Humanize will use rule-only mode (~14%). Get it at https://ollama.com"
fi
export AICB_MODEL="$MODEL"
python app.py
