#!/bin/bash
set -euo pipefail

echo "=== Setting up thinking-tokens experiment ==="

run_as_root() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

REPO_DIR="${THINKING_TOKENS_HOME:-/workspace/thinking-tokens}"

run_as_root apt-get update
run_as_root apt-get install -y git curl

python3 - <<'PY'
import sys

major, minor = sys.version_info[:2]
if not (major == 3 and 12 <= minor < 14):
    raise SystemExit(f"Python 3.12 or 3.13 is required, found {major}.{minor}")
print(f"Python {major}.{minor} detected")
PY

python3 -m pip install --upgrade pip uv
python3 -m pip install --upgrade tau2-bench litellm

# Qwen3.5 requires vLLM nightly (main branch)
if ! python3 -c "import vllm" >/dev/null 2>&1; then
  python3 -m pip install "vllm" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
fi

if [[ ! -d "$REPO_DIR" ]]; then
  if [[ -n "${THINKING_TOKENS_REPO_URL:-}" ]]; then
    git clone "$THINKING_TOKENS_REPO_URL" "$REPO_DIR"
  else
    echo "Repo not found at $REPO_DIR and THINKING_TOKENS_REPO_URL is not set."
    exit 1
  fi
fi

python3 -m pip install -e "$REPO_DIR"

python3 -c "import src.register; print('Agent registered successfully')"
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
python3 -c "import os; assert os.environ.get('GROQ_API_KEY'), 'GROQ_API_KEY not set!'"

python3 - <<'PY'
from huggingface_hub import snapshot_download

# Pre-download all models to avoid wasting GPU time on downloads
models = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
]
for model_id in models:
    try:
        snapshot_download(model_id)
        print(f"Pre-downloaded {model_id}")
    except Exception as e:
        print(f"WARNING: Failed to download {model_id}: {e}")
        print(f"  Check if the model ID is correct on huggingface.co")
PY

echo "=== Setup complete ==="
echo "Repo: $REPO_DIR"
echo "Ready to run: python scripts/select_tasks.py && python scripts/run_phase1.py --dry-run"
