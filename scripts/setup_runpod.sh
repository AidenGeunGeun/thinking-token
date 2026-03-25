#!/bin/bash
set -euo pipefail

# =============================================================================
# RunPod setup for thinking-tokens experiment
#
# Prerequisites:
#   - RunPod GPU Pod (A100 80GB recommended)
#   - Any PyTorch or CUDA template (Python version doesn't matter — we use uv)
#   - Volume size ≥ 150GB (model weights are large)
#   - Environment variables set in RunPod UI:
#       GROQ_API_KEY=gsk_...
#       HF_TOKEN=hf_...
# =============================================================================

echo "=== Setting up thinking-tokens experiment ==="

REPO_DIR="/workspace/thinking-tokens"
VENV_DIR="/workspace/venv"
HF_CACHE="/workspace/hf_cache"

# --- Persistent HuggingFace cache -------------------------------------------
# Default /root/.cache gets wiped on pod restart. Redirect to /workspace.
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
mkdir -p "$HF_CACHE"

# Make it survive new SSH sessions
grep -q 'HF_HOME=' /root/.bashrc 2>/dev/null || {
  echo "export HF_HOME=$HF_CACHE" >> /root/.bashrc
  echo "export HUGGINGFACE_HUB_CACHE=$HF_CACHE" >> /root/.bashrc
}

# --- System packages --------------------------------------------------------
apt-get update && apt-get install -y git curl

# --- Python 3.12 via uv ----------------------------------------------------
# RunPod templates ship Python 3.10-3.11, but tau2-bench requires >=3.12.
# uv handles Python version management cleanly.
pip install uv 2>/dev/null || pip3 install uv

if [[ -d "$VENV_DIR" ]]; then
  echo "Existing venv found at $VENV_DIR, reusing"
  source "$VENV_DIR/bin/activate"
else
  echo "Creating Python 3.12 virtual environment..."
  uv python install 3.12
  uv venv --python 3.12 "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

echo "Python: $(python --version) at $(which python)"

# --- Install dependencies ---------------------------------------------------
echo "Installing tau2-bench and litellm..."
uv pip install tau2-bench litellm

echo "Installing vLLM (nightly — required for Qwen3.5)..."
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# --- Clone and install project ----------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Cloning repo..."
  git clone https://github.com/AidenGeunGeun/thinking-token.git "$REPO_DIR"
else
  echo "Repo already at $REPO_DIR, pulling latest..."
  git -C "$REPO_DIR" pull --ff-only || true
fi

uv pip install -e "$REPO_DIR"

# --- Add venv activation to bashrc ------------------------------------------
grep -q 'workspace/venv/bin/activate' /root/.bashrc 2>/dev/null || {
  echo "source /workspace/venv/bin/activate" >> /root/.bashrc
  echo "cd /workspace/thinking-tokens" >> /root/.bashrc
}

# --- Verify installs --------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "import src.register; print('  Agent registration: OK')"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import tau2; print('  tau2-bench: OK')"
python -c "import litellm; print(f'  litellm: {litellm.__version__}')"

# Check API keys
python - <<'PY'
import os, sys
errors = []
if not os.environ.get("GROQ_API_KEY"):
    errors.append("GROQ_API_KEY not set — needed for user simulator")
if not os.environ.get("HF_TOKEN"):
    errors.append("HF_TOKEN not set — may be needed for model downloads")
if errors:
    for e in errors:
        print(f"  WARNING: {e}")
else:
    print("  API keys: OK")
PY

# --- Pre-download models ----------------------------------------------------
echo ""
echo "=== Downloading model weights (this takes a while) ==="
python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
]
for model_id in models:
    try:
        snapshot_download(model_id, token=os.environ.get("HF_TOKEN"))
        print(f"  {model_id}: OK")
    except Exception as e:
        print(f"  {model_id}: FAILED — {e}")
PY

# --- Done -------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd /workspace/thinking-tokens"
echo "  python scripts/select_tasks.py"
echo "  python scripts/run_phase1.py --dry-run"
echo "  python scripts/run_phase1.py --model qwen35-0.8b --condition strip_all  # smoke test"
echo "  python scripts/run_phase1.py  # full run"
