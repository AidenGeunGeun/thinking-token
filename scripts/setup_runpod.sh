#!/bin/bash
set -euo pipefail

# =============================================================================
# RunPod setup for thinking-tokens experiment
#
# Prerequisites:
#   - RunPod GPU Pod (L40S 48GB recommended)
#   - Any PyTorch or CUDA template
#   - Volume size >= 100GB
#   - Environment variables set in RunPod UI:
#       GROQ_API_KEY=gsk_...
#       HF_TOKEN=hf_...
# =============================================================================

echo "=== Setting up thinking-tokens experiment ==="

REPO_DIR="/workspace/thinking-tokens"
VENV_DIR="/workspace/venv"
HF_CACHE="/workspace/hf_cache"
LLAMA_DIR="/workspace/llama.cpp"

# --- Persistent HuggingFace cache -------------------------------------------
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
mkdir -p "$HF_CACHE"

grep -q 'HF_HOME=' /root/.bashrc 2>/dev/null || {
  echo "export HF_HOME=$HF_CACHE" >> /root/.bashrc
  echo "export HUGGINGFACE_HUB_CACHE=$HF_CACHE" >> /root/.bashrc
}

# --- System packages --------------------------------------------------------
apt-get update && apt-get install -y git curl cmake build-essential

# --- Build llama.cpp with CUDA ---------------------------------------------
if [[ -f "$LLAMA_DIR/build/bin/llama-server" ]]; then
  echo "llama.cpp already built at $LLAMA_DIR"
else
  echo "Building llama.cpp with CUDA support..."
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" 2>/dev/null || {
    echo "Repo exists, pulling latest..."
    git -C "$LLAMA_DIR" pull --ff-only || true
  }
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
  echo "llama.cpp built: $LLAMA_DIR/build/bin/llama-server"
fi

# --- Python 3.12 via uv ----------------------------------------------------
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

# --- Install Python dependencies -------------------------------------------
echo "Installing tau2-bench and litellm..."
uv pip install tau2-bench litellm huggingface-hub

# --- Clone and install project ----------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Cloning repo..."
  git clone https://github.com/AidenGeunGeun/thinking-token.git "$REPO_DIR"
else
  echo "Repo already at $REPO_DIR, pulling latest..."
  git -C "$REPO_DIR" pull --ff-only || true
fi

uv pip install -e "$REPO_DIR"

# --- Shell setup ------------------------------------------------------------
grep -q 'workspace/venv/bin/activate' /root/.bashrc 2>/dev/null || {
  echo "source /workspace/venv/bin/activate" >> /root/.bashrc
  echo "cd /workspace/thinking-tokens" >> /root/.bashrc
  echo "export LLAMA_SERVER_BIN=/workspace/llama.cpp/build/bin/llama-server" >> /root/.bashrc
}

export LLAMA_SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"

# --- Verify installs --------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "import src.register; print('  Agent registration: OK')"
python -c "import tau2; print('  tau2-bench: OK')"
python -c "import litellm; print(f'  litellm: {litellm.__version__}')"
echo "  llama-server: $LLAMA_DIR/build/bin/llama-server"
"$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -1 || true

# Check API keys
python - <<'PY'
import os
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

# --- Pre-download GGUF models -----------------------------------------------
echo ""
echo "=== Downloading GGUF models (this takes a while) ==="
python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from huggingface_hub import hf_hub_download

models = [
    ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-Q4_K_M.gguf"),
]
for repo, filename in models:
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  {repo}/{filename}: OK ({path})")
    except Exception as e:
        print(f"  {repo}/{filename}: FAILED — {e}")
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
