#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up thinking-tokens for macOS ==="

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$HOME/thinking-tokens-venv"
LLAMA_DIR="$HOME/llama.cpp"
MODEL_DIR="$HOME/thinking-tokens-models"
TAU2_REPO_DIR="$HOME/tau2-repo"
TAU2_BENCH_REF="17e07b1da2bbc0cadfddeea36412686e0604127b"
LITELLM_VERSION="1.82.6"
HUGGINGFACE_HUB_VERSION="0.33.0"

ensure_command() {
  local command_name="$1"
  local install_hint="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name"
    echo "$install_hint"
    exit 1
  fi
}

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are required. Running xcode-select --install..."
  xcode-select --install || true
  echo "Finish the Xcode Command Line Tools install, then re-run this script."
  exit 1
fi

ensure_command git "Install Git or Xcode Command Line Tools first."

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on macOS. Install it from https://brew.sh and re-run."
  exit 1
fi

for package in cmake git portaudio libsndfile uv; do
  if brew list "$package" >/dev/null 2>&1; then
    echo "Homebrew package already installed: $package"
  else
    brew install "$package"
  fi
done

ensure_command cmake "Install cmake with Homebrew: brew install cmake"
ensure_command uv "Install uv with Homebrew: brew install uv"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" pull --ff-only || true
fi

if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
  echo "Building llama.cpp with Metal support..."
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" \
    -DGGML_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(sysctl -n hw.ncpu)"
else
  echo "llama.cpp already built at $LLAMA_DIR/build/bin/llama-server"
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Existing venv found at $VENV_DIR, reusing"
else
  echo "Creating Python 3.12 virtual environment..."
  uv python install 3.12
  uv venv --python 3.12 "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Python: $(python --version) at $(command -v python)"

uv pip install \
  "tau2[all] @ git+https://github.com/sierra-research/tau2-bench.git@$TAU2_BENCH_REF" \
  "litellm==$LITELLM_VERSION" \
  "huggingface-hub==$HUGGINGFACE_HUB_VERSION"

uv pip install -e "$REPO_DIR"

if [[ ! -d "$TAU2_REPO_DIR/.git" ]]; then
  git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_REPO_DIR"
fi
git -C "$TAU2_REPO_DIR" fetch origin
git -C "$TAU2_REPO_DIR" checkout "$TAU2_BENCH_REF"

export LLAMA_SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
export TAU2_DATA_DIR="$TAU2_REPO_DIR"
mkdir -p "$MODEL_DIR"

echo ""
echo "=== Verification ==="
python -c "import src.register; print('  Agent registration: OK')"
python -c "import tau2; print('  tau2-bench: OK')"
python -c "import litellm; print(f'  litellm: {litellm.__version__}')"
echo "  llama-server: $LLAMA_SERVER_BIN"
"$LLAMA_SERVER_BIN" --version 2>&1 | sed -n '1p' || true

python - <<'PY'
import os

errors = []
if not os.environ.get("OPENROUTER_API_KEY"):
    errors.append("OPENROUTER_API_KEY not set — needed for OpenRouter user simulator and summarizer")
if not os.environ.get("HF_TOKEN"):
    errors.append("HF_TOKEN not set — may be needed for model downloads")
if errors:
    for error in errors:
        print(f"  WARNING: {error}")
else:
    print("  API keys: OK")
PY

echo ""
echo "=== Downloading local GGUF models ==="
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

model_dir = Path.home() / "thinking-tokens-models"
model_dir.mkdir(parents=True, exist_ok=True)
models = [
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q8_0.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf"),
]
for repo, filename in models:
    try:
        cached_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
        target_path = model_dir / filename
        if Path(cached_path) != target_path:
            shutil.copy2(cached_path, target_path)
        print(f"  {repo}/{filename}: OK (cache={cached_path}, local={target_path})")
    except Exception as exc:
        print(f"  {repo}/{filename}: FAILED — {exc}")
PY

echo ""
echo "Add this block to ~/.zshrc:"
cat <<EOF
export OPENROUTER_API_KEY=sk-or-...
export HF_TOKEN=hf_...
export LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN"
export TAU2_DATA_DIR="$TAU2_DATA_DIR"
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run task selection once first:"
echo 'python scripts/select_tasks.py'
echo ""
echo "# Run 2B locally:"
echo 'LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server \'
echo '  python scripts/run_phase1.py --model qwen35-2b'
echo ""
echo "# Run 4B locally:"
echo 'LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server \'
echo '  python scripts/run_phase1.py --model qwen35-4b'
echo ""
echo "# Verify pipeline first:"
echo 'python scripts/verify_pipeline.py --model qwen35-2b'
