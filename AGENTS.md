# thinking-tokens

## What This Project Is

This repo studies whether keeping a model's `<think>...</think>` tokens in history helps multi-turn agent performance on tau2-bench telecom tasks. The project wraps stock tau2-bench orchestration with a custom agent that rewrites prior assistant messages before each LLM call.

## Current Status

Phase 1: building the tau2-bench integration and RunPod workflow for the first 10-task qualitative sweep.

## Environment

- Hardware target: RunPod L40S 48GB
- Inference: llama.cpp (llama-server) with GGUF Q4_K_M
- Python: 3.12+

## Dependencies

- `tau2-bench`
- `litellm`
- `llama.cpp` (built from source with CUDA)

## Architecture

```text
thinking-tokens/
├── EXPERIMENT.md         # Experimental design
├── AGENTS.md             # This file
├── configs/
│   ├── phase1.yaml       # Phase 1 experiment config
│   └── chat_template.jinja  # Modified Qwen3.5 template (preserves thinking in history)
├── src/
│   ├── __init__.py
│   ├── agent.py          # ThinkingRetentionAgent (subclasses tau2-bench LLMAgent)
│   ├── thinking.py       # Thinking token extraction/stripping utilities
│   └── register.py       # Agent factory registration with tau2-bench
├── scripts/
│   ├── setup_runpod.sh   # RunPod bootstrap (one-time)
│   ├── run_phase1.py     # Run all 16 configurations
│   └── select_tasks.py   # Pick telecom tasks for Phase 1
├── results/              # Output (gitignored)
└── .gitignore
```

```text
tau2-bench Orchestrator
        |
        v
ThinkingRetentionAgent
        |
        +--> copy state.messages
        +--> strip or retain prior <think> blocks by strategy
        +--> call stock tau2 generate()
        +--> store original assistant output back in tau2 state
```

## Commands

```bash
# Install the local package in editable mode
pip install -e .

# Pick and save the 10 telecom tasks for Phase 1
python scripts/select_tasks.py

# Print the full Phase 1 plan without executing
python scripts/run_phase1.py --dry-run

# Run one model / one condition slice
python scripts/run_phase1.py --model qwen35-4b --condition window_3

# Run the full Phase 1 sweep
python scripts/run_phase1.py

# One-time RunPod bootstrap
bash scripts/setup_runpod.sh

# Run the unit tests for thinking utilities
python -m unittest discover -s tests
```

## Conventions

- Use Python 3.12+ syntax and standard type hints.
- Keep tau2-bench integration thin; prefer composition around stock behavior instead of forking framework logic.
- Never mutate prior message history in place when applying retention strategies.
- Keep RunPod scripts idempotent where practical.
- Prefer small, explicit JSON and YAML outputs over custom binary artifacts.
