# thinking-tokens

## What This Project Is

This repo studies whether retaining *summarized* thinking tokens in conversation history improves multi-turn agent performance on tau2-bench telecom tasks.

Anthropic proved that full thinking preservation helps (changed default in Opus 4.5+), but their solution requires proprietary encrypted replay. Open-source models (Qwen3.5, etc.) strip thinking by default. This project tests whether lightweight summarization — accessible to anyone — can bridge that gap.

## Current Status

Phase 1: tau2-bench integration and RunPod workflow for 10-task qualitative sweep. Summarizer integration is the next implementation step.

## Environment

- Hardware target: RunPod, >=48GB VRAM (H200 SXM, H100 SXM, or RTX PRO 6000)
- Inference: llama.cpp (llama-server) with GGUF Q8_0, KV cache Q8_0
- Python: 3.12+

## Dependencies

- `tau2-bench`
- `litellm`
- `llama.cpp` (built from source with CUDA)

## Architecture

```text
thinking-tokens/
├── EXPERIMENT.md         # Experimental design and motivation
├── AGENTS.md             # This file
├── DATA_SCHEMA.md        # Output data format and cache accounting
├── configs/
│   ├── phase1.yaml       # Phase 1 experiment config
│   └── chat_template.jinja  # Modified Qwen3.5 template (disables built-in stripping; Python controls retention)
├── src/
│   ├── __init__.py
│   ├── agent.py          # ThinkingRetentionAgent (subclasses tau2-bench LLMAgent)
│   ├── thinking.py       # Thinking extraction, stripping, summarization, retention
│   └── register.py       # Agent factory registration with tau2-bench
├── scripts/
│   ├── setup_runpod.sh   # RunPod bootstrap (one-time)
│   ├── run_phase1.py     # Run all configurations
│   ├── select_tasks.py   # Pick telecom tasks for Phase 1
│   └── view_results.py   # Results table viewer
├── tests/                # Unit tests
├── results/              # Output (gitignored)
└── .gitignore
```

```text
tau2-bench Orchestrator
        |
        v
ThinkingRetentionAgent
        |
        +--> receive assistant response with <think> block
        +--> extract raw thinking
        +--> summarize via cheap model (GPT-OSS-20B on Groq)
        +--> store summary in conversation history (replacing raw thinking)
        +--> apply retention strategy (strip_all / window_3 / retain_all) on copies
        +--> next turn: model sees summarized thinking from prior turns
```

### Key Design Decision: Summarizer as Infrastructure

Raw `<think>` blocks (4K-32K+ tokens) are never placed in conversation history. Instead, every thinking-on condition uses a summarizer that distills thinking into concise summaries proportional to the original length. The retention strategy then controls *which* summaries remain in history. This mirrors Anthropic's architecture (separate summarizer model) but is accessible to the open-source ecosystem.

## Commands

```bash
# Install the local package in editable mode
pip install -e .

# Pick and save the 10 telecom tasks for Phase 1
python scripts/select_tasks.py

# Print the full Phase 1 plan without executing
python scripts/run_phase1.py --dry-run

# Run the preflight smoke test
python scripts/run_phase1.py --smoke

# Run one model / one condition slice
python scripts/run_phase1.py --model qwen35-4b --condition window_3

# Run the full Phase 1 sweep
python scripts/run_phase1.py

# View results
python scripts/view_results.py
python scripts/view_results.py --detail

# One-time RunPod bootstrap
bash scripts/setup_runpod.sh

# Run the unit tests
python -m unittest discover -s tests
```

## Conventions

- Use Python 3.12+ syntax and standard type hints.
- Keep tau2-bench integration thin; prefer composition around stock behavior instead of forking framework logic.
- Never mutate prior message history in place when applying retention strategies.
- Keep RunPod scripts idempotent where practical.
- Prefer small, explicit JSON and YAML outputs over custom binary artifacts.
- Raw thinking is never placed directly in conversation history; always summarize first.
