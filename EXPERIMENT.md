# Thinking Token Retention on tau2-bench

## Research Question

Does retaining thinking tokens in conversation history improve multi-turn agentic task performance?

## Benchmark

- Domain: tau2-bench telecom
- Scale: 114 base tasks
- Conversation length: average about 20 user turns per task, with some tasks reaching 61 user turns
- Usage: tau2-bench is used in public evaluations from OpenAI, Anthropic, and Google
- Evaluation: automated grounded evaluation via database state comparison
- License: MIT

Key references:
- tau2-bench paper: arxiv.org/abs/2506.07982
- tau2-bench repo: github.com/sierra-research/tau2-bench

## Turn Definition

A turn is everything between consecutive user messages. Within a single turn, the agent may make multiple model calls while reasoning and using tools. All thinking generated during that span belongs to the same turn.

## Models

All Phase 1 runs use the Qwen3.5 family on vLLM on a RunPod A100 80GB.

| Model | Notes |
| --- | --- |
| Qwen3.5-0.8B | 11.6% tau2-bench score, mostly fails, scaling floor |
| Qwen3.5-4B | 79.9% tau2-bench score, decent |
| Qwen3.5-9B | 79.1% tau2-bench score, solid |
| Qwen3.5-35B-A3B | MoE with 3B active parameters, score TBD |

## Independent Variables

### Primary

- Retention strategy:
  - `strip_all`
  - `window_3`
  - `retain_all`

### Secondary

- Thinking mode:
  - on (`enable_thinking=True`)
  - off (`enable_thinking=False`)
- Model scale:
  - 0.8B
  - 4B
  - 9B
  - 35B-A3B

When thinking is off, retention is irrelevant. That gives 4 models x 1 baseline = 4 conditions.

When thinking is on, we run 4 models x 3 retention strategies = 12 conditions.

Total: 16 configurations.

## Phase 1

- Scope: 10 telecom tasks x 16 configurations x 1 trial
- Total: 160 task runs
- Goal: qualitative exploration
- Analysis mode: read trajectories, compare behavior, spot failure-mode differences
- Budget target: about $5 on a single RunPod A100 80GB

## Phase 2

Future work only. If Phase 1 shows promising differences, expand to a larger task set, more trials, and formal statistical analysis. Exact scope is TBD.

## Infrastructure

- GPU: RunPod A100 80GB
- Inference engine: vLLM
- Reasoning flags: `--enable-reasoning --reasoning-parser qwen3`
- User simulator: GPT-OSS-20B on Groq
- Groq pricing reference: $0.075 / M input tokens, $0.30 / M output tokens

## Evaluation

- Primary benchmark metric: tau2-bench pass^k
- Secondary analysis: qualitative inspection of saved trajectories

## Hypotheses

- H1: retaining thinking tokens improves pass rate relative to stripping them
- H2: the benefit grows with conversation length
- H3: a sliding window captures most of the benefit at lower context cost
- H0: retention strategy has no significant effect
