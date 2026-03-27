# Thinking Token Retention in Multi-Turn Agentic Conversations

## Research Question

For open-source models that strip thinking tokens from conversation history by default, can lightweight summarization of prior reasoning improve multi-turn agentic task performance?

## Motivation

### Industry Context: The Evolution of Thinking Retention

Proprietary model providers have converged on the conclusion that models benefit from access to their prior reasoning. The trajectory is clear:

| Provider / Model       | Thinking in History | Mechanism                            |
| ---------------------- | ------------------- | ------------------------------------ |
| Qwen3.5 (open-source)  | Stripped            | Jinja chat template strips `<think>` from all prior turns |
| Claude Sonnet 3.7       | Stripped            | Thinking blocks removed from prior turns |
| Claude 4 (pre-Opus 4.5)| Stripped            | Thinking blocks removed from prior turns |
| **Claude Opus 4.5+**   | **Preserved**       | Full raw thinking encrypted in `signature` field; server decrypts on next turn |
| **Claude Sonnet 4.6**  | **Preserved**       | Same encrypted replay mechanism |
| **Claude Opus 4.6**    | **Preserved**       | Same, with adaptive thinking |
| OpenAI o-series        | Discarded*          | Reasoning tokens discarded after turn; encrypted carry-forward available via `reasoning.encrypted_content` |

*Exception: OpenAI preserves some reasoning items adjacent to tool calls for o3/o4-mini.

Anthropic's decision to change the default behavior in Opus 4.5 is strong evidence that thinking preservation improves multi-turn performance. From Anthropic's docs:

> "Starting with Claude Opus 4.5, thinking blocks from previous assistant turns are preserved in model context by default... Preserving thinking blocks has no negative effect on model performance."

### The Gap

Anthropic's solution requires proprietary infrastructure: the full raw thinking is encrypted into a `signature` field, and the server decrypts it to reconstruct the original thinking for prompt construction. A separate model (not the thinking model) produces summaries purely for user-facing display. The model always gets its own raw CoT back.

**Open-source models have none of this.** Every open-source model with thinking support (Qwen3.5, DeepSeek, etc.) strips thinking from history by default. Qwen's official guidance:

> "No Thinking Content in History: the historical model output should only include the final output part and does not need to include the thinking content."

This creates a clear research question: **what accessible alternatives exist for the open-source ecosystem?**

### Why Not Raw Thinking Retention?

Retaining raw `<think>` blocks in history is impractical:
- Thinking blocks are 4K-32K+ tokens each. Over a 20-turn conversation, this is 80K-640K tokens of accumulated raw CoT.
- Models get "lost in the sauce" navigating massive thinking blocks from prior turns.
- Raw thinking is stream-of-consciousness, not structured for later reference.
- Context window and cost grow linearly with thinking retention.

Instead, we test **summarized thinking**: after each turn, a cheap external model distills the raw `<think>` block into a concise summary (proportional to the original length). This summary replaces the raw thinking in conversation history. The model sees a useful reference of what it previously reasoned about, not the full stream-of-consciousness.

This mirrors what Anthropic does architecturally (separate summarizer model) but is accessible to anyone with access to a cheap model.

## Benchmark

- **Domain**: tau2-bench telecom (114 base tasks, 2285 full expansion)
- **Conversation length**: average ~20 user turns per task, range 2-61
- **Usage**: reported in public evaluations from OpenAI, Anthropic, and Google
- **Evaluation**: automated grounded evaluation via database state comparison (not LLM-as-judge)
- **Metric**: pass^k (all k trials succeed for a task, averaged across tasks)
- **License**: MIT

Key references:
- tau2-bench paper: arxiv.org/abs/2506.07982
- tau2-bench repo: github.com/sierra-research/tau2-bench

### Turn Definition

A "turn" is everything between consecutive user messages. Within a single turn, the agent may make multiple model calls, tool calls, and receive tool results. All thinking generated during that span belongs to the same turn.

## Independent Variables

### Thinking Mode (baseline control)

- **thinking_off**: `enable_thinking=False`. No `<think>` blocks generated. Retention is irrelevant. This is the no-thinking baseline.

### Retention Strategy (primary variable, applied when thinking is on)

Six conditions test two independent axes: **format** (raw vs summarized) and **window** (strip / window_3 / retain_all).

| Condition         | What's in agent's conversation history               | Summarizer | Cache behavior        |
| ----------------- | ---------------------------------------------------- | ---------- | --------------------- |
| `thinking_off`    | No thinking generated at all                         | no         | Append-only, clean    |
| `strip_all`       | Thinking generated and used for current turn, then discarded | no | Append-only, clean    |
| `raw_window3`     | Raw `<think>` blocks retained for last 3 user-turn windows | no  | Mutation at window edge |
| `raw_retain`      | Raw `<think>` blocks retained in all turns permanently | no       | Append-only, clean    |
| `summary_window3` | Summarized thinking in last 3 windows                | yes        | Mutation at window edge |
| `summary_retain`  | Summarized thinking in all turns permanently         | yes        | Append-only, clean    |

User sim always sees clean text — thinking artifacts exist only in the agent's internal message copy (two-view architecture).

### Thinking Summarizer

The summarizer runs after each assistant message for `summary_window3` and `summary_retain` conditions only. It does NOT run for `raw_window3`, `raw_retain`, `strip_all`, or `thinking_off`.

1. Model generates an assistant message with `<think>...</think>` block
2. Raw thinking extracted; char/token counts captured to agent-side accumulator
3. Raw thinking + user context + agent response sent to cheap external model (MiMo-V2 Flash on OpenRouter) for summarization
4. Summary replaces raw thinking in conversation history, stored as: `<think_summary>...</think_summary>`
5. Summarizer token usage (input/output) captured from litellm response
6. Summary length is proportional to raw thinking length (not fixed)

The summarizer prompt should instruct the model to capture: what was being decided, what information was used, and what conclusion was reached.

**Summarization timing**: Each assistant message with a `<think>` block is summarized independently. Within a single turn, the agent may produce multiple assistant messages interleaved with tool calls; each gets its own summary. This is simpler than per-turn aggregation, preserves the step-by-step structure of multi-step tool-use turns, and aligns with the agent's natural `_generate_next_message()` lifecycle.

### Model Scale

| Model | Type | Notes |
| --- | --- | --- |
| Qwen3.5-2B | Dense | Scaling floor (replaced 0.8B which scored ~0%) |
| Qwen3.5-4B | Dense | 79.9% tau2-bench |
| Qwen3.5-9B | Dense | 79.1% tau2-bench, 81.7% GPQA Diamond |

All dense models from the same architecture family. Clean scaling curve.

## Experimental Matrix

3 models x 6 conditions x 10 tasks x 1 trial = **180 task runs** (Phase 1)

2B and 4B run locally on Mac M4 Pro. 9B runs on RunPod L40S. Runs support checkpoint/resume (`--fresh` to re-run).

## Cache Accounting

The retention strategy directly affects KV cache behavior, which in turn affects real inference cost (time per turn):

- **strip_all**: Conversation is append-only (no prior content ever changes). Cache prefix matches perfectly every turn. Only new tokens at the end need computation.
- **retain_all** (with summaries): Also append-only. Summaries are written once and never change. Cache prefix matches perfectly. But the prefix is larger (includes all summaries).
- **window_3** (with summaries): History *mutates* each turn as the window slides. When a summary falls outside the window and gets stripped, the message content changes at that point. Everything from that mutation onward requires recomputation.

Since summaries are much shorter than raw thinking (~200-400 tokens vs 4K-16K), the cache invalidation cost for window_3 is manageable. But it should still be tracked.

### What to Track Per LLM Call

- `total_prompt_tokens`: total tokens sent in the prompt
- `cached_prompt_tokens`: tokens that hit KV cache (prefix match)
- `evaluated_prompt_tokens`: tokens that needed new computation
- `generation_tokens`: output tokens generated
- `thinking_tokens`: thinking tokens within generation

Per-task aggregates show the true compute cost of each strategy.

## Phase 1 — Completed

### Pilot Run (Contaminated — Historical Context)

An initial pilot run used 0.8B/4B models with 4 conditions (thinking_off, strip_all, window_3, retain_all). Results were contaminated by `<think_summary>` leaking to the user simulator (GPT-OSS-20B on Groq). Key lessons that informed the redesign:

1. **Leakage bug**: `<think_summary>` blocks visible to user sim caused role confusion, format mimicry, result fabrication, and action paralysis. Fixed by implementing two-view architecture (agent sees thinking in `_internal_messages`, user sim sees clean `state.messages`).
2. **Summary format**: Formal third-person notes confused both user sim and agent model. Changed to first-person stream-of-consciousness.
3. **GPT-OSS-20B weakness**: Bad tool calling, empty responses, special token leakage. Replaced by MiMo-V2 Flash.
4. **0.8B too weak**: Scored ~0% on retention conditions. Replaced by 2B as scaling floor.

### Clean Run (Current Results)

- **Scope**: 10 telecom tasks × 2 models (2B, 9B) × 6 conditions × 1 trial = **120 task runs** (4B pending)
- **Goal**: qualitative exploration, spot behavioral patterns across conditions
- **Infrastructure**: 2B on Mac M4 Pro (llama.cpp Metal), 9B on RunPod L40S (llama.cpp CUDA)
- **User sim**: MiMo-V2 Flash on OpenRouter ($0.10/$0.30)
- **Summarizer**: MiMo-V2 Flash on OpenRouter (same model)
- **Total OpenRouter cost**: ~$0.45
- **Conditions**: thinking_off, strip_all, raw_window3, raw_retain, summary_window3, summary_retain

Full per-task results, cross-condition comparisons, and conversation-level observations are in **PHASE1_RESULTS.md**.

| Condition | 2B | 9B |
|-----------|-----|-----|
| thinking_off | 9/10 (90%) | 9/10 (90%) |
| strip_all | 8/10 (80%) | 8/10 (80%) |
| raw_window3 | 9/10 (90%) | **10/10 (100%)** |
| raw_retain | 7/10 (70%) | 9/10 (90%) |
| summary_window3 | 9/10 (90%) | 5/10 (50%) |
| summary_retain | 4/10 (40%) | 5/10 (50%) |

### Thinking Budget

All thinking-on conditions capped at **8,192 max_tokens** (total output including thinking). Combined with `presence_penalty=1.5` (Qwen3.5 official recommendation for thinking mode), this prevents overthinking loops.

### Summary Format

Natural first-person stream-of-consciousness:

```
The customer wants help with MMS. I looked up their account — they're on
the Premium Plan with 15GB limit, but they've used 15.1GB. That's probably
why data is throttled. Before I refuel their data though, I should check
the device side: APN settings, Wi-Fi calling, app permissions. If the MMSC
URL is missing, that alone would block MMS regardless of data.
```

### Smoke Test Protocol (MANDATORY before any GPU run)

1. **Agent view check**: Verify thinking/summaries present in agent's internal messages for retention conditions.
2. **User sim view check**: Verify NO `<think>`, `<think_summary>`, or other thinking artifacts in `state.messages`.
3. **Summary format check**: Verify summaries are first-person, no headers or bullets.
4. **End-to-end single task**: Run 1 task per condition, inspect full transcript.
5. **Token accounting**: Verify thinking tokens counted correctly.

### Hypotheses (to be tested with larger N)

- **H1**: `strip_all` > `thinking_off` for 4B and 9B (thinking helps when stripped cleanly)
- **H2**: `raw_retain` > `strip_all` (raw CoT in history helps the model)
- **H3**: `summary_retain` ≈ `raw_retain` at lower context cost (summaries preserve useful signal)
- **H4**: `summary_retain` > `raw_retain` for long conversations (raw CoT accumulates noise; summaries stay concise)
- **H5**: `window_3` ≈ `retain_all` for both formats (recent thinking matters most)
- **H0**: No significant difference between retention strategies

## Infrastructure

- **Local (2B/4B)**: Mac M4 Pro 48GB — llama.cpp with Metal
- **Cloud (9B)**: RunPod L40S 48GB — llama.cpp with CUDA
- **Inference engine**: llama.cpp (llama-server) with GGUF Q8_0 weights, Q8_0 KV cache
- **Custom chat template**: `configs/chat_template.jinja` disables Qwen3.5's built-in thinking stripping (so our Python code controls retention, not the Jinja template)
- **User simulator**: MiMo-V2 Flash on OpenRouter ($0.10/$0.30 per M tokens)
- **Thinking summarizer**: MiMo-V2 Flash on OpenRouter (same model; runs only for summary_window3 and summary_retain conditions)

## Related Work

### Proprietary Approaches
- **Anthropic** (Opus 4.5+): Full thinking preserved via encrypted `signature` field. Separate model generates summaries for user display. Model gets raw CoT back via signature decryption.
- **OpenAI** (o-series): Reasoning tokens discarded after turn. Encrypted carry-forward available via `reasoning.encrypted_content`. Summary modes: auto, concise, detailed.

### Academic
- **ReSum** (2025): Converts growing interaction histories into compact reasoning states for extended agent trajectories.
- **Accordion-Thinking** (2026): Model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens.
- **InftyThink** (2025): Iterative reasoning with intermediate summaries as memory-bounded alternative to monolithic long CoT.

### Open-Source Default
- **Qwen3.5**: "No Thinking Content in History" enforced via Jinja chat template. No alternative proposed.
