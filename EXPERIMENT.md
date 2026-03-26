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

The `window_3` and `retain_all` conditions use the **summarizer as infrastructure**: after each assistant message, raw `<think>` blocks are extracted and summarized by a cheap external model. The summary replaces the raw thinking in conversation history. The raw thinking is never placed directly in history. The `strip_all` condition discards thinking entirely (no summarization cost).

| Condition     | What's in conversation history                       | Cache behavior        |
| ------------- | ---------------------------------------------------- | --------------------- |
| `thinking_off`  | No thinking generated at all                         | Append-only, clean    |
| `strip_all`     | Thinking generated and used for current turn, then discarded (no summarization) | Append-only, clean    |
| `window_3`      | Summarized thinking retained for last 3 turns, stripped from older | Mutation at window edge |
| `retain_all`    | Summarized thinking retained in all turns permanently | Append-only, clean    |

### Thinking Summarizer

The summarizer is infrastructure, not an independent variable. It runs after every assistant message for all thinking-on conditions:

1. Model generates an assistant message with `<think>...</think>` block
2. Raw thinking extracted
3. Raw thinking sent to cheap external model (GPT-OSS-20B on Groq) for summarization
4. Summary replaces raw thinking in conversation history, stored as: `<think_summary>...</think_summary>`
5. Summary length is proportional to raw thinking length (not fixed)

The summarizer prompt should instruct the model to capture: what was being decided, what information was used, and what conclusion was reached.

**Summarization timing**: Each assistant message with a `<think>` block is summarized independently. Within a single turn, the agent may produce multiple assistant messages interleaved with tool calls; each gets its own summary. This is simpler than per-turn aggregation, preserves the step-by-step structure of multi-step tool-use turns, and aligns with the agent's natural `_generate_next_message()` lifecycle.

### Model Scale

| Model | Type | Notes |
| --- | --- | --- |
| Qwen3.5-0.8B | Dense | 11.6% tau2-bench, scaling floor |
| Qwen3.5-4B | Dense | 79.9% tau2-bench |
| Qwen3.5-9B | Dense | 79.1% tau2-bench, 81.7% GPQA Diamond |

All dense models from the same architecture family. Clean scaling curve.

## Experimental Matrix

3 models x 4 conditions x 10 tasks x 1 trial = **120 task runs** (Phase 1)

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

## Phase 1

- **Scope**: 10 telecom tasks x 3 models x 4 conditions x 1 trial = **120 task runs**
- **Goal**: qualitative exploration, not statistical analysis
- **Analysis**: read trajectories, compare behavior across conditions, spot failure-mode differences, verify summarizer quality
- **Budget target**: ~$5-7 on RunPod GPU + ~$1-2 Groq (user sim + summarizer)
- **Note**: Summarizer implementation is new; Phase 1 validates whether the approach works before scaling up

## Phase 2

Future work contingent on Phase 1 results:
- Expand to larger task set (full telecom domain or multiple domains)
- Multiple trials per configuration for pass^k analysis
- Formal statistical analysis (McNemar's test, Holm-Bonferroni correction, effect sizes with 95% CIs)
- Additional window sizes (2, 5, 10)
- Additional models (Qwen3.5-27B, 122B-A10B)
- Thinking budget as additional independent variable (4K, 8K, 16K, 32K)

## Infrastructure

- **GPU**: RunPod, any card with >=48GB VRAM (H200 SXM, H100 SXM, RTX PRO 6000 all work; cost is ~$4-7 total regardless of card choice since faster cards finish sooner)
- **Inference engine**: llama.cpp (llama-server) with GGUF Q4_K_M
- **Custom chat template**: `configs/chat_template.jinja` disables Qwen3.5's built-in thinking stripping (so our Python code controls retention, not the Jinja template)
- **User simulator**: GPT-OSS-20B on Groq
- **Thinking summarizer**: GPT-OSS-20B on Groq (same model, separate calls)
- **Groq pricing**: $0.075/M input, $0.30/M output

## Hypotheses

- **H1**: Retaining summarized thinking improves pass rate relative to stripping (retain_all > strip_all)
- **H2**: The benefit grows with conversation length (telecom's ~20 avg turns should show it)
- **H3**: A sliding window captures most of the benefit at lower context cost (window_3 ~ retain_all)
- **H0** (null): Retention strategy has no significant effect on task completion

Even a null result is publishable if methodology is sound. The Anthropic finding that full thinking preservation helps does not guarantee that summarized thinking helps the same way.

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
