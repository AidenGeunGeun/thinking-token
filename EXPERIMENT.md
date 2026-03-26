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

## Phase 1 — Completed

- **Scope**: 10 telecom tasks x 3 models x 4 conditions x 1 trial = **120 task runs**
- **Goal**: qualitative exploration, not statistical analysis
- **GPU**: RunPod L40S ($0.87/hr), ~3hrs total
- **Conditions**: thinking_off, strip_all, window_3 (summarized), retain_all (summarized)
- **Summary format**: Formal third-person ("**Internal Note:** **Decision point:**...")

### Phase 1 Results

| Model | thinking_off | strip_all | window_3 | retain_all |
| ----- | ------------ | --------- | -------- | ---------- |
| 0.8B  | 70% (7/10)   | 33% (3/9) | 0% (0/9) | 11% (1/9)  |
| 4B    | 75% (6/8)    | 67% (6/9) | 20% (2/10) | TBD      |
| 9B    | TBD          | TBD       | TBD      | TBD        |

### Phase 1 Findings

#### Finding 1: Thinking hurts small models, less so at scale

Thinking generation at 0.8B is below the capability threshold — the model wastes tokens on low-quality reasoning, leaving less room for actual tool calls and responses. At 4B, the degradation is much smaller (-8pp vs -37pp), suggesting a capability threshold between 0.8B and 4B.

| Model | thinking_off → strip_all | Delta |
| ----- | ------------------------ | ----- |
| 0.8B  | 70% → 33%               | -37pp |
| 4B    | 75% → 67%               | -8pp  |

Notably, for specific tasks with app permissions (`break_app_sms_permission`), strip_all OUTPERFORMED thinking_off (1.0 vs 0.0). Thinking helps 4B on complex tasks — but only when the thinking is stripped cleanly from history.

#### Finding 2: `<think_summary>` leaks to user simulator (CRITICAL BUG)

The `<think_summary>` blocks were stored in assistant messages returned to τ²-bench. Since τ²-bench sends the full conversation history to both the agent AND the user simulator (GPT-OSS-20B), the user sim could see the agent's internal reasoning summaries.

**Evidence**: Every assistant message in window_3/retain_all conditions contained `<think_summary>` blocks in the stored `results.json`. Direct comparison of the same task across conditions confirmed the failure mode:

| Condition    | Reward | Behavior |
| ------------ | ------ | -------- |
| thinking_off | 1.0    | Clean conversation, user sim cooperates |
| strip_all    | 1.0    | Clean conversation, user sim cooperates |
| window_3     | 0.0    | User sim mimics formal style, fabricates results |
| retain_all   | 0.0    | Same degradation pattern |

**Observed user sim degradation when exposed to summaries**:
1. **Role confusion**: User sim starts asking the agent for the agent's phone number (mirroring the "Internal Note" about needing to identify the customer)
2. **Format mimicry**: User sim generates markdown tables, numbered step-by-step guides, and structured analysis — mimicking the formal summary format instead of playing a natural customer
3. **Result fabrication**: User sim claims tool calls succeeded without actually calling tools (e.g., fabricated `can_send_mms()` returning "Yes" when no tool call was made)
4. **Action paralysis**: Both sides get stuck in loops ("please run the steps" / "sure, let me know when you're done") with neither side taking action

**Root cause**: τ²-bench uses a single `state.messages` list visible to both agent and user sim. Our agent returned messages with `<think_summary>` blocks intact, so the user sim saw them.

**Fix for Phase 2**: Strip ALL thinking artifacts (`<think>` and `<think_summary>`) from the message returned to τ²-bench. Maintain a separate internal state for the agent that preserves summaries for retention.

#### Finding 3: Summary format is wrong

The summarizer produced formal third-person notes:
```
**Internal Note:**
- **Decision point:** Must identify the customer before troubleshooting.
- **Information used:** Policy mandates customer identification.
- **Conclusion:** Prompt the user to provide a phone number.
```

This doesn't read like the model's own thinking. Compare with how models naturally think:
```
The user wants help with MMS. I need to identify them first — policy says
phone number, customer ID, or name + DOB. They haven't given me any of
those yet. I'll ask for their phone number.
```

Even without the leakage bug, the formal format may confuse the agent model itself. At 0.8B, the model started mimicking the summary format in its own output (leaked `</think_summary>` closing tags in response text). The summary should sound like first-person stream-of-consciousness — the model recalling its own prior reasoning.

#### Finding 4: GPT-OSS-20B is a weak user simulator

Independent of the leakage bug, GPT-OSS-20B exhibited:
- Wrong DOB (1990-05-15 vs actual 1985-06-15) forcing fallback to phone lookup
- Saying "I don't have a tool" for tools it can call
- Calling tools the agent didn't request (toggling data off mid-troubleshooting)
- Generating garbage tool names with special token leakage (`can_send_mms<|channel|>commentary`)
- Empty responses causing retries

These affect all conditions equally (controlled noise), but reduce the signal-to-noise ratio of the experiment. A stronger user sim would give cleaner data.

---

## Phase 2 — Redesigned

Informed by Phase 1 findings. Key changes: fix leakage bug, add raw CoT condition, use natural summary format, cap thinking budget.

### Conditions

| # | Condition        | Thinking | Format in agent history    | Window | User sim sees |
|---|------------------|----------|----------------------------|--------|---------------|
| 1 | thinking_off     | OFF      | —                          | —      | clean text    |
| 2 | strip_all        | ON       | (not retained)             | 0      | clean text    |
| 3 | raw_window3      | ON       | raw `<think>` blocks       | 3      | clean text    |
| 4 | raw_retain       | ON       | raw `<think>` blocks       | all    | clean text    |
| 5 | summary_window3  | ON       | natural first-person summary | 3    | clean text    |
| 6 | summary_retain   | ON       | natural first-person summary | all  | clean text    |

**Critical invariant**: User sim ALWAYS sees clean text with no thinking artifacts. Thinking/summaries exist only in the agent's internal message copy.

### Summary Format (Phase 2)

Natural first-person stream-of-consciousness. The summary should read like the model recalling its own reasoning:

```
The customer wants help with MMS. I looked up their account — they're on
the Premium Plan with 15GB limit, but they've used 15.1GB. That's probably
why data is throttled. Before I refuel their data though, I should check
the device side: APN settings, Wi-Fi calling, app permissions. If the MMSC
URL is missing, that alone would block MMS regardless of data.
```

NOT formal structured notes. NOT bullet points with bold headers. The model should recognize this as "what I was thinking earlier."

### Thinking Budget

All thinking-on conditions capped at **8,192 max_tokens** (total output including thinking). Combined with `presence_penalty=1.5` (Qwen3.5 official recommendation for thinking mode), this prevents overthinking loops.

### Model Scale

Same 3 dense models: Qwen3.5-0.8B, 4B, 9B. All GGUF Q8_0 on llama.cpp.

### Matrix

3 models × 6 conditions × N tasks × T trials = TBD (depends on budget and Phase 1 completion)

### User Simulator

**Phase 1**: GPT-OSS-20B on Groq ($0.075/$0.30). Weak at tool calling, role confusion, result fabrication. Controlled noise (same across conditions) but reduced signal-to-noise.

**Phase 2**: DeepSeek V3.2 on OpenRouter ($0.26/$0.38). Better agentic tool calling, stronger instruction following. Cost increase is negligible (~$1-2 total for full experiment vs GPU rental).

### Smoke Test Protocol (MANDATORY before any GPU run)

Phase 1 wasted GPU hours because the leakage bug wasn't caught pre-deployment. Phase 2 requires passing ALL smoke checks before starting real runs:

1. **Agent view check**: Print the exact messages sent to the agent model. Verify thinking/summaries are present for retention conditions.
2. **User sim view check**: Print the exact messages stored in `state.messages` (what τ²-bench sends to user sim). Verify NO `<think>`, `<think_summary>`, or other thinking artifacts are visible.
3. **Summary format check**: For summarized conditions, print the generated summary. Verify it's first-person stream-of-consciousness, not formal structured notes.
4. **End-to-end single task**: Run 1 task per condition, inspect full conversation transcript. Verify user sim behaves naturally (no role confusion, no format mimicry).
5. **Token accounting**: Verify thinking tokens are counted correctly, context growth is reasonable.

Only after ALL checks pass: start the full matrix.

### Phase 2 Hypotheses (updated)

- **H1**: `strip_all` > `thinking_off` for 4B and 9B (thinking helps when stripped cleanly)
- **H2**: `raw_retain` > `strip_all` (raw CoT in history helps the model)
- **H3**: `summary_retain` ≈ `raw_retain` at lower context cost (summaries preserve useful signal)
- **H4**: `summary_retain` > `raw_retain` for long conversations (raw CoT accumulates noise; summaries stay concise)
- **H5**: `window_3` ≈ `retain_all` for both formats (recent thinking matters most)
- **H0**: No significant difference between retention strategies

## Infrastructure

- **GPU**: RunPod, any card with >=48GB VRAM (H200 SXM, H100 SXM, RTX PRO 6000 all work; cost is ~$4-7 total regardless of card choice since faster cards finish sooner)
- **Inference engine**: llama.cpp (llama-server) with GGUF Q8_0, KV cache Q8_0
- **Custom chat template**: `configs/chat_template.jinja` disables Qwen3.5's built-in thinking stripping (so our Python code controls retention, not the Jinja template)
- **User simulator**: Phase 1: GPT-OSS-20B on Groq. Phase 2: DeepSeek V3.2 on OpenRouter ($0.26/$0.38 per M tokens)
- **Thinking summarizer**: GPT-OSS-20B on Groq (cheap, summarization-only — quality sufficient for distillation)
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
