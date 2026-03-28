# Phase 1 Results

## Experiment Configuration

- **Benchmark**: τ²-bench telecom domain (Sierra, 2025), 10 selected tasks
- **Models**: Qwen3.5-2B, Qwen3.5-4B (Mac M4 Pro 48GB), Qwen3.5-9B (RunPod L40S 48GB)
- **Quantization**: GGUF Q8_0 weights, Q8_0 KV cache (uniform across all models)
- **Inference**: llama.cpp with custom Jinja chat template (disables Qwen3.5's built-in thinking stripping from history)
- **User simulator**: MiMo-V2 Flash via OpenRouter ($0.10/$0.30 per M tokens)
- **Summarizer**: MiMo-V2 Flash via OpenRouter (used only for summary conditions)
- **Trials**: 1 per task per condition
- **Total runs**: 180 (3 models × 6 conditions × 10 tasks)
- **Total API cost**: $0.25 (OpenRouter, user sim + summarizer combined)
- **Total compute**: 2B 2.83h + 4B 4.82h (Mac, free) + 9B 2.51h (RunPod, $2.19)
- **Analysis script**: `scripts/analyze_phase1.py`

### Conditions

| # | Condition | enable_thinking | History contains | Summarizer |
|---|-----------|-----------------|------------------|------------|
| 1 | thinking_off | false | nothing | no |
| 2 | strip_all | true | nothing (thinking used for current turn only) | no |
| 3 | raw_window3 | true | raw `<think>` blocks from last 3 user-turn windows | no |
| 4 | raw_retain | true | raw `<think>` blocks from all turns | no |
| 5 | summary_window3 | true | `<think_summary>` from last 3 windows | yes |
| 6 | summary_retain | true | `<think_summary>` from all turns | yes |

The two-view architecture ensures the user simulator always sees clean messages (no `<think>` or `<think_summary>` tags). Only the agent's prompt is modified by retention strategy.

### Tasks

Tasks selected by `scripts/select_tasks.py` — deterministic sort by descending complexity then ID. Issue count = number of device/account problems the customer has.

| Label | Issues | Persona | Description |
|-------|--------|---------|-------------|
| T1 | 2 | Hard | airplane_mode, broken_app_permissions |
| T2 | 3 | None | airplane_mode, bad_network_pref, data_exceeded |
| T3 | 3 | Easy | airplane_mode, bad_wifi_calling, roaming_off |
| T4 | 4 | None | airplane_mode, broken_app_permissions, data_exceeded, roaming_off |
| T5 | 5 | None | bad_wifi_calling, broken_apn, data_off, data_exceeded, sim_unseat |
| T6 | 6 | Easy | airplane_mode, bad_network_pref, bad_wifi_calling, data_exceeded, sim_unseat, roaming_on |
| T7 | 6 | Hard | bad_network_pref, bad_wifi_calling, broken_sms_permission, data_off, sim_unseat, roaming_off |
| T8 | 7 | Hard | airplane_mode, bad_network_pref, bad_wifi_calling, broken_apn, broken_app_permissions, sim_unseat, roaming_off |
| T9 | 8 | Easy | airplane_mode, bad_network_pref, bad_wifi_calling, broken_apn, broken_storage_permission, data_off, data_exceeded, sim_unseat |
| T10 | 8 | None | airplane_mode, bad_network_pref, bad_wifi_calling, broken_sms_permission, data_off, data_exceeded, sim_unseat, roaming_off |

---

## 1. Aggregate Results

| Condition | 2B | 4B | 9B | Avg | Wilson 95% CI (pooled) |
|-----------|-----|-----|-----|-----|------------------------|
| thinking_off | 9/10 | 8/10 | 9/10 | 87% | [73%, 94%] |
| **strip_all** | 8/10 | **10/10** | 8/10 | 87% | [73%, 94%] |
| **raw_window3** | 9/10 | 8/10 | **10/10** | **90%** | [77%, 96%] |
| **raw_retain** | 7/10 | **10/10** | 9/10 | 87% | [73%, 94%] |
| summary_window3 | 9/10 | 9/10 | 5/10 | 77% | [62%, 87%] |
| summary_retain | 4/10 | 8/10 | 5/10 | **57%** | [42%, 70%] |

Bold conditions are the three Phase 2 candidates. Bold cells are the per-model best.

**Category averages** (pooling conditions within each strategy):

| Category | Conditions | Avg pass rate |
|----------|-----------|---------------|
| Raw thinking retained | raw_window3, raw_retain | **88%** |
| No reasoning in history | thinking_off, strip_all | 87% |
| Summarized thinking | summary_window3, summary_retain | 67% |

**Per-model Wilson 95% confidence intervals** (z = 1.96):

| Condition | 2B | 4B | 9B |
|-----------|-----|-----|-----|
| thinking_off | [60%, 98%] | [49%, 94%] | [60%, 98%] |
| strip_all | [49%, 94%] | [72%, 100%] | [49%, 94%] |
| raw_window3 | [60%, 98%] | [49%, 94%] | [72%, 100%] |
| raw_retain | [40%, 89%] | [72%, 100%] | [60%, 98%] |
| summary_window3 | [60%, 98%] | [60%, 98%] | [24%, 76%] |
| summary_retain | [17%, 69%] | [49%, 94%] | [24%, 76%] |

The wide intervals confirm that N=10 per model cannot distinguish conditions separated by <20 percentage points.

---

## 2. Pairwise Statistical Tests

**McNemar's exact test** on paired binary outcomes (same task, different condition). Under H₀, the number of discordant pairs favoring condition A follows a Binomial(n_discordant, 0.5) distribution. We report two-sided p-values. The exact binomial form is appropriate for n_discordant < 25 (Agresti, 2002, p. 411).

*References*: McNemar (1947), *Psychometrika* 12(2):153–157. Wilson (1927), *JASA* 22(158):209–212. Agresti & Coull (1998), *The American Statistician* 52(2):119–126.

### Phase 2 candidate comparisons

| Comparison | b | c | Tied (+/−) | N | p-value | Direction |
|------------|---|---|-----------|---|---------|-----------|
| **strip_all vs raw_window3** (pooled) | 3 | 4 | 23/0 | 30 | 1.000 | ns |
| **strip_all vs raw_retain** (pooled) | 3 | 3 | 23/1 | 30 | 1.000 | ns |
| **raw_window3 vs raw_retain** (pooled) | 4 | 3 | 23/0 | 30 | 1.000 | ns |

Where b = row condition passes and column condition fails; c = row fails and column passes.

None of the three Phase 2 candidates are distinguishable from each other at N=30. This is the primary motivation for Phase 2: more tasks to resolve these comparisons.

### Summary vs raw (the significant finding)

| Comparison | b | c | Tied (+/−) | N | p-value | Sig. |
|------------|---|---|-----------|---|---------|------|
| **raw_retain vs summary_retain** (pooled) | 9 | 0 | 17/4 | 30 | **0.004** | ** |
| raw_window3 vs summary_window3 (pooled) | 7 | 3 | 20/0 | 30 | 0.344 | ns |
| raw_window3 vs summary_window3 (9B only) | 5 | 0 | 5/0 | 10 | 0.062 | † |

The raw_retain vs summary_retain comparison is statistically significant (p = 0.004). In 30 paired observations, raw_retain beat summary_retain 9 times and summary_retain beat raw_retain **zero** times. Summary is never better on any individual task at any model size.

### Other comparisons

| Comparison | b | c | N | p-value | Note |
|------------|---|---|---|---------|------|
| strip_all vs thinking_off (pooled) | 2 | 2 | 30 | 1.000 | Thinking doesn't help on average |
| raw_window3 vs strip_all (pooled) | 4 | 3 | 30 | 1.000 | ns |
| strip_all vs thinking_off (4B only) | 2 | 0 | 10 | 0.500 | Suggestive at 4B |

---

## 3. Per-Task Pass/Fail Matrix

### 2B (Qwen3.5-2B Q8_0, Mac M4 Pro)

| Task | Issues | Persona | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all | Total |
|------|--------|---------|:---------:|:-----:|:------:|:-------:|:------:|:-------:|:-----:|
| T1 | 2 | Hard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T2 | 3 | None | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 4/6 |
| T3 | 3 | Easy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T4 | 4 | None | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 4/6 |
| T5 | 5 | None | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 3/6 |
| T6 | 6 | Easy | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | 4/6 |
| T7 | 6 | Hard | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 5/6 |
| T8 | 7 | Hard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T9 | 8 | Easy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T10 | 8 | None | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | 2/6 |
| **Total** | | | **9** | **8** | **9** | **7** | **9** | **4** | |

### 4B (Qwen3.5-4B Q8_0, Mac M4 Pro)

| Task | Issues | Persona | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all | Total |
|------|--------|---------|:---------:|:-----:|:------:|:-------:|:------:|:-------:|:-----:|
| T1 | 2 | Hard | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 5/6 |
| T2 | 3 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T3 | 3 | Easy | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 5/6 |
| T4 | 4 | None | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5/6 |
| T5 | 5 | None | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 5/6 |
| T6 | 6 | Easy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T7 | 6 | Hard | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | 4/6 |
| T8 | 7 | Hard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T9 | 8 | Easy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| T10 | 8 | None | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | 4/6 |
| **Total** | | | **8** | **10** | **8** | **10** | **9** | **8** | |

### 9B (Qwen3.5-9B Q8_0, RunPod L40S)

| Task | Issues | Persona | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all | Total |
|------|--------|---------|:---------:|:-----:|:------:|:-------:|:------:|:-------:|:-----:|
| T1 | 2 | Hard | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 5/6 |
| T2 | 3 | None | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 4/6 |
| T3 | 3 | Easy | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 5/6 |
| T4 | 4 | None | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5/6 |
| T5 | 5 | None | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 5/6 |
| T6 | 6 | Easy | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 5/6 |
| T7 | 6 | Hard | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 3/6 |
| T8 | 7 | Hard | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 4/6 |
| T9 | 8 | Easy | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 4/6 |
| T10 | 8 | None | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 4/6 |
| **Total** | | | **9** | **8** | **10** | **9** | **5** | **5** | |

### Task consistency across all 18 model × condition runs

| Task | Issues | Pass rate | Bar |
|------|--------|-----------|-----|
| T1 | 2 | 18/18 (100%) | ██████████████████ |
| T2 | 3 | 14/18 (78%) | ██████████████░░░░ |
| T3 | 3 | 16/18 (89%) | ████████████████░░ |
| T4 | 4 | 14/18 (78%) | ██████████████░░░░ |
| T5 | 5 | 13/18 (72%) | █████████████░░░░░ |
| T6 | 6 | 15/18 (83%) | ███████████████░░░ |
| T7 | 6 | 12/18 (67%) | ████████████░░░░░░ |
| T8 | 7 | 16/18 (89%) | ████████████████░░ |
| T9 | 8 | 16/18 (89%) | ████████████████░░ |
| T10 | 8 | 11/18 (61%) | ███████████░░░░░░░ |

T1 (2 issues) is universally easy (100%). T7 (6 issues, Hard persona) and T10 (8 issues, None persona) are the most discriminating tasks (67% and 61%). Task difficulty does not monotonically increase with issue count — T8 and T9 (7–8 issues) are easier than T7 (6 issues) and T10 (8 issues), suggesting persona and specific issue combinations matter.

---

## 4. Failure Mode Analysis

Every failure across all 180 runs falls into one of three categories:

| Failure type | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all |
|--------------|-----------|-------|--------|---------|--------|---------|
| **Capability** (missed required actions) | 3 | 3 | **0** | 2 | 4 | **11** |
| **Infrastructure** (too_many_errors / max_steps) | 1 | 1 | 2 | 2 | 3 | 1 |
| **Env-only** (all actions done, assertion failed) | 0 | 0 | 1 | 0 | 0 | 1 |
| **Total failures** | 4 | 4 | 3 | 4 | 7 | 13 |

**raw_window3 has zero capability failures.** Its 3 failures are all infrastructure crashes (too_many_errors) or env assertion edge cases where the agent completed every required action. **summary_retain has 11 capability failures out of 13** — the agent genuinely cannot complete required actions when summaries accumulate in history.

### Partial action rewards

The fraction of required actions the agent successfully completed, averaged per condition:

| Condition | 2B | 4B | 9B |
|-----------|-----|-----|-----|
| thinking_off | 98% | 90% | 97% |
| strip_all | 91% | 100% | 93% |
| **raw_window3** | **100%** | **100%** | **100%** |
| raw_retain | 95% | 100% | 90% |
| summary_window3 | 92% | 100% | 88% |
| summary_retain | 84% | 85% | 64% |

raw_window3 achieves 100% partial action reward across all three models — even when it "fails" (reward=0), it completed every required action. Its failures are environmental, not behavioral. summary_retain at 9B averages only 64% of required actions completed.

---

## 5. Token Usage

Per-message `usage` fields from litellm provide token counts for every LLM call.

### Agent tokens per task (average)

| Condition | 2B prompt | 2B compl. | 4B prompt | 4B compl. | 9B prompt | 9B compl. |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| thinking_off | 302,867 | 3,431 | 227,542 | 2,516 | 240,262 | 2,457 |
| strip_all | 179,150 | 4,903 | 196,881 | 6,241 | 181,161 | 5,558 |
| raw_window3 | 199,030 | 5,000 | 172,071 | 4,073 | 211,913 | 4,790 |
| raw_retain | 198,770 | 4,998 | 217,935 | 5,305 | 197,996 | 4,914 |
| summary_window3 | 271,601 | 6,991 | 170,736 | 5,229 | 106,353 | 3,110 |
| summary_retain | 370,488 | 7,914 | 209,883 | 6,030 | 150,075 | 4,688 |

### Thinking overhead (agent completion tokens vs thinking_off baseline)

| Condition | 2B | 4B | 9B |
|-----------|-----|-----|-----|
| thinking_off | baseline | baseline | baseline |
| strip_all | 1.4× | 2.5× | 2.3× |
| raw_window3 | 1.5× | 1.6× | 1.9× |
| raw_retain | 1.5× | 2.1× | 2.0× |
| summary_window3 | 2.0× | 2.1× | 1.3× |
| summary_retain | 2.3× | 2.4× | 1.9× |

Summary conditions generate *more* agent completion tokens than raw conditions, not fewer. The summarizer adds overhead (an extra API call per message with thinking) but does not reduce the model's own thinking output. At 2B, summary_retain generates 2.3× the baseline tokens while achieving only 40% pass rate.

---

## 6. Duration and Cost

### Seconds per task (average)

| Condition | 2B | 4B | 9B |
|-----------|------|------|------|
| thinking_off | 112s | 174s | 103s |
| strip_all | 117s | 263s | 140s |
| raw_window3 | 155s | 254s | 198s |
| raw_retain | 138s | 298s | 131s |
| summary_window3 | 260s | 373s | 188s |
| summary_retain | 238s | 374s | 145s |

### Cost efficiency (seconds per successful task)

| Condition | 2B | 4B | 9B |
|-----------|------|------|------|
| thinking_off | 124s | 217s | 114s |
| strip_all | 147s | 263s | 176s |
| raw_window3 | 172s | 318s | 198s |
| raw_retain | 197s | 298s | 145s |
| summary_window3 | 289s | 414s | 376s |
| summary_retain | 594s | 467s | 290s |

raw_window3 costs ~1.4× thinking_off in wall-clock time per successful task. summary_retain costs 2–5× and performs worst.

### Compute costs

| Model | Hardware | Hours | Cost |
|-------|----------|-------|------|
| 2B | Mac M4 Pro | 2.83h | $0 |
| 4B | Mac M4 Pro | 4.82h | $0 |
| 9B | RunPod L40S ($0.87/hr) | 2.51h | $2.19 |
| API | OpenRouter MiMo-V2 Flash | — | $0.25 |
| **Total** | | **10.16h** | **$2.44** |

---

## 7. Task Difficulty × Condition Interaction

### By issue count

| Difficulty | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all |
|------------|-----------|-------|--------|---------|--------|---------|
| Easy (2–3 issues, n=9) | 89% | 78% | **100%** | 89% | 100% | 78% |
| Medium (4–6 issues, n=12) | 92% | 92% | 75% | 83% | 67% | **42%** |
| Hard (7–9 issues, n=9) | 78% | 89% | **100%** | 89% | 67% | **56%** |

raw_window3 maintains 100% on both easy and hard tasks. summary_retain collapses from 78% (easy) to 42% (medium) to 56% (hard). The gap between raw and summary widens on complex tasks with more conversation turns — consistent with the hypothesis that summaries lose procedural context over extended interactions.

### By persona

| Persona | think_off | strip | raw_w3 | raw_all | sum_w3 | sum_all |
|---------|-----------|-------|--------|---------|--------|---------|
| None (n=12) | 75% | 75% | 92% | 75% | 67% | 50% |
| Easy (n=9) | 100% | 89% | 89% | 100% | 78% | 67% |
| Hard (n=9) | 89% | 100% | 89% | 89% | 89% | 56% |

None-persona tasks are hardest for summary conditions (50% for summary_retain). The user simulator with no persona instruction may behave less predictably, amplifying the effect of conversation history quality.

---

## 8. Termination Reasons

| | 2B | | | 4B | | | 9B | | |
|---|---|---|---|---|---|---|---|---|---|
| **Condition** | user_stop | too_many | max_steps | user_stop | too_many | max_steps | user_stop | too_many | max_steps |
| thinking_off | 10 | — | — | 9 | 1 | — | 10 | — | — |
| strip_all | 9 | 1 | — | 10 | — | — | 10 | — | — |
| raw_window3 | 9 | 1 | — | 9 | 1 | — | 10 | — | — |
| raw_retain | 8 | 1 | 1 | 10 | — | — | 10 | — | — |
| summary_window3 | 10 | — | — | 9 | 1 | — | 8 | 2 | — |
| summary_retain | 9 | 1 | — | 10 | — | — | 10 | — | — |

`user_stop` = customer ended conversation (satisfied or gave up). `too_many_errors` = repeated invalid tool calls or message format errors. `max_steps` = hit conversation length limit. All conditions produce occasional infrastructure failures.

---

## 9. Conversation Structure

### Messages and tool calls per task (average)

| Condition | 2B msgs | 2B tools | 4B msgs | 4B tools | 9B msgs | 9B tools |
|-----------|---------|----------|---------|----------|---------|----------|
| thinking_off | 98 | 11.4 | 96 | 5.7 | 90 | 5.6 |
| strip_all | 76 | 6.0 | 92 | 6.9 | 69 | 5.7 |
| raw_window3 | 90 | 5.4 | 80 | 5.8 | 91 | 5.8 |
| raw_retain | 108 | 5.5 | 88 | 6.0 | 80 | 6.0 |
| summary_window3 | 83 | 10.9 | 94 | 5.2 | 65 | 4.6 |
| summary_retain | 101 | 17.0 | 77 | 6.7 | 50 | 6.6 |

2B/summary_retain averages 17.0 tool calls per task (3× the ~5–6 tool calls in raw conditions), indicating the 2B model flails when summaries accumulate. 9B/summary_retain has the shortest conversations (50 messages) combined with the worst pass rate — consistent with premature escalation to human agents observed in conversation logs.

### Tool calls: passed vs failed tasks

| Condition | Passed (avg tools) | Failed (avg tools) |
|-----------|-------------------|--------------------|
| thinking_off | 7.9 (n=26) | 5.2 (n=4) |
| strip_all | 6.3 (n=26) | 5.5 (n=4) |
| raw_window3 | 5.9 (n=27) | 3.7 (n=3) |
| raw_retain | 5.8 (n=26) | 6.2 (n=4) |
| summary_window3 | 7.7 (n=23) | 4.1 (n=7) |
| summary_retain | 11.6 (n=17) | 8.1 (n=13) |

summary_retain successes require 11.6 tool calls on average (vs 5–6 for raw conditions), suggesting the model compensates for degraded context by making more attempts.

---

## 10. Model Scaling

| Model | Avg pass rate | Best condition | Worst condition | Spread |
|-------|--------------|----------------|-----------------|--------|
| 2B | 77% | thinking_off (90%) | summary_retain (40%) | 50pp |
| 4B | 88% | strip_all, raw_retain (100%) | thinking_off, raw_window3, summary_retain (80%) | 20pp |
| 9B | 77% | raw_window3 (100%) | summary_window3, summary_retain (50%) | 50pp |

4B is the most robust model (20pp spread), while 2B and 9B are equally sensitive to condition choice (50pp spread). The best condition shifts across scales: 2B favors thinking_off, 4B favors strip_all/raw_retain, 9B favors raw_window3.

---

## 11. Key Findings

### Finding 1: Summarization actively harms performance

summary_retain is the worst condition at every model size (40%, 80%, 50%). Pooling across models, raw_retain beats summary_retain on 9 tasks and loses on 0 (McNemar's exact p = 0.004). This is the only statistically significant result in the dataset.

The failure mechanism is capability degradation, not infrastructure: 11 of 13 summary_retain failures involve the agent missing required actions. Summaries preserve diagnostic reasoning ("the customer's data is exceeded") but collapse procedural state ("I already called refuel_data with 2GB"). On multi-step agentic tasks with irreversible actions, this is catastrophic.

### Finding 2: raw_window3 has the strongest overall signal

raw_window3 achieves the highest average pass rate (90%), the only perfect score at any model size (10/10 at 9B), 100% partial action reward across all models (zero capability failures), and maintains 100% pass rate on both easy and hard task categories.

However, raw_window3 is not statistically distinguishable from strip_all or raw_retain at this sample size (all pairwise p = 1.0). Resolving this is the goal of Phase 2.

### Finding 3: Summaries cost more and perform worse

Summary conditions generate more agent completion tokens than raw conditions (2.0–2.3× baseline vs 1.5–2.1×) and take longer per task (238–374s vs 131–298s). The summarizer adds latency and API cost without reducing the model's own thinking output. The cost-performance frontier is strictly dominated: raw conditions achieve higher accuracy at lower cost.

### Finding 4: Confidence intervals preclude fine-grained claims

With N=10 per model, the 95% Wilson CI for 90% is [60%, 98%]. The only resolvable contrasts are those with large effect sizes (raw vs summary, ~30pp). Distinguishing strip_all (87%) from raw_window3 (90%) or raw_retain (87%) requires substantially more observations.

---

## 12. Limitations

1. **Small N**: 10 tasks × 1 trial per cell. Only the summary vs raw contrast achieves statistical significance.
2. **Task selection bias**: 10 tasks selected by complexity sort, not random sample of the 114 telecom base tasks. Findings may not generalize to the full distribution.
3. **Single trial**: Stochastic variation in model generation and user simulator behavior is not captured. The same task under the same condition may produce different outcomes on re-run.
4. **No agent-view logging**: The agent's internal `_internal_messages` (containing raw `<think>` and `<think_summary>` blocks) are not saved to disk. Token-level analysis of thinking content relies on message-level `usage` fields, not direct measurement of thinking block sizes.
5. **User simulator quality**: MiMo-V2 Flash occasionally produces invalid tool calls or format errors, contributing to `too_many_errors` terminations distributed across all conditions.
6. **Single model family**: All models are Qwen3.5 (dense, same architecture). Results may not transfer to other architectures or model families.

---

## 13. Phase 2 Implications

Phase 1 establishes three signals worth pursuing:

1. **Summary conditions can be dropped.** The evidence against summarization is strong (p = 0.004) and consistent across all model sizes. Phase 2 need not include summary conditions.

2. **Three candidates remain.** strip_all (the open-source default), raw_window3, and raw_retain are statistically indistinguishable at N=10. Phase 2 will resolve this by running these three conditions on the full 114 telecom base task set.

3. **raw_window3 is the front-runner on observational evidence.** Highest pass rate, zero capability failures, 100% partial action reward, best hard-task performance. But this could be noise at N=10. Phase 2 will confirm or refute.

### Phase 2 design requirements

- **3 conditions**: strip_all, raw_window3, raw_retain
- **114 tasks**: full telecom base set (eliminates task selection bias)
- **1 trial minimum**: 114 × 3 × 3 models = 1,026 runs
- **Statistical power**: 114 paired observations per model provides ~80% power to detect a 10pp difference (87% vs 97%) via McNemar's test
- **Thinking analysis fix**: The agent accumulator (commit `945f867`) now captures raw thinking metrics. Phase 2 data will include per-message thinking token counts.
