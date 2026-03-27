# Phase 1 Results

## Experiment Configuration

- **Benchmark**: τ²-bench telecom domain, 10 selected tasks (3 easy, 4 medium, 3 hard)
- **Models**: Qwen3.5-2B Q8_0 (Mac M4 Pro), Qwen3.5-9B Q8_0 (RunPod L40S 48GB)
- **Inference**: llama.cpp with custom Jinja chat template (disables Qwen3.5's built-in thinking stripping)
- **User simulator**: MiMo-V2 Flash on OpenRouter ($0.10/$0.30 per M tokens)
- **Summarizer**: MiMo-V2 Flash on OpenRouter (same model, used for summary conditions)
- **Trials**: 1 per task per condition
- **Total runs**: 120 (2 models × 6 conditions × 10 tasks)
- **Total OpenRouter cost**: ~$0.45
- **4B not yet run** (scheduled for Mac after 2B)

### Conditions

| # | Condition | enable_thinking | History contains | Summarizer |
|---|-----------|-----------------|------------------|------------|
| 1 | thinking_off | false | nothing | no |
| 2 | strip_all | true | nothing (thinking used for current turn only) | no |
| 3 | raw_window3 | true | raw `<think>` blocks from last 3 user-turn windows | no |
| 4 | raw_retain | true | raw `<think>` blocks from all turns | no |
| 5 | summary_window3 | true | `<think_summary>` from last 3 windows | yes |
| 6 | summary_retain | true | `<think_summary>` from all turns | yes |

### Tasks

| ID | Short description | Subtask count | Persona |
|----|-------------------|---------------|---------|
| T1 | airplane_mode, bad_network_pref, data_exceeded | 3 | None |
| T2 | airplane_mode, bad_wifi_calling, roaming_off | 3 | Easy |
| T3 | airplane_mode, broken_app_permissions | 3 | Hard |
| T4 | airplane_mode, bad_network_pref, bad_wifi_calling, data_exceeded, sim_unseat, roaming_on | 6 | Easy |
| T5 | airplane_mode, broken_app_permissions, data_exceeded, roaming_off | 4 | None |
| T6 | bad_network_pref, bad_wifi_calling, broken_sms_permission, data_off, sim_unseat, roaming_off | 6 | Hard |
| T7 | bad_wifi_calling, broken_apn, data_off, data_exceeded, sim_unseat | 5 | None |
| T8 | airplane_mode, bad_network_pref, bad_wifi_calling, broken_apn, broken_app_permissions, sim_unseat, roaming_off | 7 | Hard |
| T9 | airplane_mode, bad_network_pref, bad_wifi_calling, broken_apn, broken_storage_permission, data_off, data_exceeded, sim_unseat | 8 | Easy |
| T10 | airplane_mode, bad_network_pref, bad_wifi_calling, broken_sms_permission, data_off, data_exceeded, sim_unseat, roaming_off | 8 | None |

---

## Aggregate Results

| Condition | 2B | 9B |
|-----------|-----|-----|
| thinking_off | 9/10 (90%) | 9/10 (90%) |
| strip_all | 8/10 (80%) | 8/10 (80%) |
| raw_window3 | 9/10 (90%) | **10/10 (100%)** |
| raw_retain | 7/10 (70%) | 9/10 (90%) |
| summary_window3 | 9/10 (90%) | 5/10 (50%) |
| summary_retain | 4/10 (40%) | 5/10 (50%) |

---

## Per-Task Matrix

### 2B (Qwen3.5-2B Q8_0, Mac M4 Pro)

| Task | Subtasks | thinking_off | strip_all | raw_w3 | raw_ret | sum_w3 | sum_ret |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | 3 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| T2 | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T3 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T4 | 6 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| T5 | 4 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| T6 | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| T7 | 5 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| T8 | 7 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T9 | 8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T10 | 8 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Total** | | **9** | **8** | **9** | **7** | **9** | **4** |

### 9B (Qwen3.5-9B Q8_0, RunPod L40S)

| Task | Subtasks | thinking_off | strip_all | raw_w3 | raw_ret | sum_w3 | sum_ret |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | 3 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| T2 | 3 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| T3 | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T4 | 6 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| T5 | 4 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| T6 | 6 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| T7 | 5 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| T8 | 7 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| T9 | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| T10 | 8 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Total** | | **9** | **8** | **10** | **9** | **5** | **5** |

---

## Conversation-Level Observations

### Termination Reasons by Condition (9B)

| Condition | Passes | transfer_to_human_agents | too_many_errors | max_steps | user_stop (failed) |
|-----------|--------|--------------------------|-----------------|-----------|---------------------|
| thinking_off | 9 | 0 | 0 | 0 | 1 |
| strip_all | 8 | 2 | 0 | 0 | 0 |
| raw_window3 | 10 | 0 | 0 | 0 | 0 |
| raw_retain | 9 | 1 | 0 | 0 | 0 |
| summary_window3 | 5 | 2 | 3 | 0 | 0 |
| summary_retain | 5 | 5 | 0 | 0 | 0 |

### Transfer Details (9B)

Every 9B `summary_retain` failure ended with `transfer_to_human_agents`:

| Task | Msgs at transfer | What agent had done before transfer | What remained |
|------|------------------|-------------------------------------|---------------|
| T6 | 26 | SIM reseat, network mode, data on, roaming on | wifi calling, sms permission, data refuel |
| T7 | 50 | SIM reseat, data on, data refuel 2GB (transferred, caught self, refueled, transferred again) | wifi calling, APN |
| T8 | 12 | Nothing — never looked up customer account | everything |
| T9 | 62 | enable roaming, refuel 2GB, airplane off, SIM reseat, network mode | wifi calling, sms permission |
| T10 | 24 | Account lookup, identified data overage + MMSC not set | data refuel, APN reset, all device checks |

### Data Refuel Amounts

When the agent performed data refueling, the amount chosen:

| Conversation | Amount | Task Result |
|--------------|--------|-------------|
| 9B thinking_off T1 | 1.0 GB | ❌ (MMS worked, reward 0.0) |
| 9B raw_window3 T1 | 2.0 GB | ✅ |
| 9B summary_window3 T1 | 2.0 GB | ✅ |
| 9B summary_retain T4 | 2.0 GB | ✅ |
| 9B summary_retain T7 | 2.0 GB | ❌ (remaining issues unfixed) |
| 9B summary_retain T9 | 2.0 GB | ❌ (transferred before fixing remaining) |
| 2B thinking_off T7 | 1.0 GB | ❌ (MMS worked, reward 0.0) |
| 2B raw_window3 T7 | 2.0 GB | ✅ |

In the two cases where 1.0 GB was refueled (both `thinking_off`), MMS functionality was restored and the customer confirmed success, but the task reward was 0.0 — the evaluation's `assert_data_refueling_amount` check requires exactly 2.0 GB.

### User Simulator Behavior Patterns

**Agent-driven (typical in passing runs)**: Agent performs account lookup, identifies root causes, gives step-by-step instructions ("please run check_status_bar"), customer follows instructions and reports results.

**User sim autonomous (observed in some summary failures)**: User sim makes tool calls independently without agent instruction — checks network status, toggles airplane mode, changes network preferences, reseats SIM. In 9B summary_window3 T9, the user sim made approximately 40 consecutive tool calls autonomously.

**Environment corruption (9B summary_window3 T9)**: User sim called `set_apn_settings` with a JSON string instead of a dict object. The error `'str' object has no attribute 'apn_name'` persisted across subsequent tool calls (`toggle_roaming`, `toggle_airplane_mode`), rendering the environment inoperable. The user sim accidentally re-enabled airplane mode during error recovery attempts.

### Cross-Condition Comparisons on Same Tasks

**T1 at 9B** (3 issues):
- thinking_off: Agent troubleshoots correctly, refuels 1GB (user chose amount). MMS works. Reward 0.0.
- strip_all: Agent checks APN/wifi/permissions but misses airplane mode. User sim discovers airplane mode at msg 29 and fixes it independently. Agent never refuels data. Transfers at msg 52. Reward 0.0.
- raw_window3: Agent identifies data overage early, refuels 2GB, guides device fixes. Reward 1.0.

**T7 at 2B** (5 issues):
- thinking_off: 116 messages. Agent refuels 1GB. MMS works. Reward 0.0.
- strip_all: 85 messages. Troubleshooting loop. Reward 0.0.
- raw_window3: 98 messages. Agent refuels 2GB, resolves all 5 issues. Reward 1.0.
- raw_retain: 98 messages. Similar to raw_window3. Reward 1.0.
- summary_window3: 74 messages. Reward 1.0.
- summary_retain: 77 messages. Reward 0.0.

**T6 at 9B** (6 issues, Hard persona):
- thinking_off: 102 msgs. Reward 1.0.
- strip_all: 64 msgs. Reward 1.0.
- raw_window3: 82 msgs. Reward 1.0.
- raw_retain: 60 msgs. Agent exhaustively troubleshoots with elderly confused user. Mobile data never restored. Transfers. Reward 0.0.
- summary_window3: 36 msgs. Agent partially fixes (SIM, network, data toggle, roaming). Transfers before testing MMS or checking wifi calling/permissions. Reward 0.0.
- summary_retain: 26 msgs. Similar partial fix then transfer. Reward 0.0.

**T9 at 9B** (8 issues):
- raw_window3: 82 msgs. Agent resolves all 8 issues in order. Reward 1.0.
- summary_window3: 109 msgs. User sim takes over, corrupts environment. Reward 0.0.
- summary_retain: 62 msgs. Agent fixes 5 issues, transfers before checking wifi calling and sms permission. Transfer summary names those exact remaining items. Reward 0.0.

---

## Message Count Statistics

### Average message count per task (completed runs only)

| Condition | 2B avg msgs | 9B avg msgs |
|-----------|-------------|-------------|
| thinking_off | 98 | 80 |
| strip_all | 77 | 65 |
| raw_window3 | 83 | 86 |
| raw_retain | 108 | 80 |
| summary_window3 | 83 | 60 |
| summary_retain | 101 | 50 |

9B summary_retain has the lowest average message count (50), consistent with early termination via transfer.

---

## Data Quality Notes

1. **thinking_analysis.jsonl records all zeros** for `raw_thinking_tokens_approx` and `summary_tokens_approx` across all conditions. Token accounting data is not usable. The logging function reads from already-stripped messages. This is a known bug — the agent's internal thinking state is not captured in the output files.

2. **Duplicate 9B thinking_off directory**: `qwen35-9b_thinking_off_20260326T211419Z` (1 task, smoke test) and `qwen35-9b_thinking_off_20260326T211752Z` (10 tasks, actual run). Only the 10-task run is included in results.

3. **No agent-view data captured**: The two-view architecture (`_internal_messages` vs `state.messages`) works correctly at runtime, but `results.json` only contains the public view (stripped messages). Raw thinking content is not preserved in any output file.

4. **Single trial**: Each task × condition combination was run exactly once. No statistical significance can be claimed. Results are qualitative signal for Phase 2 design.

---

## Observations Without Interpretation

- raw_window3 is the only condition that achieved 10/10 at either model scale.
- All 5 summary_retain failures at 9B terminated via transfer_to_human_agents.
- 0 out of 10 raw_window3 runs at 9B terminated via transfer.
- The two thinking_off failures where reward=0.0 despite MMS working both involved 1.0 GB refuel instead of 2.0 GB.
- T3 (2 subtasks, simplest task) passed across all 12 condition × model combinations.
- T10 at 2B failed in 4/6 conditions. T10 at 9B failed in 2/6 conditions (both summary).
- 9B summary conditions show an inverted pattern vs 2B: at 2B, summary_window3 (9/10) outperforms summary_retain (4/10). At 9B, both summary conditions score 5/10.
- strip_all scored exactly 8/10 at both model scales.
- thinking_off scored exactly 9/10 at both model scales.
