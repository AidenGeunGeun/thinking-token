# Data Schema

This project does not define a custom trajectory format. tau2-bench remains the source of truth for run outputs.

## Files Saved Per Run

Each run directory contains:

- tau2-bench's standard trajectory output (`results.json` and related artifacts)
- `thinking_analysis.jsonl`, a lightweight per-turn analysis file derived from the trajectory

## Directory Layout

```text
results/
└── phase1/
    └── {model}_{condition}_{timestamp}/
        ├── results.json
        ├── thinking_analysis.jsonl
        └── summary.json
```

## `thinking_analysis.jsonl`

One JSON object per turn.

```json
{
  "task_id": "telecom_001",
  "trial": 0,
  "turn_index": 3,
  "retention_strategy": "window_3",
  "assistant_message_count": 2,
  "thinking_text_chars": 812,
  "thinking_tokens_approx": 203,
  "retained_at_end": true,
  "window_size": 3,
  "source": "tau2 trajectory"
}
```

### Field Notes

- `task_id`: tau2-bench task identifier
- `trial`: zero-based trial index
- `turn_index`: zero-based user turn index
- `retention_strategy`: `strip_all`, `window_3`, or `retain_all`
- `assistant_message_count`: number of assistant messages observed in the turn
- `thinking_text_chars`: total extracted `<think>` text length for the turn
- `thinking_tokens_approx`: rough token estimate using `len(text) // 4`
- `retained_at_end`: whether this turn's thinking is still retained in the terminal conversation state after the run completes
- `window_size`: integer for windowed strategies, otherwise `null`
- `source`: provenance marker for downstream analysis

`retained_at_end` reflects terminal state only. It does not mean the turn was unavailable to every later prompt; for windowed retention, turns can be retained for some intermediate prompts and later fall out of the window. Actual per-prompt retention is enforced at each LLM call by `src/thinking.py`'s `apply_retention_strategy()`.

The analysis file is intentionally narrow: it complements tau2-bench trajectories instead of replacing them.
