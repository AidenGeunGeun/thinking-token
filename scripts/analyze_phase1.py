#!/usr/bin/env python3
"""Comprehensive Phase 1 results analysis.

Extracts every available metric from results.json files across all models
and conditions. Produces tables suitable for academic write-up.

Usage:
    python scripts/analyze_phase1.py
    python scripts/analyze_phase1.py --section tokens   # run specific section
    python scripts/analyze_phase1.py --csv               # emit CSV tables
"""

import argparse
import json
import glob
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class SimMetrics:
    """Metrics extracted from a single simulation."""

    model: str
    condition: str
    task_id: str
    task_short: str
    n_required_actions: int
    reward: float
    passed: bool
    partial_action_reward: float  # fraction of actions completed
    n_actions_passed: int
    n_actions_total: int
    env_assertions_met: bool
    duration: float
    termination_reason: str
    # Message counts
    n_messages: int
    n_assistant_msgs: int
    n_user_msgs: int
    n_tool_msgs: int
    n_tool_calls: int
    # Token usage (from per-message usage fields)
    agent_prompt_tokens: int
    agent_completion_tokens: int
    user_prompt_tokens: int
    user_completion_tokens: int
    # Timing
    agent_generation_time: float  # sum of generation_time_seconds
    # Cost
    agent_cost: float
    user_cost: float
    # Persona
    persona: str


@dataclass
class ConditionAgg:
    """Aggregated metrics for a model × condition."""

    model: str
    condition: str
    sims: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.sims)

    @property
    def n_passed(self):
        return sum(1 for s in self.sims if s.passed)

    @property
    def pass_rate(self):
        return self.n_passed / self.n if self.n else 0

    @property
    def avg_partial(self):
        return sum(s.partial_action_reward for s in self.sims) / self.n if self.n else 0

    @property
    def avg_duration(self):
        return sum(s.duration for s in self.sims) / self.n if self.n else 0

    @property
    def total_duration(self):
        return sum(s.duration for s in self.sims)

    @property
    def avg_messages(self):
        return sum(s.n_messages for s in self.sims) / self.n if self.n else 0

    @property
    def avg_agent_prompt(self):
        return sum(s.agent_prompt_tokens for s in self.sims) / self.n if self.n else 0

    @property
    def avg_agent_completion(self):
        return (
            sum(s.agent_completion_tokens for s in self.sims) / self.n if self.n else 0
        )

    @property
    def avg_user_prompt(self):
        return sum(s.user_prompt_tokens for s in self.sims) / self.n if self.n else 0

    @property
    def avg_user_completion(self):
        return (
            sum(s.user_completion_tokens for s in self.sims) / self.n if self.n else 0
        )

    @property
    def avg_agent_gen_time(self):
        return sum(s.agent_generation_time for s in self.sims) / self.n if self.n else 0

    @property
    def total_user_cost(self):
        return sum(s.user_cost for s in self.sims)

    @property
    def avg_tool_calls(self):
        return sum(s.n_tool_calls for s in self.sims) / self.n if self.n else 0


# ── Loading ──────────────────────────────────────────────────────────────────

MODELS = [
    ("2B", "results/phase1", "qwen35-2b"),
    ("4B", "results/phase1", "qwen35-4b"),
    ("9B", "results/phase1_9b/phase1", "qwen35-9b"),
]

CONDITIONS = [
    "thinking_off",
    "strip_all",
    "raw_window3",
    "raw_retain",
    "summary_window3",
    "summary_retain",
]

CONDITION_SHORT = {
    "thinking_off": "think_off",
    "strip_all": "strip",
    "raw_window3": "raw_w3",
    "raw_retain": "raw_all",
    "summary_window3": "sum_w3",
    "summary_retain": "sum_all",
}


def task_short_name(task_id: str) -> str:
    """Convert full task ID to short display name."""
    # Extract issues between ] and [PERSONA
    body = task_id.split("]", 1)[1] if "]" in task_id else task_id
    if "[PERSONA:" in body:
        issues_str, persona_str = body.rsplit("[PERSONA:", 1)
        persona = persona_str.rstrip("]")[0]  # H, E, N
    else:
        issues_str = body
        persona = "?"
    issues = [i for i in issues_str.split("|") if i]
    return f"T{len(issues)}i_{persona}"


def task_issue_count(task_id: str) -> int:
    body = task_id.split("]", 1)[1] if "]" in task_id else task_id
    if "[PERSONA:" in body:
        issues_str = body.rsplit("[PERSONA:", 1)[0]
    else:
        issues_str = body
    return len([i for i in issues_str.split("|") if i])


def task_persona(task_id: str) -> str:
    if "[PERSONA:" in task_id:
        return task_id.rsplit("[PERSONA:", 1)[1].rstrip("]")
    return "None"


def extract_sim_metrics(model: str, condition: str, sim: dict) -> SimMetrics:
    """Extract all metrics from a single simulation dict."""
    task_id = sim["task_id"]
    reward_info = sim.get("reward_info", {})

    # Action checks
    action_checks = reward_info.get("action_checks") or []
    n_actions = len(action_checks)
    n_passed = sum(1 for ac in action_checks if ac.get("action_match", False))
    partial = n_passed / n_actions if n_actions else 1.0

    # Env assertions
    env_assertions = reward_info.get("env_assertions") or []
    env_met = (
        all(ea.get("met", False) for ea in env_assertions) if env_assertions else True
    )

    # Message-level metrics
    messages = sim.get("messages", [])
    n_assistant = n_user = n_tool = n_tool_calls = 0
    agent_prompt = agent_completion = 0
    user_prompt = user_completion = 0
    agent_gen_time = 0.0

    for msg in messages:
        role = msg.get("role", "")
        usage = msg.get("usage") or {}
        prompt_tok = usage.get("prompt_tokens", 0) or 0
        comp_tok = usage.get("completion_tokens", 0) or 0
        gen_time = msg.get("generation_time_seconds") or 0.0

        if role == "assistant":
            n_assistant += 1
            agent_prompt += prompt_tok
            agent_completion += comp_tok
            agent_gen_time += gen_time
            tc = msg.get("tool_calls")
            if tc:
                n_tool_calls += len(tc)
        elif role == "user":
            n_user += 1
            user_prompt += prompt_tok
            user_completion += comp_tok
        elif role == "tool":
            n_tool += 1

    persona = task_persona(task_id)

    return SimMetrics(
        model=model,
        condition=condition,
        task_id=task_id,
        task_short=task_short_name(task_id),
        n_required_actions=n_actions,
        reward=reward_info.get("reward", 0),
        passed=reward_info.get("reward", 0) == 1.0,
        partial_action_reward=partial,
        n_actions_passed=n_passed,
        n_actions_total=n_actions,
        env_assertions_met=env_met,
        duration=sim.get("duration", 0),
        termination_reason=sim.get("termination_reason", "unknown"),
        n_messages=len(messages),
        n_assistant_msgs=n_assistant,
        n_user_msgs=n_user,
        n_tool_msgs=n_tool,
        n_tool_calls=n_tool_calls,
        agent_prompt_tokens=agent_prompt,
        agent_completion_tokens=agent_completion,
        user_prompt_tokens=user_prompt,
        user_completion_tokens=user_completion,
        agent_generation_time=agent_gen_time,
        agent_cost=sim.get("agent_cost", 0),
        user_cost=sim.get("user_cost", 0) or 0,
        persona=persona,
    )


def load_all_data() -> tuple[list[SimMetrics], dict[tuple, ConditionAgg]]:
    """Load all results and return flat list + aggregated dict."""
    all_sims = []
    aggs = {}  # (model, condition) -> ConditionAgg

    for model_label, base_path, prefix in MODELS:
        for condition in CONDITIONS:
            pattern = f"{base_path}/{prefix}_{condition}_*/"
            dirs = sorted(glob.glob(pattern))
            # Find the one with 10 simulations
            for d in dirs:
                spath = os.path.join(d, "summary.json")
                rpath = os.path.join(d, "results.json")
                if not os.path.exists(spath) or not os.path.exists(rpath):
                    continue
                with open(spath) as f:
                    summary = json.load(f)
                if summary.get("num_simulations", 0) < 10:
                    continue
                with open(rpath) as f:
                    results = json.load(f)

                agg = ConditionAgg(model=model_label, condition=condition)
                for sim in results.get("simulations", []):
                    m = extract_sim_metrics(model_label, condition, sim)
                    all_sims.append(m)
                    agg.sims.append(m)
                aggs[(model_label, condition)] = agg
                break  # take first valid dir

    return all_sims, aggs


# ── Statistical helpers ──────────────────────────────────────────────────────


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI)."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar's test p-value.
    b = cases where A passes and B fails.
    c = cases where A fails and B passes.
    Under H0: b/(b+c) ~ Binomial(b+c, 0.5)
    """
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial two-sided
    k = min(b, c)
    # p-value = 2 * P(X <= k) for Binomial(n, 0.5)
    from math import comb

    p = 0.0
    for i in range(k + 1):
        p += comb(n, i) * 0.5**n
    return min(1.0, 2 * p)


# ── Analysis sections ────────────────────────────────────────────────────────


def section_aggregate(all_sims, aggs):
    """Section 1: Aggregate pass rates."""
    print("=" * 90)
    print("1. AGGREGATE PASS RATES")
    print("=" * 90)
    print()

    # Main table
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f"{'Pass':>6} {'Rate':>6} {'95% CI':>14}"
    header += f"{'Avg':>7}"
    print(header)
    print("-" * 90)

    for c in CONDITIONS:
        row = f"{c:<18}"
        rates = []
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                lo, hi = wilson_ci(agg.n_passed, agg.n)
                row += f"{agg.n_passed:>4}/10 {agg.pass_rate:>5.0%} [{lo:>4.0%},{hi:>4.0%}]"
                rates.append(agg.pass_rate)
            else:
                row += f"{'?':>6} {'?':>6} {'?':>14}"
        avg = sum(rates) / len(rates) if rates else 0
        row += f"{avg:>6.0%}"
        print(row)

    # Category averages
    print()
    print("Category averages:")
    cats = {
        "No reasoning in history": ["thinking_off", "strip_all"],
        "Raw thinking retained": ["raw_window3", "raw_retain"],
        "Summarized thinking": ["summary_window3", "summary_retain"],
    }
    for cat_name, conds in cats.items():
        rates = []
        for m in ["2B", "4B", "9B"]:
            for c in conds:
                agg = aggs.get((m, c))
                if agg:
                    rates.append(agg.pass_rate)
        print(f"  {cat_name}: {sum(rates) / len(rates):.0%} (n={len(rates)} cells)")
    print()


def section_partial_rewards(all_sims, aggs):
    """Section 2: Partial action rewards (more granular than pass/fail)."""
    print("=" * 90)
    print("2. PARTIAL ACTION REWARDS (avg fraction of required actions completed)")
    print("=" * 90)
    print()

    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f"{'Partial':>9} {'Pass':>6}"
    print(header)
    print("-" * 60)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f"{agg.avg_partial:>8.1%} {agg.pass_rate:>5.0%}"
            else:
                row += f"{'?':>9} {'?':>6}"
        print(row)

    print()
    print("Gap between partial action reward and pass rate shows")
    print("'almost passed' tasks — high partial but reward=0.")
    print()

    # Find biggest gaps
    print("Largest partial-vs-pass gaps (tasks almost passing):")
    for m in ["2B", "4B", "9B"]:
        for c in CONDITIONS:
            agg = aggs.get((m, c))
            if not agg:
                continue
            for sim in agg.sims:
                if not sim.passed and sim.partial_action_reward >= 0.7:
                    missed = sim.n_actions_total - sim.n_actions_passed
                    print(
                        f"  {m}/{c}: {sim.task_short} — {sim.n_actions_passed}/{sim.n_actions_total} actions ({sim.partial_action_reward:.0%}), missed {missed}"
                    )
    print()


def section_per_task_matrix(all_sims, aggs):
    """Section 3: Per-task pass/fail matrix."""
    print("=" * 90)
    print("3. PER-TASK PASS/FAIL MATRIX")
    print("=" * 90)

    # Get ordered task list
    task_ids = []
    seen = set()
    for s in all_sims:
        if s.task_id not in seen:
            task_ids.append(s.task_id)
            seen.add(s.task_id)
    task_ids.sort(key=lambda t: (task_issue_count(t), t))

    # Assign stable T1-T10 labels
    task_labels = {tid: f"T{i + 1}" for i, tid in enumerate(task_ids)}

    for model in ["2B", "4B", "9B"]:
        print(f"\n--- {model} ---")
        header = f"{'Task':<5} {'Issues':>6} {'Per':>4}"
        for c in CONDITIONS:
            header += f" {CONDITION_SHORT[c]:>7}"
        header += f" {'Sum':>5}"
        print(header)
        print("-" * 78)

        for tid in task_ids:
            label = task_labels[tid]
            n_issues = task_issue_count(tid)
            persona = task_persona(tid)[0]
            row = f"{label:<5} {n_issues:>4}   {persona:>3}"
            row_sum = 0
            for c in CONDITIONS:
                agg = aggs.get((model, c))
                if agg:
                    match = [s for s in agg.sims if s.task_id == tid]
                    if match:
                        v = 1 if match[0].passed else 0
                        row_sum += v
                        row += f" {'  pass' if v else '  FAIL':>7}"
                    else:
                        row += f" {'?':>7}"
                else:
                    row += f" {'?':>7}"
            row += f" {row_sum:>3}/6"
            print(row)

        # Column totals
        row = f"{'TOTAL':<5} {'':>6} {'':>4}"
        for c in CONDITIONS:
            agg = aggs.get((model, c))
            if agg:
                row += f" {agg.n_passed:>5}/10"
        print(row)

    # Task difficulty ranking
    print(f"\n--- Task Legend ---")
    for tid in task_ids:
        label = task_labels[tid]
        n_issues = task_issue_count(tid)
        persona = task_persona(tid)
        # Count passes across all conditions and models
        total_pass = sum(1 for s in all_sims if s.task_id == tid and s.passed)
        total_n = sum(1 for s in all_sims if s.task_id == tid)
        body = tid.split("]", 1)[1].rsplit("[PERSONA:", 1)[0] if "]" in tid else tid
        print(
            f"  {label}: {n_issues} issues, {persona} persona, {total_pass}/{total_n} pass — {body[:70]}"
        )
    print()


def section_task_difficulty(all_sims, aggs):
    """Section 4: Task difficulty × condition interaction."""
    print("=" * 90)
    print("4. TASK DIFFICULTY × CONDITION INTERACTION")
    print("=" * 90)
    print()

    # Group tasks by difficulty
    difficulty_bins = {
        "easy (2-3)": (2, 3),
        "medium (4-6)": (4, 6),
        "hard (7-9)": (7, 9),
    }

    header = f"{'Difficulty':<14}"
    for c in CONDITIONS:
        header += f" {CONDITION_SHORT[c]:>7}"
    print(header)
    print("-" * 68)

    for bin_name, (lo, hi) in difficulty_bins.items():
        row = f"{bin_name:<14}"
        for c in CONDITIONS:
            matching = [
                s
                for s in all_sims
                if s.condition == c and lo <= task_issue_count(s.task_id) <= hi
            ]
            if matching:
                rate = sum(1 for s in matching if s.passed) / len(matching)
                row += f" {rate:>5.0%}/{len(matching):>1}"
            else:
                row += f" {'?':>7}"
        print(row)

    print()
    print("(Format: pass_rate/N where N = number of task-model observations in bin)")
    print()

    # Persona interaction
    print("Persona × condition:")
    header = f"{'Persona':<10}"
    for c in CONDITIONS:
        header += f" {CONDITION_SHORT[c]:>7}"
    print(header)
    print("-" * 62)

    for persona in ["None", "Easy", "Hard"]:
        row = f"{persona:<10}"
        for c in CONDITIONS:
            matching = [
                s for s in all_sims if s.condition == c and s.persona == persona
            ]
            if matching:
                rate = sum(1 for s in matching if s.passed) / len(matching)
                row += f" {rate:>5.0%}/{len(matching):>1}"
            else:
                row += f" {'?':>7}"
        print(row)
    print()


def section_tokens(all_sims, aggs):
    """Section 5: Token usage analysis."""
    print("=" * 90)
    print("5. TOKEN USAGE (per-task averages from message-level usage fields)")
    print("=" * 90)
    print()

    print(
        "Agent tokens (llama-server, local inference — prompt = context sent, completion = generated):"
    )
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m + ' prompt':>10} {m + ' compl':>10}"
    print(header)
    print("-" * 82)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f" {agg.avg_agent_prompt:>9.0f} {agg.avg_agent_completion:>9.0f}"
            else:
                row += f" {'?':>10} {'?':>10}"
        print(row)

    print()
    print("User sim tokens (MiMo-V2 Flash via OpenRouter):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m + ' prompt':>10} {m + ' compl':>10}"
    print(header)
    print("-" * 82)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f" {agg.avg_user_prompt:>9.0f} {agg.avg_user_completion:>9.0f}"
            else:
                row += f" {'?':>10} {'?':>10}"
        print(row)

    print()

    # Total tokens per task
    print("Total tokens per task (agent + user, prompt + completion):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m:>10}"
    print(header)
    print("-" * 52)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                total = (
                    agg.avg_agent_prompt
                    + agg.avg_agent_completion
                    + agg.avg_user_prompt
                    + agg.avg_user_completion
                )
                row += f" {total:>9.0f}"
            else:
                row += f" {'?':>10}"
        print(row)

    print()

    # Thinking overhead: tokens for thinking-on vs thinking-off
    print("Thinking overhead (agent completion tokens, thinking_on vs off):")
    for m in ["2B", "4B", "9B"]:
        off = aggs.get((m, "thinking_off"))
        if not off:
            continue
        base = off.avg_agent_completion
        print(f"  {m} thinking_off baseline: {base:.0f} avg completion tokens/task")
        for c in CONDITIONS:
            if c == "thinking_off":
                continue
            agg = aggs.get((m, c))
            if agg:
                overhead = agg.avg_agent_completion - base
                ratio = agg.avg_agent_completion / base if base else 0
                print(
                    f"    {c}: {agg.avg_agent_completion:.0f} ({overhead:+.0f}, {ratio:.1f}x)"
                )
    print()


def section_duration_cost(all_sims, aggs):
    """Section 6: Duration and cost analysis."""
    print("=" * 90)
    print("6. DURATION AND COST ANALYSIS")
    print("=" * 90)
    print()

    # GPU hourly rates
    GPU_RATES = {"2B": 0.0, "4B": 0.0, "9B": 0.87}  # Mac is free, RunPod L40S
    GPU_LABELS = {
        "2B": "Mac M4 Pro (free)",
        "4B": "Mac M4 Pro (free)",
        "9B": "RunPod L40S ($0.87/hr)",
    }

    print("Duration per task (seconds):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m + ' avg':>8} {m + ' total':>8}"
    print(header)
    print("-" * 70)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f" {agg.avg_duration:>7.0f}s {agg.total_duration:>7.0f}s"
            else:
                row += f" {'?':>8} {'?':>8}"
        print(row)

    # Total compute time per model
    print()
    print("Total compute time per model (all 6 conditions):")
    for m in ["2B", "4B", "9B"]:
        total_s = sum(aggs[(m, c)].total_duration for c in CONDITIONS if (m, c) in aggs)
        total_h = total_s / 3600
        gpu_cost = total_h * GPU_RATES[m]
        print(
            f"  {m}: {total_s:.0f}s ({total_h:.2f}h) on {GPU_LABELS[m]} = ${gpu_cost:.2f} compute"
        )

    # Cost per successful task
    print()
    print("Cost efficiency (seconds per successful task):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m:>12}"
    print(header)
    print("-" * 55)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg and agg.n_passed > 0:
                spt = agg.total_duration / agg.n_passed
                row += f" {spt:>10.0f}s"
            elif agg:
                row += f" {'∞':>12}"
            else:
                row += f" {'?':>12}"
        print(row)

    # Agent generation time breakdown
    print()
    print("Agent generation time (model inference only, excl. API/overhead):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m + ' gen':>8} {m + ' pct':>6}"
    print(header)
    print("-" * 62)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg and agg.avg_duration > 0:
                pct = agg.avg_agent_gen_time / agg.avg_duration * 100
                row += f" {agg.avg_agent_gen_time:>7.0f}s {pct:>4.0f}%"
            else:
                row += f" {'?':>8} {'?':>6}"
        print(row)

    # User sim (OpenRouter) cost
    print()
    print("User sim API cost (OpenRouter MiMo-V2 Flash):")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m:>10}"
    print(header)
    print("-" * 52)

    total_api = 0
    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f" ${agg.total_user_cost:>8.4f}"
                total_api += agg.total_user_cost
            else:
                row += f" {'?':>10}"
        print(row)
    print(f"\nTotal API cost across all runs: ${total_api:.4f}")
    print()


def section_termination(all_sims, aggs):
    """Section 7: Termination reason analysis."""
    print("=" * 90)
    print("7. TERMINATION REASONS")
    print("=" * 90)
    print()

    reasons = sorted(set(s.termination_reason for s in all_sims))

    for model in ["2B", "4B", "9B"]:
        print(f"--- {model} ---")
        header = f"{'Condition':<18}"
        for r in reasons:
            short_r = r.replace("TerminationReason.", "").lower()[:12]
            header += f" {short_r:>12}"
        print(header)
        print("-" * (18 + 13 * len(reasons)))

        for c in CONDITIONS:
            agg = aggs.get((model, c))
            if not agg:
                continue
            row = f"{c:<18}"
            for r in reasons:
                count = sum(1 for s in agg.sims if s.termination_reason == r)
                row += f" {count:>12}" if count else f" {'.':>12}"
            print(row)
        print()


def section_pairwise(all_sims, aggs):
    """Section 8: Pairwise statistical comparisons."""
    print("=" * 90)
    print("8. PAIRWISE COMPARISONS (McNemar's exact test, paired by task)")
    print("=" * 90)
    print()
    print(
        "Key comparisons on same tasks. b = A passes & B fails, c = A fails & B passes."
    )
    print("p-value from exact McNemar's test. N=10 per model, N=30 pooled.")
    print()

    comparisons = [
        ("strip_all", "thinking_off", "Does thinking help? (strip_all vs no thinking)"),
        ("raw_window3", "strip_all", "Does raw retention help? (window_3 vs strip)"),
        (
            "raw_retain",
            "strip_all",
            "Does full raw retention help? (retain_all vs strip)",
        ),
        ("raw_window3", "raw_retain", "Window vs full retention?"),
        ("summary_retain", "raw_retain", "Summary vs raw? (retain_all)"),
        ("summary_window3", "raw_window3", "Summary vs raw? (window_3)"),
    ]

    # Get task lists
    task_ids = sorted(set(s.task_id for s in all_sims))

    for cond_a, cond_b, question in comparisons:
        print(f"  {question}")
        print(f"  {cond_a} vs {cond_b}")

        for scope, models in [
            ("per-model", [["2B"], ["4B"], ["9B"]]),
            ("pooled", [["2B", "4B", "9B"]]),
        ]:
            for model_set in models:
                label = model_set[0] if len(model_set) == 1 else "ALL"
                b = c = tied_pass = tied_fail = 0
                for m in model_set:
                    agg_a = aggs.get((m, cond_a))
                    agg_b = aggs.get((m, cond_b))
                    if not agg_a or not agg_b:
                        continue
                    for tid in task_ids:
                        sim_a = next((s for s in agg_a.sims if s.task_id == tid), None)
                        sim_b = next((s for s in agg_b.sims if s.task_id == tid), None)
                        if not sim_a or not sim_b:
                            continue
                        if sim_a.passed and not sim_b.passed:
                            b += 1
                        elif not sim_a.passed and sim_b.passed:
                            c += 1
                        elif sim_a.passed and sim_b.passed:
                            tied_pass += 1
                        else:
                            tied_fail += 1

                p = mcnemar_exact(b, c)
                n = b + c + tied_pass + tied_fail
                direction = (
                    f"{cond_a} better"
                    if b > c
                    else f"{cond_b} better"
                    if c > b
                    else "tied"
                )
                sig = (
                    "***"
                    if p < 0.001
                    else "**"
                    if p < 0.01
                    else "*"
                    if p < 0.05
                    else "†"
                    if p < 0.10
                    else "ns"
                )
                print(
                    f"    {label:>4}: b={b} c={c} tied={tied_pass}+/{tied_fail}- (n={n}) p={p:.3f} {sig} → {direction}"
                )

        print()


def section_model_scaling(all_sims, aggs):
    """Section 9: Model scaling analysis."""
    print("=" * 90)
    print("9. MODEL SCALING PATTERNS")
    print("=" * 90)
    print()

    print("Pass rate by model (averaged across conditions):")
    for m in ["2B", "4B", "9B"]:
        rates = [aggs[(m, c)].pass_rate for c in CONDITIONS if (m, c) in aggs]
        avg = sum(rates) / len(rates)
        best_c = max(
            CONDITIONS, key=lambda c: aggs[(m, c)].pass_rate if (m, c) in aggs else -1
        )
        worst_c = min(
            CONDITIONS, key=lambda c: aggs[(m, c)].pass_rate if (m, c) in aggs else 2
        )
        best_r = aggs[(m, best_c)].pass_rate
        worst_r = aggs[(m, worst_c)].pass_rate
        spread = best_r - worst_r
        print(
            f"  {m}: avg={avg:.0%}, best={best_c}({best_r:.0%}), worst={worst_c}({worst_r:.0%}), spread={spread:.0%}"
        )

    print()
    print("Condition sensitivity by model (spread = best - worst pass rate):")
    print("Higher spread = more sensitive to retention strategy choice.")
    print()

    # Which tasks are universally hard/easy?
    print(
        "Task consistency across models (pass count out of 18 = 3 models × 6 conditions):"
    )
    task_ids = sorted(
        set(s.task_id for s in all_sims), key=lambda t: (task_issue_count(t), t)
    )
    task_labels = {tid: f"T{i + 1}" for i, tid in enumerate(task_ids)}

    for tid in task_ids:
        total = sum(1 for s in all_sims if s.task_id == tid and s.passed)
        n = sum(1 for s in all_sims if s.task_id == tid)
        n_issues = task_issue_count(tid)
        label = task_labels[tid]
        bar = "█" * total + "░" * (n - total)
        print(f"  {label} ({n_issues}i): {total:>2}/{n} {bar}")
    print()


def section_message_patterns(all_sims, aggs):
    """Section 10: Message count and tool call patterns."""
    print("=" * 90)
    print("10. CONVERSATION STRUCTURE")
    print("=" * 90)
    print()

    print("Average messages per task:")
    header = f"{'Condition':<18}"
    for m in ["2B", "4B", "9B"]:
        header += f" {m + ' msgs':>8} {m + ' tools':>8}"
    print(header)
    print("-" * 66)

    for c in CONDITIONS:
        row = f"{c:<18}"
        for m in ["2B", "4B", "9B"]:
            agg = aggs.get((m, c))
            if agg:
                row += f" {agg.avg_messages:>7.0f} {agg.avg_tool_calls:>7.1f}"
            else:
                row += f" {'?':>8} {'?':>8}"
        print(row)

    # Tool calls for passed vs failed
    print()
    print("Tool calls: passed vs failed tasks:")
    for c in CONDITIONS:
        passed_tc = [s.n_tool_calls for s in all_sims if s.condition == c and s.passed]
        failed_tc = [
            s.n_tool_calls for s in all_sims if s.condition == c and not s.passed
        ]
        avg_p = sum(passed_tc) / len(passed_tc) if passed_tc else 0
        avg_f = sum(failed_tc) / len(failed_tc) if failed_tc else 0
        print(
            f"  {c:<18} passed: {avg_p:>5.1f} avg tool calls (n={len(passed_tc)}), "
            f"failed: {avg_f:>5.1f} (n={len(failed_tc)})"
        )
    print()


def section_summary(all_sims, aggs):
    """Final summary of key findings."""
    print("=" * 90)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 90)
    print()

    # Best condition per model
    print("Best condition per model:")
    for m in ["2B", "4B", "9B"]:
        ranked = sorted(
            CONDITIONS, key=lambda c: -aggs[(m, c)].pass_rate if (m, c) in aggs else 0
        )
        best = ranked[0]
        rate = aggs[(m, best)].pass_rate
        print(f"  {m}: {best} ({rate:.0%})")

    # Overall best
    print()
    print("Overall condition ranking (avg across models):")
    ranked = sorted(
        CONDITIONS,
        key=lambda c: (
            -sum(aggs[(m, c)].pass_rate for m in ["2B", "4B", "9B"] if (m, c) in aggs)
        ),
    )
    for i, c in enumerate(ranked):
        rates = [aggs[(m, c)].pass_rate for m in ["2B", "4B", "9B"] if (m, c) in aggs]
        avg = sum(rates) / len(rates)
        per_model = " | ".join(
            f"{m}={aggs[(m, c)].pass_rate:.0%}"
            for m in ["2B", "4B", "9B"]
            if (m, c) in aggs
        )
        print(f"  {i + 1}. {c:<18} avg={avg:.0%}  ({per_model})")

    print()
    print("Category ranking:")
    cats = [
        ("Raw thinking retained", ["raw_window3", "raw_retain"]),
        ("No reasoning in history", ["thinking_off", "strip_all"]),
        ("Summarized thinking", ["summary_window3", "summary_retain"]),
    ]
    for cat_name, conds in cats:
        rates = []
        for m in ["2B", "4B", "9B"]:
            for c in conds:
                agg = aggs.get((m, c))
                if agg:
                    rates.append(agg.pass_rate)
        print(f"  {cat_name}: avg={sum(rates) / len(rates):.0%}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

SECTIONS = {
    "aggregate": section_aggregate,
    "partial": section_partial_rewards,
    "matrix": section_per_task_matrix,
    "difficulty": section_task_difficulty,
    "tokens": section_tokens,
    "cost": section_duration_cost,
    "termination": section_termination,
    "pairwise": section_pairwise,
    "scaling": section_model_scaling,
    "messages": section_message_patterns,
    "summary": section_summary,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 results analysis")
    parser.add_argument(
        "--section", choices=list(SECTIONS.keys()), help="Run specific section only"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Output CSV format (not implemented yet)"
    )
    args = parser.parse_args()

    print("Loading data...")
    all_sims, aggs = load_all_data()
    print(
        f"Loaded {len(all_sims)} simulations across {len(aggs)} model×condition cells.\n"
    )

    if args.section:
        SECTIONS[args.section](all_sims, aggs)
    else:
        for name, func in SECTIONS.items():
            func(all_sims, aggs)


if __name__ == "__main__":
    main()
