#!/usr/bin/env python3
"""View Phase 1 results as a clean table. Run anytime, even mid-experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "phase1"

MODEL_ORDER = ["qwen35-0.8b", "qwen35-4b", "qwen35-9b"]
CONDITION_ORDER = ["thinking_off", "strip_all", "window_3", "retain_all"]


def load_run_data(run_dir: Path) -> dict | None:
    """Load summary + detailed simulation data from a run directory."""
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        summary["_dir"] = run_dir.name

        if results_path.exists():
            results = json.loads(results_path.read_text())
            sims = results.get("simulations", [])
            tasks = []
            total_duration = 0.0
            total_messages = 0
            total_partial_reward = 0.0
            total_partial_possible = 0
            term_counts: dict[str, int] = {}

            for sim in sims:
                reward_info = sim.get("reward_info", {}) or {}
                reward = (
                    reward_info.get("reward", 0.0)
                    if isinstance(reward_info, dict)
                    else 0.0
                )
                task_id = sim.get("task_id", "?")
                short_id = (
                    task_id.split("]")[-1][:50] if "]" in task_id else task_id[:50]
                )
                term = sim.get("termination_reason", "?")
                if "." in term:
                    term = term.split(".")[-1]
                duration = sim.get("duration", 0)
                msgs = len(sim.get("messages", []))

                # Extract partial action rewards
                action_reward = 0
                action_total = 0
                if isinstance(reward_info, dict):
                    for key, val in reward_info.items():
                        if isinstance(val, dict) and "checks" in val:
                            for check in val["checks"]:
                                if isinstance(check, dict) and "reward" in check:
                                    action_total += 1
                                    action_reward += check["reward"]

                tasks.append(
                    {
                        "task": short_id,
                        "reward": reward,
                        "termination": term,
                        "duration": duration,
                        "messages": msgs,
                        "action_reward": action_reward,
                        "action_total": action_total,
                    }
                )
                total_duration += duration
                total_messages += msgs
                term_counts[term] = term_counts.get(term, 0) + 1

            summary["_tasks"] = tasks
            summary["_avg_duration"] = total_duration / len(sims) if sims else 0
            summary["_avg_messages"] = total_messages / len(sims) if sims else 0
            summary["_total_duration"] = total_duration
            summary["_term_counts"] = term_counts

        return summary
    except (json.JSONDecodeError, OSError):
        return None


def print_summary_table(runs: list[dict]) -> None:
    grid: dict[tuple[str, str], dict] = {}
    for r in runs:
        key = (r.get("model", "?"), r.get("condition", "?"))
        grid[key] = r

    hdr = f"{'Model':<20} {'Condition':<14} {'Pass':<6} {'N':<4} {'Rate':<7} {'Infra':<7} {'Avg Dur':<9} {'Avg Msgs':<10} {'Termination Reasons'}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            key = (model, condition)
            if key in grid:
                r = grid[key]
                passed = r.get("full_reward_count", 0)
                total = r.get("num_simulations", 0)
                terms = r.get("_term_counts", {})
                infra = terms.get("infrastructure_error", 0)
                clean = total - infra
                rate = f"{passed / clean * 100:.0f}%" if clean > 0 else "-"
                infra_str = f"({infra})" if infra > 0 else ""
                avg_dur = f"{r.get('_avg_duration', 0):.0f}s"
                avg_msgs = f"{r.get('_avg_messages', 0):.0f}"
                term_str = ", ".join(f"{k}:{v}" for k, v in sorted(terms.items()))
                print(
                    f"{model:<20} {condition:<14} {passed:<6} {clean:<4} {rate:<7} {infra_str:<7} {avg_dur:<9} {avg_msgs:<10} {term_str}"
                )
            else:
                print(
                    f"{model:<20} {condition:<14} {'·':<6} {'·':<4} {'·':<7} {'·':<7} {'·':<9} {'·':<10}"
                )
        print()

    print("=" * len(hdr))


def print_detailed_view(runs: list[dict]) -> None:
    for r in runs:
        tasks = r.get("_tasks", [])
        if not tasks:
            continue

        model = r.get("model", "?")
        condition = r.get("condition", "?")
        passed = r.get("full_reward_count", 0)
        total = r.get("num_simulations", 0)

        print(f"\n{'━' * 90}")
        print(
            f"  {model} / {condition}  —  {passed}/{total} passed  —  total {r.get('_total_duration', 0):.0f}s"
        )
        print(f"{'━' * 90}")
        print(
            f"  {'Task':<52} {'Rew':<6} {'Actions':<10} {'Term':<18} {'Msgs':<6} {'Time'}"
        )
        print(f"  {'─' * 86}")

        for t in tasks:
            reward_icon = "✓" if t["reward"] >= 1.0 else "✗"
            reward_str = f"{reward_icon} {t['reward']:.1f}"
            if t["action_total"] > 0:
                action_str = f"{t['action_reward']:.0f}/{t['action_total']}"
            else:
                action_str = "-"
            duration = f"{t['duration']:.0f}s"
            print(
                f"  {t['task']:<52} {reward_str:<6} {action_str:<10} {t['termination']:<18} {t['messages']:<6} {duration}"
            )
        print()


def main() -> None:
    if not RESULTS_ROOT.exists():
        print("No results found yet.")
        sys.exit(0)

    runs = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("summary"):
            continue
        data = load_run_data(d)
        if data:
            runs.append(data)

    if not runs:
        print("No completed runs found. Checking for in-progress...")
        for d in sorted(RESULTS_ROOT.iterdir()):
            if d.is_dir() and (d / "results.json").exists():
                print(f"  {d.name}: results.json exists (no summary yet)")
        sys.exit(0)

    print(f"\n  Phase 1 Results  —  {len(runs)}/16 configurations completed\n")
    print_summary_table(runs)

    if "--detail" in sys.argv or "-d" in sys.argv:
        print_detailed_view(runs)
    else:
        print("  Tip: run with --detail or -d for per-task breakdown\n")

    # In-progress
    all_dirs = [
        d
        for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("summary")
    ]
    in_progress = len(all_dirs) - len(runs)
    if in_progress > 0:
        print(f"  {in_progress} run(s) in progress")


if __name__ == "__main__":
    main()
