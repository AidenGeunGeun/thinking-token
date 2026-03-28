#!/usr/bin/env python3
"""View benchmark results as a clean table. Run anytime, even mid-experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1 import (  # noqa: E402
    CONFIG_PATH,
    load_conditions,
    load_config,
    load_models,
    resolve_results_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="path to experiment config"
    )
    parser.add_argument(
        "--detail", "-d", action="store_true", help="show per-task breakdown"
    )
    return parser.parse_args(argv)


def load_run_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            return None
        summary["_dir"] = run_dir.name
        return summary
    except (json.JSONDecodeError, OSError):
        return None


def is_completed_summary(summary: dict[str, Any]) -> bool:
    return summary.get("status") in (None, "complete")


def load_run_data(
    run_dir: Path, summary: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Load summary + detailed simulation data from a completed run directory."""
    results_path = run_dir / "results.json"
    try:
        summary = summary or load_run_summary(run_dir)
        if summary is None or not is_completed_summary(summary):
            return None

        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            sims = results.get("simulations", [])
            tasks = []
            total_duration = 0.0
            total_messages = 0
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

                action_reward = 0
                action_total = 0
                if isinstance(reward_info, dict):
                    for val in reward_info.values():
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


def build_run_grid(runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grid: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        key = (run.get("model", "?"), run.get("condition", "?"))
        grid[key] = run
    return grid


def matches_config_run(
    run_dir: Path,
    summary: dict[str, Any] | None,
    allowed_pairs: set[tuple[str, str]],
) -> bool:
    if summary is not None:
        return (
            summary.get("model", "?"),
            summary.get("condition", "?"),
        ) in allowed_pairs

    return any(
        run_dir.name.startswith(f"{model}_{condition}_")
        for model, condition in allowed_pairs
    )


def print_summary_table(
    run_grid: dict[tuple[str, str], dict[str, Any]],
    model_order: list[str],
    condition_order: list[str],
) -> None:
    hdr = f"{'Model':<20} {'Condition':<14} {'Pass':<6} {'N':<4} {'Rate':<7} {'Infra':<7} {'Avg Dur':<9} {'Avg Msgs':<10} {'Termination Reasons'}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for model in model_order:
        for condition in condition_order:
            key = (model, condition)
            if key in run_grid:
                run = run_grid[key]
                passed = run.get("full_reward_count", 0)
                total = run.get("num_simulations", 0)
                terms = run.get("_term_counts", {})
                infra = terms.get("infrastructure_error", 0)
                clean = total - infra
                rate = f"{passed / clean * 100:.0f}%" if clean > 0 else "-"
                infra_str = f"({infra})" if infra > 0 else ""
                avg_dur = f"{run.get('_avg_duration', 0):.0f}s"
                avg_msgs = f"{run.get('_avg_messages', 0):.0f}"
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


def print_detailed_view(ordered_runs: list[dict[str, Any]]) -> None:
    for run in ordered_runs:
        tasks = run.get("_tasks", [])
        if not tasks:
            continue

        model = run.get("model", "?")
        condition = run.get("condition", "?")
        passed = run.get("full_reward_count", 0)
        total = run.get("num_simulations", 0)

        print(f"\n{'━' * 90}")
        print(
            f"  {model} / {condition}  —  {passed}/{total} passed  —  total {run.get('_total_duration', 0):.0f}s"
        )
        print(f"{'━' * 90}")
        print(
            f"  {'Task':<52} {'Rew':<6} {'Actions':<10} {'Term':<18} {'Msgs':<6} {'Time'}"
        )
        print(f"  {'─' * 86}")

        for task in tasks:
            reward_icon = "✓" if task["reward"] >= 1.0 else "✗"
            reward_str = f"{reward_icon} {task['reward']:.1f}"
            if task["action_total"] > 0:
                action_str = f"{task['action_reward']:.0f}/{task['action_total']}"
            else:
                action_str = "-"
            duration = f"{task['duration']:.0f}s"
            print(
                f"  {task['task']:<52} {reward_str:<6} {action_str:<10} {task['termination']:<18} {task['messages']:<6} {duration}"
            )
        print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    results_root = resolve_results_root(config)
    model_order = [model.short_name for model in load_models(config, [])]
    condition_order = [condition.name for condition in load_conditions(config, [])]
    allowed_pairs = {
        (model, condition) for model in model_order for condition in condition_order
    }
    total_configs = len(model_order) * len(condition_order)
    experiment_name = config.get("experiment", {}).get("name", results_root.name)

    if not results_root.exists():
        print(f"No results found yet in {results_root}.")
        sys.exit(0)

    run_dirs = [
        run_dir
        for run_dir in sorted(results_root.iterdir())
        if run_dir.is_dir() and not run_dir.name.startswith("summary")
    ]
    runs = []
    in_progress_dirs = []
    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        if not matches_config_run(run_dir, summary, allowed_pairs):
            continue
        data = load_run_data(run_dir, summary)
        if data:
            runs.append(data)
            continue
        if (summary and summary.get("status") == "running") or (
            summary is None and (run_dir / "results.json").exists()
        ):
            in_progress_dirs.append(run_dir)

    if not runs:
        print("No completed runs found. Checking for in-progress...")
        for run_dir in in_progress_dirs:
            print(f"  {run_dir.name}: in progress")
        sys.exit(0)

    run_grid = build_run_grid(runs)
    ordered_runs = [
        run_grid[(model, condition)]
        for model in model_order
        for condition in condition_order
        if (model, condition) in run_grid
    ]

    print(
        f"\n  Results: {experiment_name}  —  {len(run_grid)}/{total_configs} configurations completed\n"
    )
    print_summary_table(run_grid, model_order, condition_order)

    if args.detail:
        print_detailed_view(ordered_runs)
    else:
        print("  Tip: run with --detail or -d for per-task breakdown\n")

    if in_progress_dirs:
        print(f"  {len(in_progress_dirs)} run(s) in progress")


if __name__ == "__main__":
    main()
