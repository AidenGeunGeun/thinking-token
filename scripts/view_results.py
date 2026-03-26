#!/usr/bin/env python3
"""View Phase 1 results as a clean table. Run anytime, even mid-experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "phase1"

MODEL_ORDER = ["qwen35-0.8b", "qwen35-4b", "qwen35-9b", "qwen35-35b-a3b"]
CONDITION_ORDER = ["thinking_off", "strip_all", "window_3", "retain_all"]


def load_summaries() -> list[dict]:
    summaries = []
    for summary_path in sorted(RESULTS_ROOT.glob("*/summary.json")):
        try:
            data = json.loads(summary_path.read_text())
            data["_dir"] = summary_path.parent.name
            summaries.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def load_detailed_results(run_dir: Path) -> dict:
    """Load per-task reward details from results.json."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text())
        sims = data.get("simulations", [])
        task_results = []
        for sim in sims:
            reward_info = sim.get("reward_info", {}) or {}
            reward = (
                reward_info.get("reward", 0.0) if isinstance(reward_info, dict) else 0.0
            )
            task_id = sim.get("task_id", "?")
            # Shorten task ID for display
            short_id = task_id.split("]")[-1][:40] if "]" in task_id else task_id[:40]
            term = sim.get("termination_reason", "?")
            if "." in term:
                term = term.split(".")[-1]
            task_results.append(
                {
                    "task": short_id,
                    "reward": reward,
                    "termination": term,
                    "duration": sim.get("duration", 0),
                    "messages": len(sim.get("messages", [])),
                }
            )
        return {"tasks": task_results}
    except (json.JSONDecodeError, OSError):
        return {}


def print_summary_table(summaries: list[dict]) -> None:
    print("=" * 72)
    print(f"{'Model':<20} {'Condition':<15} {'Passed':<8} {'Total':<8} {'Rate':<8}")
    print("-" * 72)

    grid: dict[tuple[str, str], dict] = {}
    for s in summaries:
        key = (s.get("model", "?"), s.get("condition", "?"))
        grid[key] = s

    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            key = (model, condition)
            if key in grid:
                s = grid[key]
                passed = s.get("full_reward_count", 0)
                total = s.get("num_simulations", 0)
                rate = f"{passed / total * 100:.0f}%" if total > 0 else "-"
                print(f"{model:<20} {condition:<15} {passed:<8} {total:<8} {rate:<8}")
            else:
                print(f"{model:<20} {condition:<15} {'...':<8} {'...':<8} {'...':<8}")
        print()

    print("=" * 72)


def print_detailed_view(summaries: list[dict]) -> None:
    for s in summaries:
        run_dir = RESULTS_ROOT / s["_dir"]
        details = load_detailed_results(run_dir)
        if not details.get("tasks"):
            continue

        model = s.get("model", "?")
        condition = s.get("condition", "?")
        print(f"\n{'─' * 72}")
        print(f"  {model} / {condition}")
        print(f"{'─' * 72}")
        print(f"  {'Task':<42} {'Reward':<8} {'Term':<15} {'Msgs':<6} {'Time':<6}")
        print(f"  {'─' * 68}")

        for t in details["tasks"]:
            reward_str = f"{'✓' if t['reward'] >= 1.0 else '✗'} {t['reward']:.2f}"
            duration = f"{t['duration']:.0f}s"
            print(
                f"  {t['task']:<42} {reward_str:<8} {t['termination']:<15} {t['messages']:<6} {duration:<6}"
            )


def main() -> None:
    if not RESULTS_ROOT.exists():
        print("No results found yet.")
        sys.exit(0)

    summaries = load_summaries()
    if not summaries:
        # Fall back to scanning results.json files without summary.json
        print("No summary files found. Checking for raw results...")
        for d in sorted(RESULTS_ROOT.iterdir()):
            if d.is_dir() and (d / "results.json").exists():
                print(f"  {d.name}: results.json exists (no summary yet)")
        sys.exit(0)

    print(f"\nPhase 1 Results ({len(summaries)} runs completed)\n")
    print_summary_table(summaries)

    if "--detail" in sys.argv or "-d" in sys.argv:
        print_detailed_view(summaries)

    # In-progress indicator
    all_dirs = [
        d
        for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("summary")
    ]
    completed = len(summaries)
    total_dirs = len(all_dirs)
    if total_dirs > completed:
        print(
            f"\n  ({total_dirs - completed} run(s) still in progress or missing summary)"
        )

    running = list(RESULTS_ROOT.glob("*_llama.log"))
    if running:
        print(f"  Server logs: {', '.join(r.name for r in running)}")


if __name__ == "__main__":
    main()
