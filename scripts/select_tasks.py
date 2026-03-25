#!/usr/bin/env python3
"""Pick telecom tasks for Phase 1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="base", help="tau2 telecom split name")
    parser.add_argument(
        "--count", type=int, default=10, help="number of tasks to select"
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "configs" / "phase1_tasks.json"),
        help="where to save the selected task list",
    )
    return parser.parse_args()


def task_metadata(task) -> dict:
    criteria = task.evaluation_criteria
    actions = list(criteria.actions or []) if criteria else []
    agent_actions = sum(1 for action in actions if action.requestor == "assistant")
    user_actions = sum(1 for action in actions if action.requestor == "user")
    env_assertions = len(criteria.env_assertions or []) if criteria else 0
    nl_assertions = len(criteria.nl_assertions or []) if criteria else 0
    communicate = len(criteria.communicate_info or []) if criteria else 0
    estimated_subtasks = max(
        agent_actions + user_actions, env_assertions, communicate, 1
    )
    return {
        "id": task.id,
        "estimated_subtasks": estimated_subtasks,
        "agent_actions": agent_actions,
        "user_actions": user_actions,
        "env_assertions": env_assertions,
        "nl_assertions": nl_assertions,
        "communicate_items": communicate,
    }


def select_task_mix(task_summaries: list[dict], count: int) -> list[dict]:
    buckets = {
        "2-3": [
            task for task in task_summaries if 2 <= task["estimated_subtasks"] <= 3
        ],
        "5-6": [
            task for task in task_summaries if 5 <= task["estimated_subtasks"] <= 6
        ],
        "7-9": [
            task for task in task_summaries if 7 <= task["estimated_subtasks"] <= 9
        ],
    }
    targets = [("2-3", 3), ("5-6", 4), ("7-9", 3)]

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for bucket_name, target in targets:
        bucket = sorted(
            buckets[bucket_name],
            key=lambda item: (-item["estimated_subtasks"], item["id"]),
        )
        for item in bucket[:target]:
            if item["id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["id"])

    if len(selected) < count:
        remaining = sorted(
            (item for item in task_summaries if item["id"] not in selected_ids),
            key=lambda item: (-item["estimated_subtasks"], item["id"]),
        )
        selected.extend(remaining[: count - len(selected)])

    return selected[:count]


def main() -> None:
    args = parse_args()
    try:
        from tau2.runner.helpers import get_tasks  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "tau2-bench is required to select tasks. Install it first with `pip install tau2-bench`."
        ) from exc

    tasks = get_tasks("telecom", task_split_name=args.split)
    summaries = [task_metadata(task) for task in tasks]
    selected = select_task_mix(summaries, args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": "telecom",
        "task_split": args.split,
        "selected_at": datetime.now(UTC).isoformat(),
        "selection_method": "estimated_subtasks_proxy",
        "task_ids": [task["id"] for task in selected],
        "tasks": selected,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Selected {len(selected)} tasks from telecom/{args.split}:")
    for task in selected:
        print(
            "- "
            f"{task['id']} | subtasks~{task['estimated_subtasks']} | "
            f"agent_actions={task['agent_actions']} | user_actions={task['user_actions']} | "
            f"env_assertions={task['env_assertions']}"
        )
    print(f"Saved selection to {output_path}")


if __name__ == "__main__":
    main()
