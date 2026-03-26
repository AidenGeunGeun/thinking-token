#!/usr/bin/env python3
"""Verify thinking-retention invariants on a single tau2 task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase1
from src.agent import _extract_reasoning
from src.thinking import THINK_SUMMARY_PATTERN

INVARIANT_ORDER = ["INV-1", "INV-2", "INV-3", "INV-4", "INV-5", "INV-6", "INV-7"]
RAW_CONDITIONS = {"raw_window3", "raw_retain"}
SUMMARY_CONDITIONS = {"summary_window3", "summary_retain"}
NON_RETAINING_CONDITIONS = {"thinking_off", "strip_all"}


@dataclass(frozen=True)
class InvariantResult:
    status: str
    evidence: str = ""


@dataclass(frozen=True)
class ConditionReport:
    name: str
    invariants: dict[str, InvariantResult]
    summary_text: str = ""
    task_id: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(run_phase1.CONFIG_PATH),
        help="path to phase1.yaml",
    )
    parser.add_argument(
        "--model",
        default="",
        help="model short_name to verify (default: first model in config)",
    )
    parser.add_argument(
        "--condition",
        default="",
        help="single condition name to verify",
    )
    return parser.parse_args(argv)


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "unknown"))
    return str(getattr(message, "role", "unknown"))


def _content_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _raw_data_of(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        raw_data = message.get("raw_data")
        return raw_data if isinstance(raw_data, dict) else None
    raw_data = getattr(message, "raw_data", None)
    return raw_data if isinstance(raw_data, dict) else None


def _truncate(text: str, limit: int = 200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _load_snapshot(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Debug snapshot at {path} did not contain a JSON list")
    normalized: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "role": str(item.get("role", "unknown")),
                "content": str(item.get("content", "") or ""),
            }
        )
    return normalized


def _load_snapshot_meta(path: Path) -> list[dict[str, Any]]:
    meta_path = Path(f"{path}.meta")
    if not meta_path.exists():
        return []
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    assistant_turns = payload.get("assistant_turns", [])
    if not isinstance(assistant_turns, list):
        return []
    return [item for item in assistant_turns if isinstance(item, dict)]


def _simulation_messages(simulation: Any) -> list[Any]:
    if hasattr(simulation, "get_messages"):
        return list(simulation.get_messages())
    return list(getattr(simulation, "messages", []) or [])


def _assistant_messages(messages: list[Any]) -> list[Any]:
    return [message for message in messages if _role_of(message) == "assistant"]


def _first_summary_text(internal_messages: list[dict[str, str]]) -> str:
    for message in _assistant_messages(internal_messages):
        match = THINK_SUMMARY_PATTERN.search(_content_of(message))
        if match:
            return match.group(1).strip()
    return ""


def _thinking_generated(
    public_assistant_messages: list[Any],
    internal_messages: list[dict[str, str]],
    snapshot_meta: list[dict[str, Any]],
) -> bool:
    for message in _assistant_messages(internal_messages):
        content = _content_of(message)
        if "<think>" in content or "<think_summary>" in content:
            return True

    for turn in snapshot_meta:
        if turn.get("has_think") or turn.get("has_think_summary"):
            return True
        raw_thinking_chars = turn.get("raw_thinking_chars", 0)
        if isinstance(raw_thinking_chars, int) and raw_thinking_chars > 0:
            return True

    for message in public_assistant_messages:
        reasoning = _extract_reasoning(_raw_data_of(message))
        if reasoning:
            return True

    return False


def _no_tag_in_messages(messages: list[Any], tag: str) -> InvariantResult:
    for message in messages:
        content = _content_of(message)
        if tag in content:
            return InvariantResult(
                "FAIL",
                f"Found {tag} in assistant message: {_truncate(content)}",
            )
    return InvariantResult("PASS")


def _summary_quality(summary_text: str) -> InvariantResult:
    if not summary_text:
        return InvariantResult("FAIL", "No <think_summary> content found")

    failures: list[str] = []
    if summary_text.startswith("Internal Note"):
        failures.append("starts with 'Internal Note'")
    if "##" in summary_text or "###" in summary_text:
        failures.append("contains markdown headers")
    for line in summary_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            failures.append("contains bullet points")
            break

    if failures:
        return InvariantResult("FAIL", "; ".join(failures))
    return InvariantResult("PASS")


def evaluate_invariants(
    condition: run_phase1.ConditionConfig,
    public_messages: list[Any],
    internal_messages: list[dict[str, str]],
    snapshot_meta: list[dict[str, Any]] | None = None,
) -> ConditionReport:
    public_assistant_messages = _assistant_messages(public_messages)
    internal_assistant_messages = _assistant_messages(internal_messages)
    snapshot_meta = snapshot_meta or []
    generated_thinking = _thinking_generated(
        public_assistant_messages, internal_messages, snapshot_meta
    )
    summary_text = _first_summary_text(internal_messages)

    invariants: dict[str, InvariantResult] = {
        "INV-1": _no_tag_in_messages(public_assistant_messages, "<think>"),
        "INV-2": _no_tag_in_messages(public_assistant_messages, "<think_summary>"),
    }

    if condition.name in RAW_CONDITIONS:
        if not public_assistant_messages:
            invariants["INV-3"] = InvariantResult("skip", "Task completed in 0 turns")
        elif any(
            "<think>" in _content_of(message) for message in internal_assistant_messages
        ):
            invariants["INV-3"] = InvariantResult("PASS")
        elif not generated_thinking:
            invariants["INV-3"] = InvariantResult(
                "skip", "Model generated no observable thinking content"
            )
        else:
            invariants["INV-3"] = InvariantResult(
                "FAIL",
                "No internal assistant message retained <think> content",
            )
    else:
        invariants["INV-3"] = InvariantResult("skip")

    if condition.name in SUMMARY_CONDITIONS:
        if not public_assistant_messages:
            invariants["INV-4"] = InvariantResult("skip", "Task completed in 0 turns")
        elif any(
            "<think_summary>" in _content_of(message)
            for message in internal_assistant_messages
        ):
            invariants["INV-4"] = InvariantResult("PASS")
        else:
            invariants["INV-4"] = InvariantResult(
                "FAIL",
                "No internal assistant message retained <think_summary> content",
            )
    else:
        invariants["INV-4"] = InvariantResult("skip")

    if condition.name in NON_RETAINING_CONDITIONS:
        leaked = next(
            (
                _content_of(message)
                for message in internal_assistant_messages
                if "<think>" in _content_of(message)
                or "<think_summary>" in _content_of(message)
            ),
            "",
        )
        if leaked:
            invariants["INV-5"] = InvariantResult(
                "FAIL",
                f"Found retained thinking in internal message: {_truncate(leaked)}",
            )
        else:
            invariants["INV-5"] = InvariantResult("PASS")
    else:
        invariants["INV-5"] = InvariantResult("skip")

    if condition.name in SUMMARY_CONDITIONS:
        invariants["INV-6"] = _summary_quality(summary_text)
    else:
        invariants["INV-6"] = InvariantResult("skip")

    if not condition.enable_thinking:
        invariants["INV-7"] = InvariantResult("skip")
    elif not public_assistant_messages:
        invariants["INV-7"] = InvariantResult("skip", "Task completed in 0 turns")
    elif generated_thinking:
        invariants["INV-7"] = InvariantResult("PASS")
    else:
        evidence = "No internal thinking tags or assistant raw_data reasoning found"
        invariants["INV-7"] = InvariantResult("FAIL", evidence)

    return ConditionReport(
        name=condition.name, invariants=invariants, summary_text=summary_text
    )


def _resolve_task_ids(config: dict[str, Any]) -> list[str]:
    task_ids = run_phase1.load_task_ids(config)
    if task_ids and not task_ids[0].startswith("<select "):
        return task_ids[:1]

    from tau2.runner.helpers import get_tasks  # type: ignore[import-not-found]

    tasks = get_tasks(
        config["experiment"]["domain"],
        task_split_name=config["experiment"]["task_split"],
    )
    if not tasks:
        raise RuntimeError("No tau2 tasks available for verification")

    task_id = getattr(tasks[0], "task_id", "") or getattr(tasks[0], "id", "")
    if not task_id:
        raise RuntimeError("Unable to determine a task id for verification")
    return [str(task_id)]


def _run_single_task(
    config: dict[str, Any],
    model: run_phase1.ModelConfig,
    condition: run_phase1.ConditionConfig,
    task_ids: list[str],
    run_dir: Path,
):
    import src.register  # noqa: F401
    from tau2.data_model.simulation import TextRunConfig  # type: ignore[import-not-found]
    from tau2.runner.batch import run_tasks  # type: ignore[import-not-found]
    from tau2.runner.helpers import get_tasks  # type: ignore[import-not-found]

    experiment = dict(config["experiment"])
    experiment["trials"] = 1
    tasks = get_tasks(
        experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
    )
    run_config = TextRunConfig(
        domain=experiment["domain"],
        task_set_name=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
        num_trials=1,
        agent="thinking_retention",
        llm_agent=run_phase1.agent_llm_name(model),
        llm_args_agent=run_phase1.agent_llm_args(
            condition,
            int(config["llama"]["port"]),
            model,
            config,
        ),
        user="user_simulator",
        llm_user=run_phase1.user_model_name(config),
        llm_args_user={},
        max_concurrency=1,
        verbose_logs=False,
    )
    results = run_tasks(
        run_config,
        tasks,
        save_path=run_dir / "results.json",
        save_dir=run_dir,
        console_display=False,
    )
    return results


def verify_condition(
    config: dict[str, Any],
    model: run_phase1.ModelConfig,
    condition: run_phase1.ConditionConfig,
    task_ids: list[str],
    llama_server: str,
) -> ConditionReport:
    llama_config = config["llama"]
    port = int(llama_config["port"])

    verify_dir = PROJECT_ROOT / "results" / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"verify_{condition.name}_", dir=verify_dir
    ) as tmpdir:
        run_dir = Path(tmpdir)
        log_path = verify_dir / f"llama_{condition.name}.log"
        snapshot_path = run_dir / "internal_snapshot.json"
        process = None

        run_phase1.configure_condition_environment(condition, config)
        os.environ["THINKING_DEBUG_SNAPSHOT_PATH"] = str(snapshot_path)

        try:
            model_path = run_phase1.resolve_model_path(model)
            command = run_phase1.build_llama_command(
                model_path, llama_config, llama_server
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                run_phase1.wait_for_server(process, port)
                results = _run_single_task(config, model, condition, task_ids, run_dir)
        finally:
            run_phase1.stop_process(process)
            os.environ.pop("THINKING_DEBUG_SNAPSHOT_PATH", None)

        if not snapshot_path.exists():
            raise RuntimeError(
                f"Debug snapshot was not written for condition {condition.name}"
            )

        if len(results.simulations) != 1:
            raise RuntimeError(
                f"Expected exactly 1 simulation for {condition.name}, got {len(results.simulations)}"
            )

        simulation = results.simulations[0]
        report = evaluate_invariants(
            condition,
            _simulation_messages(simulation),
            _load_snapshot(snapshot_path),
            _load_snapshot_meta(snapshot_path),
        )
        return ConditionReport(
            name=report.name,
            invariants=report.invariants,
            summary_text=report.summary_text,
            task_id=str(getattr(simulation, "task_id", task_ids[0])),
        )


def print_condition_report(report: ConditionReport) -> None:
    print(f"\n=== {report.name} ({report.task_id}) ===")
    for invariant in INVARIANT_ORDER:
        result = report.invariants[invariant]
        print(f"{invariant}: {result.status}")
        if result.evidence:
            print(f"  evidence: {result.evidence}")
    if report.invariants["INV-6"].status != "skip":
        print("--- First summary text (eyeball quality check) ---")
        print(report.summary_text)
        print("---")


def print_summary_table(reports: list[ConditionReport]) -> None:
    condition_width = max(len("Condition"), *(len(report.name) for report in reports))
    print("\n=== Verification Report ===\n")
    header = "Condition".ljust(condition_width)
    header += "  " + "  ".join(invariant.ljust(5) for invariant in INVARIANT_ORDER)
    print(header)

    failures: list[str] = []
    for report in reports:
        row = report.name.ljust(condition_width)
        statuses = []
        for invariant in INVARIANT_ORDER:
            status = report.invariants[invariant].status
            statuses.append(status.ljust(5))
            if status == "FAIL":
                failures.append(f"{report.name} {invariant}")
        row += "  " + "  ".join(statuses)
        print(row)

    if failures:
        print(f"\nOverall: FAIL ({', '.join(failures)})")
    else:
        print("\nOverall: PASS")


def main(args: argparse.Namespace) -> int:
    config = run_phase1.load_config(Path(args.config))
    models = run_phase1.load_models(config, [args.model] if args.model else [])
    conditions = run_phase1.load_conditions(
        config, [args.condition] if args.condition else []
    )
    task_ids = _resolve_task_ids(config)
    llama_server = run_phase1.validate_runtime_environment()

    if not models:
        raise SystemExit("No models configured")

    model = models[0]
    reports: list[ConditionReport] = []

    print(f"Verifying model: {model.short_name}")
    print(f"Task id: {task_ids[0]}")
    for condition in conditions:
        report = verify_condition(config, model, condition, task_ids, llama_server)
        reports.append(report)
        print_condition_report(report)

    print_summary_table(reports)
    return (
        1
        if any(
            result.status == "FAIL"
            for report in reports
            for result in report.invariants.values()
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
