#!/usr/bin/env python3
"""Run Phase 1: 10 telecom tasks x 4 models x 4 conditions = 160 runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1.yaml"
TASKS_PATH = PROJECT_ROOT / "configs" / "phase1_tasks.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase1"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    short_name: str
    extra_vllm_args: str = ""


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    enable_thinking: bool
    retention_strategy: str


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="path to phase1.yaml"
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model short_name to run; repeat for multiple models",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="condition name to run; repeat for multiple conditions",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the run plan only"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(config: dict[str, Any], selected: list[str]) -> list[ModelConfig]:
    models = [ModelConfig(**item) for item in config["models"]]
    if not selected:
        return models
    selected_set = set(selected)
    filtered = [model for model in models if model.short_name in selected_set]
    missing = selected_set - {model.short_name for model in filtered}
    if missing:
        raise SystemExit(f"Unknown model(s): {sorted(missing)}")
    return filtered


def load_conditions(
    config: dict[str, Any], selected: list[str]
) -> list[ConditionConfig]:
    conditions = [ConditionConfig(**item) for item in config["conditions"]]
    if not selected:
        return conditions
    selected_set = set(selected)
    filtered = [condition for condition in conditions if condition.name in selected_set]
    missing = selected_set - {condition.name for condition in filtered}
    if missing:
        raise SystemExit(f"Unknown condition(s): {sorted(missing)}")
    return filtered


def load_task_ids(config: dict[str, Any]) -> list[str]:
    if TASKS_PATH.exists():
        payload = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
        return list(payload.get("task_ids", []))
    subset = config["experiment"]["task_subset"]
    return [f"<select {subset} telecom tasks with scripts/select_tasks.py>"]


def build_vllm_command(model: ModelConfig, vllm_config: dict[str, Any]) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        model.id,
        "--port",
        str(vllm_config["port"]),
        *shlex.split(vllm_config["base_args"]),
    ]
    if model.extra_vllm_args:
        cmd.extend(shlex.split(model.extra_vllm_args))
    return cmd


def wait_for_vllm(
    process: subprocess.Popen[Any], port: int, timeout_seconds: int = 900
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"vLLM exited early with code {exit_code}")
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for vLLM health endpoint at {url}")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def user_model_name(config: dict[str, Any]) -> str:
    return f"{config['user_sim']['provider']}/{config['user_sim']['model']}"


def agent_llm_name(model: ModelConfig) -> str:
    return f"hosted_vllm/{model.id}"


def agent_llm_args(condition: ConditionConfig, port: int) -> dict[str, Any]:
    return {
        "api_base": f"http://127.0.0.1:{port}/v1",
        "api_key": os.environ.get("VLLM_API_KEY", ""),
        "temperature": 0.0,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": condition.enable_thinking,
            }
        },
    }


def print_plan(
    config: dict[str, Any],
    models: list[ModelConfig],
    conditions: list[ConditionConfig],
    task_ids: list[str],
) -> None:
    task_count = (
        config["experiment"]["task_subset"]
        if task_ids and task_ids[0].startswith("<select ")
        else len(task_ids)
    )
    total_runs = (
        len(models) * len(conditions) * config["experiment"]["trials"] * task_count
    )
    print("Phase 1 plan")
    print(
        f"- benchmark: {config['experiment']['benchmark']} ({config['experiment']['domain']}/{config['experiment']['task_split']})"
    )
    print(f"- models: {', '.join(model.short_name for model in models)}")
    print(f"- conditions: {', '.join(condition.name for condition in conditions)}")
    print(f"- task ids: {', '.join(task_ids)}")
    print(f"- projected task runs: {total_runs}")
    for model in models:
        for condition in conditions:
            print(
                f"  - {model.short_name} / {condition.name} | "
                f"thinking={'on' if condition.enable_thinking else 'off'} | "
                f"retention={condition.retention_strategy}"
            )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_thinking_records(results, condition: ConditionConfig) -> list[dict[str, Any]]:
    from src.thinking import (
        count_thinking_tokens_approx,
        extract_thinking,
        identify_turn_boundaries,
    )

    window_size = None
    if condition.retention_strategy.startswith("window_"):
        window_size = int(condition.retention_strategy.split("_", 1)[1])

    records: list[dict[str, Any]] = []
    for simulation in results.simulations:
        messages = (
            simulation.get_messages()
            if hasattr(simulation, "get_messages")
            else list(simulation.messages or [])
        )
        user_indices = identify_turn_boundaries(messages)
        if not user_indices:
            continue
        assistant_turns: list[int] = []
        assistant_counts_by_turn: dict[int, int] = {}
        for turn_index, user_index in enumerate(user_indices):
            next_user_index = (
                user_indices[turn_index + 1]
                if turn_index + 1 < len(user_indices)
                else len(messages)
            )
            turn_messages = messages[user_index:next_user_index]
            assistant_messages = [
                message
                for message in turn_messages
                if getattr(message, "role", None) == "assistant"
            ]
            assistant_counts_by_turn[turn_index] = len(assistant_messages)
            if assistant_messages:
                assistant_turns.append(turn_index)

        keep_turns: set[int] = set()
        if condition.retention_strategy == "retain_all":
            keep_turns = set(assistant_turns)
        elif window_size is not None:
            keep_turns = set(assistant_turns[-window_size:])

        for turn_index, user_index in enumerate(user_indices):
            next_user_index = (
                user_indices[turn_index + 1]
                if turn_index + 1 < len(user_indices)
                else len(messages)
            )
            turn_messages = messages[user_index:next_user_index]
            assistant_messages = [
                message
                for message in turn_messages
                if getattr(message, "role", None) == "assistant"
            ]
            thinking_text = []
            for message in assistant_messages:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    extracted, _ = extract_thinking(content)
                    if extracted:
                        thinking_text.append(extracted)

            combined_thinking = "\n\n".join(thinking_text)
            retained = (
                turn_index in keep_turns and assistant_counts_by_turn[turn_index] > 0
            )

            records.append(
                {
                    "task_id": simulation.task_id,
                    "trial": simulation.trial,
                    "turn_index": turn_index,
                    "retention_strategy": condition.retention_strategy,
                    "assistant_message_count": len(assistant_messages),
                    "thinking_text_chars": len(combined_thinking),
                    "thinking_tokens_approx": count_thinking_tokens_approx(
                        combined_thinking
                    ),
                    "retained_for_future_prompts": retained,
                    "window_size": window_size,
                    "source": "tau2 trajectory",
                }
            )
    return records


def save_thinking_analysis(run_dir: Path, results, condition: ConditionConfig) -> None:
    records = build_thinking_records(results, condition)
    output_path = run_dir / "thinking_analysis.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def execute_condition_run(
    run_dir: Path,
    experiment: dict[str, Any],
    user_llm: str,
    model: ModelConfig,
    condition: ConditionConfig,
    task_ids: list[str],
    port: int,
) -> dict[str, Any]:
    import src.register  # noqa: F401
    from tau2.data_model.simulation import TextRunConfig  # type: ignore[import-not-found]
    from tau2.runner.batch import run_tasks  # type: ignore[import-not-found]
    from tau2.runner.helpers import get_tasks  # type: ignore[import-not-found]

    if task_ids and task_ids[0].startswith("<select "):
        raise RuntimeError(
            "No task selection found. Run `python scripts/select_tasks.py` first."
        )

    tasks = get_tasks(
        experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = TextRunConfig(
        domain=experiment["domain"],
        task_set_name=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
        num_trials=experiment["trials"],
        agent="thinking_retention",
        llm_agent=agent_llm_name(model),
        llm_args_agent=agent_llm_args(condition, port),
        user="user_simulator",
        llm_user=user_llm,
        llm_args_user={},
        max_concurrency=1,
        verbose_logs=False,
    )

    results = run_tasks(
        config,
        tasks,
        save_path=run_dir / "results.json",
        save_dir=run_dir,
        console_display=True,
    )
    save_thinking_analysis(run_dir, results, condition)

    passed = sum(
        1
        for simulation in results.simulations
        if simulation.reward_info is not None and simulation.reward_info.reward >= 1.0
    )
    summary = {
        "model": model.short_name,
        "model_id": model.id,
        "condition": condition.name,
        "enable_thinking": condition.enable_thinking,
        "retention_strategy": condition.retention_strategy,
        "task_ids": task_ids,
        "num_simulations": len(results.simulations),
        "full_reward_count": passed,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_summary(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    models = load_models(config, args.model)
    conditions = load_conditions(config, args.condition)
    task_ids = load_task_ids(config)

    print_plan(config, models, conditions, task_ids)
    if args.dry_run:
        return
    if task_ids and task_ids[0].startswith("<select "):
        raise SystemExit(
            "No task selection found. Run `python scripts/select_tasks.py` first."
        )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    port = int(config["vllm"]["port"])
    user_llm = user_model_name(config)

    for model in models:
        model_timestamp = utc_timestamp()
        model_log_path = RESULTS_ROOT / f"{model.short_name}_{model_timestamp}_vllm.log"
        model_log_path.parent.mkdir(parents=True, exist_ok=True)
        process = None
        log_handle = model_log_path.open("w", encoding="utf-8")
        try:
            command = build_vllm_command(model, config["vllm"])
            print(f"\nStarting vLLM for {model.short_name}: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_for_vllm(process, port)
            print(f"vLLM ready for {model.short_name}")

            for condition in conditions:
                run_timestamp = utc_timestamp()
                run_dir = (
                    RESULTS_ROOT
                    / f"{model.short_name}_{condition.name}_{run_timestamp}"
                )
                print(f"Running {model.short_name} / {condition.name} -> {run_dir}")
                os.environ["RETENTION_STRATEGY"] = condition.retention_strategy
                summary = execute_condition_run(
                    run_dir=run_dir,
                    experiment=config["experiment"],
                    user_llm=user_llm,
                    model=model,
                    condition=condition,
                    task_ids=task_ids,
                    port=port,
                )
                all_summaries.append(summary)
        finally:
            stop_process(process)
            log_handle.close()

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": all_summaries,
    }
    write_summary(RESULTS_ROOT / f"summary_{utc_timestamp()}.json", manifest)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
