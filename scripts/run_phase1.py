#!/usr/bin/env python3
"""Run a configured thinking-token retention benchmark experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Suppress noisy litellm cost-tracking errors for local models
import logging

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("tau2.utils.llm_utils").setLevel(logging.WARNING)


def _register_model_costs() -> None:
    """Register custom model pricing so litellm stops complaining."""
    try:
        import litellm  # type: ignore[import-untyped]

        # Local models — free (GPU cost is tracked separately)
        for name in [
            "Qwen3.5-2B-Q8_0.gguf",
            "Qwen3.5-4B-Q8_0.gguf",
            "Qwen3.5-9B-Q8_0.gguf",
        ]:
            litellm.model_cost[name] = {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "max_tokens": 131072,
            }
            litellm.model_cost[f"openai/{name}"] = litellm.model_cost[name]

        # OpenRouter MiMo V2 Flash — user simulator and summarizer
        # Register both the alias and the versioned name OpenRouter resolves to
        _mimo_cost = {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000003,
            "max_tokens": 131072,
        }
        litellm.model_cost["openrouter/xiaomi/mimo-v2-flash"] = _mimo_cost
        litellm.model_cost["openrouter/xiaomi/mimo-v2-flash-20251210"] = _mimo_cost
        litellm.model_cost["xiaomi/mimo-v2-flash-20251210"] = _mimo_cost
    except ImportError:
        pass


_register_model_costs()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1.yaml"
# Defaults for Phase 1; overridden by config fields tasks_file / results_dir
_DEFAULT_TASKS_PATH = PROJECT_ROOT / "configs" / "phase1_tasks.json"
_DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "phase1"
DEFAULT_LLAMA_SERVER = "/workspace/llama.cpp/build/bin/llama-server"
_shutdown_requested = False
PROGRESS_FILE = "progress.json"
TASK_CHECKPOINTS_DIR = ".task_checkpoints"
TASK_RESULTS_SCRATCH = ".task_results.json"


@dataclass(frozen=True)
class ModelConfig:
    hf_repo: str
    hf_file: str
    short_name: str


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    enable_thinking: bool
    retention_strategy: str
    summarize_thinking: bool = False


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="path to experiment config"
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the first model, first condition, and first task only",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing results and rerun all selected conditions",
    )
    return parser.parse_args(argv)


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


def resolve_tasks_path(config: dict[str, Any]) -> Path:
    """Resolve the tasks file path from config or fall back to default."""
    tasks_file = config.get("experiment", {}).get("tasks_file")
    if tasks_file:
        return PROJECT_ROOT / tasks_file
    return _DEFAULT_TASKS_PATH


def resolve_results_root(config: dict[str, Any]) -> Path:
    """Resolve the results directory from config or fall back to default."""
    results_dir = config.get("experiment", {}).get("results_dir")
    if results_dir:
        return PROJECT_ROOT / results_dir
    return _DEFAULT_RESULTS_ROOT


def load_task_ids(config: dict[str, Any]) -> list[str]:
    tasks_path = resolve_tasks_path(config)
    if tasks_path.exists():
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        return list(payload.get("task_ids", []))
    subset = config["experiment"].get("task_subset", "?")
    return [f"<select {subset} telecom tasks with scripts/select_tasks.py>"]


def resolve_model_path(model: ModelConfig) -> str:
    """Resolve the local path for a GGUF model, downloading if needed.

    Uses huggingface_hub so the file lives in HF_HOME (persistent on RunPod).
    Returns the absolute path to the .gguf file.
    """
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

    return hf_hub_download(
        repo_id=model.hf_repo,
        filename=model.hf_file,
        token=os.environ.get("HF_TOKEN"),
    )


def configured_llama_server() -> str:
    return os.environ.get("LLAMA_SERVER_BIN", DEFAULT_LLAMA_SERVER)


def is_executable_file(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_file() and os.access(candidate, os.X_OK)


def resolve_llama_server_bin() -> str:
    configured = os.environ.get("LLAMA_SERVER_BIN")
    if configured and is_executable_file(configured):
        return configured

    if is_executable_file(DEFAULT_LLAMA_SERVER):
        return DEFAULT_LLAMA_SERVER

    if configured:
        raise SystemExit(
            "LLAMA_SERVER_BIN points to "
            f"{configured}, but neither it nor the default llama-server path is an "
            "executable file: "
            f"{DEFAULT_LLAMA_SERVER}"
        )

    raise SystemExit(
        "llama-server not found. Set LLAMA_SERVER_BIN to an executable file or "
        f"build the default binary at {DEFAULT_LLAMA_SERVER}"
    )


def validate_runtime_environment() -> str:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY not set — needed for user simulator and summarizer"
        )
    return resolve_llama_server_bin()


def build_llama_command(
    model_path: str, llama_config: dict[str, Any], llama_server: str
) -> list[str]:
    return [
        llama_server,
        "-m",
        model_path,
        "--port",
        str(llama_config["port"]),
        *shlex.split(llama_config["base_args"]),
    ]


def wait_for_server(
    process: subprocess.Popen[Any], port: int, timeout_seconds: int = 600
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"llama-server exited early with code {exit_code}")
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(3)
    raise TimeoutError(f"Timed out waiting for llama-server health at {url}")


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
    return f"openai/{model.short_name}"


def _qwen_sampling_params(model: ModelConfig, enable_thinking: bool) -> dict[str, Any]:
    """Return Qwen3.5 official recommended sampling params per model size and mode.

    Source: each model's HuggingFace model card "Best Practices" section.
    Task type: "general tasks" (τ²-bench is customer service, not coding).

    0.8B/2B use different non-thinking params than 4B/9B.
    Thinking params are identical across all sizes.
    """
    is_small = any(tag in model.short_name for tag in ("0.8b", "2b"))

    if enable_thinking:
        # Same for all sizes: thinking mode, general tasks
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
        }
    elif is_small:
        # 0.8B/2B: non-thinking mode, text tasks
        return {
            "temperature": 1.0,
            "top_p": 1.0,
            "presence_penalty": 2.0,
        }
    else:
        # 4B/9B: instruct (non-thinking) mode, general tasks
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
        }


def agent_llm_args(
    condition: ConditionConfig,
    port: int,
    model: ModelConfig,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_tokens = 8192
    if config and "generation" in config:
        max_tokens = config["generation"].get("max_tokens", 8192)
    sampling = _qwen_sampling_params(model, condition.enable_thinking)
    return {
        "api_base": f"http://127.0.0.1:{port}/v1",
        "api_key": "sk-no-key-required",
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "presence_penalty": sampling["presence_penalty"],
        "max_tokens": max_tokens,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": condition.enable_thinking,
            },
        },
    }


def print_plan(
    config: dict[str, Any],
    models: list[ModelConfig],
    conditions: list[ConditionConfig],
    task_ids: list[str],
    *,
    smoke: bool = False,
) -> None:
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    if smoke:
        task_count = 1 if task_ids else 0
    elif task_ids and task_ids[0].startswith("<select "):
        task_count = config["experiment"]["task_subset"]
    else:
        task_count = len(task_ids)
    total_runs = (
        len(models) * len(conditions) * config["experiment"]["trials"] * task_count
    )
    print(f"Experiment plan: {experiment_name}")
    print(
        f"- benchmark: {config['experiment']['benchmark']} "
        f"({config['experiment']['domain']}/{config['experiment']['task_split']})"
    )
    print(f"- server: llama.cpp ({configured_llama_server()})")
    print(f"- quantization: Q8_0 (uniform)")
    print(f"- user sim: {user_model_name(config)}")
    print(f"- projected configs: {len(models) * len(conditions)}")
    if smoke:
        print("- mode: smoke (first model, first condition, first task only)")
    print(f"- models: {', '.join(model.short_name for model in models)}")
    print(f"- conditions: {', '.join(condition.name for condition in conditions)}")
    print(f"- task ids: {', '.join(task_ids)}")
    print(f"- projected task runs: {total_runs}")
    for model in models:
        for condition in conditions:
            sampling = _qwen_sampling_params(model, condition.enable_thinking)
            print(
                f"  - {model.short_name} / {condition.name} | "
                f"thinking={'on' if condition.enable_thinking else 'off'} | "
                f"retention={condition.retention_strategy} | "
                f"temp={sampling['temperature']} top_p={sampling['top_p']} "
                f"pp={sampling['presence_penalty']}"
            )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _results_metadata(results_path: Path) -> tuple[int, bool] | None:
    payload = _read_json_file(results_path)
    if payload is None:
        return None
    simulations = payload.get("simulations")
    if not isinstance(simulations, list):
        return None
    has_meaningful_simulation = False
    for simulation in simulations:
        if not isinstance(simulation, dict):
            continue
        messages = simulation.get("messages")
        if isinstance(messages, list) and len(messages) > 5:
            has_meaningful_simulation = True
            break
    return len(simulations), has_meaningful_simulation


def _serialize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_serialize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_json_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_json_value(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _serialize_json_value(model_dump(mode="json"))
        except TypeError:
            return _serialize_json_value(model_dump())
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        return json.loads(str(model_dump_json()))
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _serialize_results_payload(results: Any) -> dict[str, Any]:
    payload = _serialize_json_value(results)
    if isinstance(payload, dict):
        payload.setdefault("simulations", [])
        return payload
    simulations = getattr(results, "simulations", [])
    return {"simulations": _serialize_json_value(simulations)}


def _task_checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / TASK_CHECKPOINTS_DIR


def _task_checkpoint_path(run_dir: Path, task_id: str) -> Path:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
    return _task_checkpoint_dir(run_dir) / f"{digest}.json"


def _task_results_scratch_path(run_dir: Path) -> Path:
    return run_dir / TASK_RESULTS_SCRATCH


def _read_progress(run_dir: Path) -> dict[str, Any] | None:
    return _read_json_file(run_dir / PROGRESS_FILE)


def _write_progress(
    run_dir: Path,
    completed_task_ids: list[str],
    *,
    num_simulations: int,
) -> None:
    write_summary(
        run_dir / PROGRESS_FILE,
        {
            "completed_task_ids": completed_task_ids,
            "num_completed_tasks": len(completed_task_ids),
            "num_simulations": num_simulations,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )


def _run_has_checkpoint_state(run_dir: Path) -> bool:
    checkpoint_dir = _task_checkpoint_dir(run_dir)
    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.json")):
        return True
    progress = _read_progress(run_dir) or {}
    completed_task_ids = progress.get("completed_task_ids")
    return isinstance(completed_task_ids, list) and bool(completed_task_ids)


def _load_task_checkpoints(run_dir: Path) -> dict[str, dict[str, Any]]:
    checkpoint_dir = _task_checkpoint_dir(run_dir)
    if not checkpoint_dir.exists():
        return {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for checkpoint_path in sorted(checkpoint_dir.glob("*.json")):
        payload = _read_json_file(checkpoint_path)
        if payload is None:
            continue
        task_id = payload.get("task_id")
        results_payload = payload.get("results")
        thinking_records = payload.get("thinking_records")
        if not isinstance(task_id, str):
            continue
        if not isinstance(results_payload, dict):
            continue
        if not isinstance(thinking_records, list):
            continue
        checkpoints[task_id] = payload
    return checkpoints


def _write_task_checkpoint(
    run_dir: Path,
    task_id: str,
    results_payload: dict[str, Any],
    thinking_records: list[dict[str, Any]],
) -> None:
    checkpoint_dir = _task_checkpoint_dir(run_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_summary(
        _task_checkpoint_path(run_dir, task_id),
        {
            "task_id": task_id,
            "results": results_payload,
            "thinking_records": thinking_records,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )


def _simulation_sort_key(
    simulation: dict[str, Any], task_order: dict[str, int], original_index: int
) -> tuple[int, int, int]:
    trial = simulation.get("trial", 0)
    if not isinstance(trial, int):
        trial = 0
    task_id = simulation.get("task_id")
    if not isinstance(task_id, str):
        task_id = ""
    task_index = task_order.get(task_id, len(task_order))
    return trial, task_index, original_index


def _aggregate_results_payload(
    task_ids: list[str],
    checkpoints: dict[str, dict[str, Any]],
    full_tasks: list[Any],
) -> dict[str, Any] | None:
    ordered_checkpoints = [
        checkpoints[task_id] for task_id in task_ids if task_id in checkpoints
    ]
    if not ordered_checkpoints:
        return None

    payload = copy.deepcopy(ordered_checkpoints[0]["results"])
    task_order = {task_id: index for index, task_id in enumerate(task_ids)}
    simulations: list[dict[str, Any]] = []
    for checkpoint in ordered_checkpoints:
        result_payload = checkpoint["results"]
        result_simulations = result_payload.get("simulations", [])
        if isinstance(result_simulations, list):
            simulations.extend(copy.deepcopy(result_simulations))

    simulations = [
        simulation
        for _, simulation in sorted(
            enumerate(simulations),
            key=lambda item: _simulation_sort_key(item[1], task_order, item[0]),
        )
    ]
    payload["simulations"] = simulations
    payload["tasks"] = _serialize_json_value(full_tasks)
    return payload


def _aggregate_thinking_records(
    task_ids: list[str], checkpoints: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        checkpoint = checkpoints.get(task_id)
        if checkpoint is None:
            continue
        records.extend(copy.deepcopy(checkpoint["thinking_records"]))
    return records


def _write_jsonl_records(
    path: Path, records: list[dict[str, Any]], *, append: bool
) -> None:
    if not records and append:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _rebuild_checkpoint_artifacts(
    run_dir: Path,
    task_ids: list[str],
    full_tasks: list[Any],
) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    checkpoints = _load_task_checkpoints(run_dir)
    completed_task_ids = [task_id for task_id in task_ids if task_id in checkpoints]
    results_payload = _aggregate_results_payload(task_ids, checkpoints, full_tasks)
    num_simulations = 0
    if results_payload is not None:
        simulations = results_payload.get("simulations", [])
        num_simulations = len(simulations) if isinstance(simulations, list) else 0
        write_summary(run_dir / "results.json", results_payload)
    thinking_records = _aggregate_thinking_records(task_ids, checkpoints)
    if checkpoints or (run_dir / "thinking_analysis.jsonl").exists():
        _write_jsonl_records(
            run_dir / "thinking_analysis.jsonl",
            thinking_records,
            append=False,
        )
    _write_progress(run_dir, completed_task_ids, num_simulations=num_simulations)
    return checkpoints, completed_task_ids, num_simulations


def _simulation_reward(simulation: Any) -> float:
    reward_info = (
        simulation.get("reward_info")
        if isinstance(simulation, dict)
        else getattr(simulation, "reward_info", None)
    )
    if isinstance(reward_info, dict):
        reward = reward_info.get("reward", 0.0)
    elif reward_info is not None:
        reward = getattr(reward_info, "reward", 0.0)
    else:
        reward = 0.0
    return float(reward or 0.0)


def _full_reward_count(simulations: list[Any]) -> int:
    return sum(1 for simulation in simulations if _simulation_reward(simulation) >= 1.0)


def _task_id_value(task: Any) -> str | None:
    if isinstance(task, str):
        return task
    task_id = getattr(task, "id", None)
    return task_id if isinstance(task_id, str) else None


def find_resumable_condition_run(
    results_root: Path,
    model_short_name: str,
    condition_name: str,
    *,
    task_ids: list[str],
) -> Path | None:
    pattern = f"{model_short_name}_{condition_name}_*/summary.json"
    candidates: list[tuple[int, Path]] = []
    for summary_path in sorted(results_root.glob(pattern), reverse=True):
        payload = _read_json_file(summary_path)
        if payload is None:
            continue
        if payload.get("status") == "complete":
            continue
        if payload.get("task_ids") != task_ids:
            continue
        run_dir = summary_path.parent
        score = (
            1
            if _run_has_checkpoint_state(run_dir) or (run_dir / "results.json").exists()
            else 0
        )
        candidates.append((score, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return candidates[0][1]


def apply_smoke_selection(
    models: list[ModelConfig],
    conditions: list[ConditionConfig],
    task_ids: list[str],
) -> tuple[list[ModelConfig], list[ConditionConfig], list[str]]:
    smoke_models = models[:1]
    smoke_conditions = conditions[:1]
    smoke_tasks = task_ids[:1]
    return smoke_models, smoke_conditions, smoke_tasks


def configure_condition_environment(
    condition: ConditionConfig, config: dict[str, Any]
) -> None:
    os.environ["RETENTION_STRATEGY"] = condition.retention_strategy
    os.environ["SUMMARIZE_THINKING"] = str(condition.summarize_thinking).lower()
    # Rate-limit safety: 1s delay between turns to stay under OpenRouter TPM limits
    os.environ["TURN_DELAY_SECONDS"] = str(
        config.get("generation", {}).get("turn_delay_seconds", 1.0)
    )
    if condition.summarize_thinking:
        summarizer_config = config.get("summarizer", {})
        os.environ["SUMMARIZER_MODEL"] = summarizer_config.get("model", "")
        os.environ["SUMMARIZER_PROMPT"] = summarizer_config.get("prompt", "")
    else:
        os.environ["SUMMARIZE_THINKING"] = "false"
        os.environ.pop("SUMMARIZER_MODEL", None)
        os.environ.pop("SUMMARIZER_PROMPT", None)


def completed_condition_runs(
    results_root: Path,
    model_short_name: str,
    condition_name: str,
    *,
    task_ids: list[str] | None = None,
    trials: int | None = None,
) -> list[Path]:
    pattern = f"{model_short_name}_{condition_name}_*/summary.json"
    matches: list[Path] = []
    for summary_path in sorted(results_root.glob(pattern)):
        payload = _read_json_file(summary_path)
        if payload is None:
            continue
        if payload.get("status") not in (None, "complete"):
            continue
        if payload.get("contaminated"):
            continue
        results_path = summary_path.parent / "results.json"
        results_metadata = _results_metadata(results_path)
        if results_metadata is None:
            continue
        results_count, has_meaningful_results = results_metadata
        if not has_meaningful_results:
            continue
        if task_ids is None and trials is None:
            matches.append(summary_path)
            continue
        if task_ids is not None and payload.get("task_ids") != task_ids:
            continue
        if trials is not None and results_count != len(task_ids or []) * trials:
            continue
        matches.append(summary_path)
    return matches


def collect_completed_conditions(
    results_root: Path,
    models: list[ModelConfig],
    conditions: list[ConditionConfig],
    task_ids: list[str],
    trials: int,
) -> list[tuple[str, str]]:
    completed: list[tuple[str, str]] = []
    for model in models:
        for condition in conditions:
            if completed_condition_runs(
                results_root,
                model.short_name,
                condition.name,
                task_ids=task_ids,
                trials=trials,
            ):
                completed.append((model.short_name, condition.name))
    return completed


def print_resume_summary(completed: list[tuple[str, str]]) -> None:
    if not completed:
        print("Starting fresh: 0 completed conditions found")
        return
    print(f"Resuming: skipping {len(completed)} completed conditions")
    for model_short_name, condition_name in completed:
        print(f"- {model_short_name} / {condition_name}")


def cleanup_partial_run(run_dir: Path, exc: BaseException | None = None) -> None:
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.json"
    if not run_dir.exists():
        return
    if results_path.exists() or _run_has_checkpoint_state(run_dir):
        payload = _read_json_file(summary_path) or {}
        payload["status"] = "failed"
        payload["error"] = (
            str(exc) if exc is not None else "Run failed before completion"
        )
        payload["generated_at"] = datetime.now(UTC).isoformat()
        write_summary(summary_path, payload)
        print(f"Preserved failed run: {run_dir.name}")
        return
    shutil.rmtree(run_dir)
    print(f"Cleaned up partial run: {run_dir.name}")


def execute_condition_run_with_cleanup(
    run_dir: Path,
    config: dict[str, Any],
    experiment: dict[str, Any],
    user_llm: str,
    model: ModelConfig,
    condition: ConditionConfig,
    task_ids: list[str],
    port: int,
) -> dict[str, Any]:
    try:
        return execute_condition_run(
            run_dir=run_dir,
            config=config,
            experiment=experiment,
            user_llm=user_llm,
            model=model,
            condition=condition,
            task_ids=task_ids,
            port=port,
        )
    except BaseException as exc:
        cleanup_partial_run(run_dir, exc)
        raise


def request_graceful_shutdown(_signum: int, frame: Any) -> None:
    del frame
    global _shutdown_requested
    if _shutdown_requested:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.default_int_handler(signal.SIGINT, None)
        return
    _shutdown_requested = True
    print(
        "\nGraceful shutdown requested. Will stop after current condition.\n"
        "Press Ctrl+C again to force quit.",
        flush=True,
    )


def exit_if_shutdown_requested(completed_this_session: list[str]) -> None:
    if not _shutdown_requested:
        return
    print("Graceful shutdown complete. Completed this session:")
    if completed_this_session:
        for label in completed_this_session:
            print(f"- {label}")
    else:
        print("- None")
    sys.exit(0)


def build_thinking_records(results, condition: ConditionConfig) -> list[dict[str, Any]]:
    """Build terminal-state thinking metadata from completed trajectories.

    `retained_at_end` reflects whether a turn's thinking is still present in the
    final conversation state after the run finishes. Prompt-time retention is
    applied separately at each LLM call by `src.thinking.apply_retention_strategy()`.
    """

    from src.thinking import (
        THINK_SUMMARY_PATTERN,
        count_thinking_tokens_approx,
        extract_thinking,
        identify_turn_boundaries,
    )

    window_size = None
    if condition.retention_strategy == "retain_all":
        window_size = None
    elif condition.retention_strategy == "strip_all":
        window_size = None
    elif condition.retention_strategy.startswith("window_"):
        window_size = int(condition.retention_strategy.split("_", 1)[1])
    else:
        raise ValueError(
            f"Unsupported retention strategy: {condition.retention_strategy}"
        )

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
            combined_summary = ""
            if condition.summarize_thinking:
                summary_blocks = []
                for message in assistant_messages:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        blocks = [
                            m.strip() for m in THINK_SUMMARY_PATTERN.findall(content)
                        ]
                        summary_blocks.extend(block for block in blocks if block)
                combined_summary = "\n\n".join(summary_blocks)
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
                    "raw_thinking_chars": len(combined_thinking),
                    "raw_thinking_tokens_approx": count_thinking_tokens_approx(
                        combined_thinking
                    ),
                    "summary_chars": (
                        len(combined_summary) if condition.summarize_thinking else None
                    ),
                    "summary_tokens_approx": (
                        count_thinking_tokens_approx(combined_summary)
                        if condition.summarize_thinking
                        else None
                    ),
                    "summarizer_input_tokens": None,
                    "summarizer_output_tokens": None,
                    "retained_at_end": retained,
                    "window_size": window_size,
                    "prompt_tokens_total": None,
                    "prompt_tokens_cached": None,
                    "prompt_tokens_evaluated": None,
                    "generation_tokens": None,
                    "thinking_tokens_in_generation": None,
                    "source": "tau2 trajectory",
                }
            )
    return records


def _sum_optional_agent_values(
    agent_records: list[dict[str, Any]], field: str
) -> int | None:
    values = [
        value for record in agent_records if (value := record.get(field)) is not None
    ]
    if not values:
        return None
    return sum(values)


def merge_agent_thinking_records(
    records: list[dict[str, Any]], agent_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged_records: list[dict[str, Any]] = []
    agent_idx = 0
    expected_agent_records = sum(
        int(record["assistant_message_count"]) for record in records
    )
    task_run_count = len({(record["task_id"], record["trial"]) for record in records})

    for record in records:
        assistant_count = int(record["assistant_message_count"])
        if assistant_count == 0:
            merged_records.append(record)
            continue

        next_agent_idx = agent_idx + assistant_count
        if next_agent_idx > len(agent_records):
            raise ValueError(
                "Agent thinking records do not align with trajectory assistant messages: "
                f"expected {expected_agent_records} assistant-message records across "
                f"{task_run_count} task/trial runs, got {len(agent_records)}"
            )

        turn_agent_records = agent_records[agent_idx:next_agent_idx]
        agent_idx = next_agent_idx

        merged_record = dict(record)
        merged_record["raw_thinking_chars"] = sum(
            int(agent_record.get("raw_thinking_chars", 0))
            for agent_record in turn_agent_records
        )
        merged_record["raw_thinking_tokens_approx"] = sum(
            int(agent_record.get("raw_thinking_tokens_approx", 0))
            for agent_record in turn_agent_records
        )
        merged_record["summary_chars"] = _sum_optional_agent_values(
            turn_agent_records, "summary_chars"
        )
        merged_record["summary_tokens_approx"] = _sum_optional_agent_values(
            turn_agent_records, "summary_tokens_approx"
        )
        merged_record["summarizer_input_tokens"] = _sum_optional_agent_values(
            turn_agent_records, "summarizer_input_tokens"
        )
        merged_record["summarizer_output_tokens"] = _sum_optional_agent_values(
            turn_agent_records, "summarizer_output_tokens"
        )
        merged_records.append(merged_record)

    if agent_idx != len(agent_records):
        extra = len(agent_records) - agent_idx
        print(
            f"  WARNING: discarding {extra} overflow agent thinking records across "
            f"{task_run_count} task/trial runs; expected {agent_idx} assistant-message "
            f"records from trajectory data, got {len(agent_records)} "
            "(likely from tau2 task retries)"
        )

    return merged_records


def abort_on_thinking_contamination(run_dir: Path, summary: dict[str, Any]) -> None:
    analysis_path = run_dir / "thinking_analysis.jsonl"
    if not analysis_path.exists():
        return

    leaked = 0
    for line in analysis_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        raw_tokens = rec.get("raw_thinking_tokens_approx")
        if raw_tokens is None:
            raw_tokens = rec.get("thinking_tokens_approx", 0)
        if raw_tokens > 0:
            leaked += 1

    if not leaked:
        return

    print(
        f"  WARNING: thinking_off leaked thinking in {leaked} turns — "
        "baseline may be contaminated"
    )
    summary["contaminated"] = True
    write_summary(run_dir / "summary.json", summary)
    raise RuntimeError(
        f"thinking_off contamination detected in {leaked} turns; aborting run"
    )


def collect_thinking_analysis_records(
    results, condition: ConditionConfig
) -> list[dict[str, Any]]:
    from src.agent import get_thinking_records

    records = build_thinking_records(results, condition)
    return merge_agent_thinking_records(records, get_thinking_records())


def save_thinking_analysis(
    run_dir: Path,
    results,
    condition: ConditionConfig,
    *,
    append: bool = False,
) -> None:
    records = collect_thinking_analysis_records(results, condition)
    output_path = run_dir / "thinking_analysis.jsonl"
    _write_jsonl_records(output_path, records, append=append)


def execute_condition_run(
    run_dir: Path,
    config: dict[str, Any],
    experiment: dict[str, Any],
    user_llm: str,
    model: ModelConfig,
    condition: ConditionConfig,
    task_ids: list[str],
    port: int,
) -> dict[str, Any]:
    import src.register  # noqa: F401
    from src.agent import clear_thinking_records
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
    checkpoints, completed_task_ids, num_simulations = _rebuild_checkpoint_artifacts(
        run_dir,
        task_ids,
        tasks,
    )
    initial_summary = {
        "model": model.short_name,
        "model_hf_repo": model.hf_repo,
        "model_hf_file": model.hf_file,
        "condition": condition.name,
        "enable_thinking": condition.enable_thinking,
        "retention_strategy": condition.retention_strategy,
        "task_ids": task_ids,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    def _current_simulations() -> list[dict[str, Any]]:
        payload = _aggregate_results_payload(task_ids, checkpoints, tasks)
        if payload is None:
            return []
        simulations = payload.get("simulations", [])
        return simulations if isinstance(simulations, list) else []

    def _write_running_summary() -> None:
        simulations = _current_simulations()
        write_summary(
            run_dir / "summary.json",
            {
                **initial_summary,
                "status": "running",
                "completed_task_ids": completed_task_ids,
                "num_simulations": len(simulations),
                "full_reward_count": _full_reward_count(simulations),
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )

    _write_running_summary()
    run_config = TextRunConfig(
        domain=experiment["domain"],
        task_set_name=experiment["domain"],
        task_split_name=experiment["task_split"],
        task_ids=task_ids,
        num_trials=experiment["trials"],
        agent="thinking_retention",
        llm_agent=agent_llm_name(model),
        llm_args_agent=agent_llm_args(condition, port, model, config),
        user="user_simulator",
        llm_user=user_llm,
        llm_args_user={},
        max_concurrency=1,
        verbose_logs=False,
    )
    remaining_tasks = [
        task for task in tasks if _task_id_value(task) not in set(completed_task_ids)
    ]
    scratch_results_path = _task_results_scratch_path(run_dir)

    try:
        for task in remaining_tasks:
            task_id = _task_id_value(task)
            if task_id is None:
                raise RuntimeError("Task is missing a string id")

            clear_thinking_records()
            scratch_results_path.unlink(missing_ok=True)
            task_results = run_tasks(
                run_config,
                [task],
                save_path=scratch_results_path,
                save_dir=run_dir,
                console_display=True,
            )
            task_results_payload = _serialize_results_payload(task_results)
            task_thinking_records = collect_thinking_analysis_records(
                task_results, condition
            )
            _write_task_checkpoint(
                run_dir,
                task_id,
                task_results_payload,
                task_thinking_records,
            )
            checkpoints[task_id] = {
                "task_id": task_id,
                "results": task_results_payload,
                "thinking_records": task_thinking_records,
            }
            _write_jsonl_records(
                run_dir / "thinking_analysis.jsonl",
                task_thinking_records,
                append=True,
            )
            results_payload = _aggregate_results_payload(task_ids, checkpoints, tasks)
            if results_payload is not None:
                simulations = results_payload.get("simulations", [])
                num_simulations = (
                    len(simulations) if isinstance(simulations, list) else 0
                )
                write_summary(run_dir / "results.json", results_payload)
            completed_task_ids = [
                task_id for task_id in task_ids if task_id in checkpoints
            ]
            _write_progress(
                run_dir,
                completed_task_ids,
                num_simulations=num_simulations,
            )
            _write_running_summary()
    finally:
        scratch_results_path.unlink(missing_ok=True)

    simulations = _current_simulations()
    summary = {
        **initial_summary,
        "task_ids": task_ids,
        "status": "complete",
        "completed_task_ids": completed_task_ids,
        "num_simulations": len(simulations),
        "full_reward_count": _full_reward_count(simulations),
        "contaminated": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_summary(run_dir / "summary.json", summary)
    return summary


def main(args: argparse.Namespace) -> None:
    global _shutdown_requested
    _shutdown_requested = False
    config = load_config(Path(args.config))
    models = load_models(config, args.model)
    conditions = load_conditions(config, args.condition)
    task_ids = load_task_ids(config)
    results_root = resolve_results_root(config)

    if args.smoke:
        models, conditions, task_ids = apply_smoke_selection(
            models, conditions, task_ids
        )

    print_plan(config, models, conditions, task_ids, smoke=args.smoke)
    if args.dry_run:
        return
    if task_ids and task_ids[0].startswith("<select "):
        raise SystemExit(
            "No task selection found. Run `python scripts/select_tasks.py` first."
        )

    previous_sigint_handler = None
    if not args.smoke:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_graceful_shutdown)

    try:
        llama_server = validate_runtime_environment()

        # Ensure HF cache is on persistent volume (RunPod wipes /root/.cache on restart)
        if "HF_HOME" not in os.environ and Path("/workspace").exists():
            os.environ["HF_HOME"] = "/workspace/hf_cache"
            os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_cache"

        results_root.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            print("Fresh run: ignoring existing results")
            completed_conditions: set[tuple[str, str]] = set()
        elif args.smoke:
            completed_conditions = set()
        else:
            completed_conditions = set(
                collect_completed_conditions(
                    results_root,
                    models,
                    conditions,
                    task_ids,
                    int(config["experiment"]["trials"]),
                )
            )
            print_resume_summary(sorted(completed_conditions))

        all_summaries: list[dict[str, Any]] = []
        completed_this_session: list[str] = []
        llama_config = config["llama"]
        port = int(llama_config["port"])
        user_llm = user_model_name(config)
        resume_enabled = not args.fresh and not args.smoke

        for model in models:
            model_timestamp = utc_timestamp()
            log_path = results_root / f"{model.short_name}_{model_timestamp}_llama.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            process = None
            log_handle = log_path.open("w", encoding="utf-8")
            try:
                print(f"\nResolving GGUF path for {model.short_name}...")
                model_path = resolve_model_path(model)
                print(f"  -> {model_path}")
                command = build_llama_command(model_path, llama_config, llama_server)
                print(
                    f"Starting llama-server for {model.short_name}: {' '.join(command)}"
                )
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wait_for_server(process, port)
                print(f"llama-server ready for {model.short_name}")

                for condition in conditions:
                    exit_if_shutdown_requested(completed_this_session)
                    condition_key = (model.short_name, condition.name)
                    if resume_enabled and condition_key in completed_conditions:
                        print(
                            f"Skipping completed {model.short_name} / {condition.name}"
                        )
                        continue

                    run_dir = None
                    if resume_enabled:
                        run_dir = find_resumable_condition_run(
                            results_root,
                            model.short_name,
                            condition.name,
                            task_ids=task_ids,
                        )
                    if run_dir is None:
                        run_timestamp = utc_timestamp()
                        run_dir = (
                            results_root
                            / f"{model.short_name}_{condition.name}_{run_timestamp}"
                        )
                        print(
                            f"Running {model.short_name} / {condition.name} -> {run_dir}"
                        )
                    else:
                        print(
                            f"Resuming {model.short_name} / {condition.name} -> {run_dir}"
                        )
                    configure_condition_environment(condition, config)
                    summary = execute_condition_run_with_cleanup(
                        run_dir=run_dir,
                        config=config,
                        experiment=config["experiment"],
                        user_llm=user_llm,
                        model=model,
                        condition=condition,
                        task_ids=task_ids,
                        port=port,
                    )
                    all_summaries.append(summary)
                    completed_this_session.append(
                        f"{model.short_name} / {condition.name}"
                    )

                    # Validate: thinking_off should produce no thinking content
                    if not condition.enable_thinking:
                        abort_on_thinking_contamination(run_dir, summary)

                    completed_conditions.add(condition_key)
                    exit_if_shutdown_requested(completed_this_session)
            finally:
                stop_process(process)
                log_handle.close()

        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "runs": all_summaries,
        }
        write_summary(results_root / f"summary_{utc_timestamp()}.json", manifest)
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        main(cli_args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BaseException as exc:
        if cli_args.smoke:
            message = str(exc)
            if isinstance(exc, SystemExit):
                code = exc.code
                message = code if isinstance(code, str) else f"exit code {code}"
            print(f"SMOKE TEST FAILED: {message}", file=sys.stderr)
        raise
    else:
        if cli_args.smoke:
            print("SMOKE TEST PASSED")
