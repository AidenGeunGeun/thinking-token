from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest import mock

from scripts import run_phase1


@dataclass
class MockMessage:
    role: str
    content: str | None = None


@dataclass
class MockSimulation:
    task_id: str
    trial: int
    messages: list[MockMessage]

    def get_messages(self) -> list[MockMessage]:
        return self.messages


@dataclass
class MockResults:
    simulations: list[MockSimulation]


class RunPhase1Test(unittest.TestCase):
    def setUp(self) -> None:
        run_phase1._shutdown_requested = False

    def test_build_thinking_records_marks_terminal_window_state(self) -> None:
        results = MockResults(
            simulations=[
                MockSimulation(
                    task_id="telecom_001",
                    trial=0,
                    messages=[
                        MockMessage("user", "u1"),
                        MockMessage("assistant", "<think>t1</think>a1"),
                        MockMessage("user", "u2"),
                        MockMessage("assistant", "<think>t2</think>a2"),
                        MockMessage("user", "u3"),
                        MockMessage("assistant", "<think>t3</think>a3"),
                        MockMessage("user", "u4"),
                        MockMessage("assistant", "<think>t4</think>a4"),
                    ],
                )
            ]
        )
        condition = run_phase1.ConditionConfig(
            name="raw_window3",
            enable_thinking=True,
            retention_strategy="window_3",
        )

        records = run_phase1.build_thinking_records(results, condition)

        self.assertEqual(
            [record["retained_at_end"] for record in records], [False, True, True, True]
        )
        self.assertEqual(records[0]["raw_thinking_chars"], len("t1"))
        self.assertNotIn("retained_for_future_prompts", records[0])

    def test_build_thinking_records_emits_schema_fields(self) -> None:
        """Verify all DATA_SCHEMA.md fields are present in records."""
        results = MockResults(
            simulations=[
                MockSimulation(
                    task_id="telecom_001",
                    trial=0,
                    messages=[
                        MockMessage("user", "u1"),
                        MockMessage("assistant", "<think>reasoning</think>reply"),
                    ],
                )
            ]
        )
        condition = run_phase1.ConditionConfig(
            name="strip_all",
            enable_thinking=True,
            retention_strategy="strip_all",
            summarize_thinking=False,
        )

        records = run_phase1.build_thinking_records(results, condition)

        self.assertEqual(len(records), 1)
        expected_keys = {
            "task_id",
            "trial",
            "turn_index",
            "retention_strategy",
            "assistant_message_count",
            "raw_thinking_chars",
            "raw_thinking_tokens_approx",
            "summary_chars",
            "summary_tokens_approx",
            "summarizer_input_tokens",
            "summarizer_output_tokens",
            "retained_at_end",
            "window_size",
            "prompt_tokens_total",
            "prompt_tokens_cached",
            "prompt_tokens_evaluated",
            "generation_tokens",
            "thinking_tokens_in_generation",
            "source",
        }
        self.assertEqual(set(records[0].keys()), expected_keys)

    def test_build_thinking_records_extracts_summary_metrics_when_enabled(self) -> None:
        results = MockResults(
            simulations=[
                MockSimulation(
                    task_id="telecom_001",
                    trial=0,
                    messages=[
                        MockMessage("user", "u1"),
                        MockMessage(
                            "assistant",
                            "<think_summary>condensed note</think_summary>reply",
                        ),
                    ],
                )
            ]
        )
        condition = run_phase1.ConditionConfig(
            name="summary_window3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )

        records = run_phase1.build_thinking_records(results, condition)

        self.assertEqual(records[0]["raw_thinking_chars"], 0)
        self.assertEqual(records[0]["summary_chars"], len("condensed note"))
        self.assertIsNone(records[0]["summarizer_input_tokens"])
        self.assertIsNone(records[0]["summarizer_output_tokens"])

    def test_save_thinking_analysis_merges_agent_records_by_turn(self) -> None:
        results = MockResults(
            simulations=[
                MockSimulation(
                    task_id="telecom_001",
                    trial=0,
                    messages=[
                        MockMessage("user", "u1"),
                        MockMessage("assistant", "first reply"),
                        MockMessage("assistant", "second reply"),
                        MockMessage("user", "u2"),
                        MockMessage("assistant", "third reply"),
                    ],
                )
            ]
        )
        condition = run_phase1.ConditionConfig(
            name="summary_window3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )
        agent_records = [
            {
                "raw_thinking_chars": 12,
                "raw_thinking_tokens_approx": 3,
                "summary_chars": None,
                "summary_tokens_approx": None,
                "summarizer_input_tokens": None,
                "summarizer_output_tokens": None,
                "has_tool_calls": False,
            },
            {
                "raw_thinking_chars": 4,
                "raw_thinking_tokens_approx": 1,
                "summary_chars": 6,
                "summary_tokens_approx": 1,
                "summarizer_input_tokens": 100,
                "summarizer_output_tokens": 10,
                "has_tool_calls": False,
            },
            {
                "raw_thinking_chars": 8,
                "raw_thinking_tokens_approx": 2,
                "summary_chars": 5,
                "summary_tokens_approx": 1,
                "summarizer_input_tokens": 50,
                "summarizer_output_tokens": 7,
                "has_tool_calls": False,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            with mock.patch(
                "src.agent.get_thinking_records", return_value=agent_records
            ):
                run_phase1.save_thinking_analysis(run_dir, results, condition)

            records = [
                json.loads(line)
                for line in (run_dir / "thinking_analysis.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["assistant_message_count"], 2)
        self.assertEqual(records[0]["raw_thinking_chars"], 16)
        self.assertEqual(records[0]["raw_thinking_tokens_approx"], 4)
        self.assertEqual(records[0]["summary_chars"], 6)
        self.assertEqual(records[0]["summary_tokens_approx"], 1)
        self.assertEqual(records[0]["summarizer_input_tokens"], 100)
        self.assertEqual(records[0]["summarizer_output_tokens"], 10)
        self.assertEqual(records[1]["raw_thinking_chars"], 8)
        self.assertEqual(records[1]["summary_chars"], 5)
        self.assertEqual(records[1]["summarizer_input_tokens"], 50)
        self.assertEqual(records[1]["summarizer_output_tokens"], 7)

    def test_execute_condition_run_clears_thinking_records_before_run_tasks(
        self,
    ) -> None:
        import importlib

        agent_module = importlib.import_module("src.agent")
        agent_module.clear_thinking_records()
        agent_module._thinking_records.append({"stale": True})

        class FakeTextRunConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        def fake_get_tasks(
            domain: str, task_split_name: str, task_ids: list[str]
        ) -> list[str]:
            self.assertEqual(domain, "telecom")
            self.assertEqual(task_split_name, "phase1")
            self.assertEqual(task_ids, ["task-1"])
            return ["task-1"]

        def fake_run_tasks(
            config: object,
            tasks: object,
            save_path: object,
            save_dir: Path,
            console_display: object,
        ) -> MockResults:
            self.assertEqual(agent_module.get_thinking_records(), [])
            self.assertIsInstance(config, FakeTextRunConfig)
            self.assertEqual(tasks, ["task-1"])
            results_path = save_dir / "results.json"
            self.assertEqual(save_path, results_path)
            self.assertTrue(console_display)
            summary = json.loads(
                (save_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "running")
            self.assertEqual(summary["task_ids"], ["task-1"])
            results_path.write_text(json.dumps({"simulations": []}), encoding="utf-8")
            return MockResults(simulations=[])

        tau2_module = ModuleType("tau2")
        tau2_data_model_module = ModuleType("tau2.data_model")
        tau2_data_model_simulation_module = ModuleType("tau2.data_model.simulation")
        tau2_runner_module = ModuleType("tau2.runner")
        tau2_runner_batch_module = ModuleType("tau2.runner.batch")
        tau2_runner_helpers_module = ModuleType("tau2.runner.helpers")
        setattr(tau2_data_model_simulation_module, "TextRunConfig", FakeTextRunConfig)
        setattr(tau2_runner_batch_module, "run_tasks", fake_run_tasks)
        setattr(tau2_runner_helpers_module, "get_tasks", fake_get_tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            with mock.patch.dict(
                sys.modules,
                {
                    "src.register": ModuleType("src.register"),
                    "tau2": tau2_module,
                    "tau2.data_model": tau2_data_model_module,
                    "tau2.data_model.simulation": tau2_data_model_simulation_module,
                    "tau2.runner": tau2_runner_module,
                    "tau2.runner.batch": tau2_runner_batch_module,
                    "tau2.runner.helpers": tau2_runner_helpers_module,
                },
                clear=False,
            ):
                with mock.patch.object(run_phase1, "save_thinking_analysis"):
                    summary = run_phase1.execute_condition_run(
                        run_dir=run_dir,
                        experiment={
                            "domain": "telecom",
                            "task_split": "phase1",
                            "trials": 1,
                        },
                        user_llm="openrouter/user-sim",
                        model=run_phase1.ModelConfig("repo/a", "a.gguf", "model-a"),
                        condition=run_phase1.ConditionConfig(
                            "retain_all", True, "retain_all"
                        ),
                        task_ids=["task-1"],
                        port=8080,
                    )
                written_summary = json.loads(
                    (run_dir / "summary.json").read_text(encoding="utf-8")
                )

        self.assertEqual(summary["num_simulations"], 0)
        self.assertEqual(summary["condition"], "retain_all")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(agent_module.get_thinking_records(), [])
        self.assertEqual(written_summary["status"], "complete")

    def test_configure_condition_environment_sets_summarizer_env_vars(self) -> None:
        config = {
            "summarizer": {
                "model": "openrouter/xiaomi/mimo-v2-flash",
                "prompt": (
                    "Customer: {user_message}\n"
                    "Reasoning: {thinking_text}\n"
                    "Response: {response_text}"
                ),
            }
        }
        condition = run_phase1.ConditionConfig(
            name="summary_window3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            run_phase1.configure_condition_environment(condition, config)

            self.assertEqual(os.environ["RETENTION_STRATEGY"], "window_3")
            self.assertEqual(os.environ["SUMMARIZE_THINKING"], "true")
            self.assertEqual(
                os.environ["SUMMARIZER_MODEL"], "openrouter/xiaomi/mimo-v2-flash"
            )
            self.assertEqual(
                os.environ["SUMMARIZER_PROMPT"],
                (
                    "Customer: {user_message}\n"
                    "Reasoning: {thinking_text}\n"
                    "Response: {response_text}"
                ),
            )

    def test_configure_condition_environment_clears_summarizer_env_vars(self) -> None:
        condition = run_phase1.ConditionConfig(
            name="strip_all",
            enable_thinking=True,
            retention_strategy="strip_all",
        )

        with mock.patch.dict(
            os.environ,
            {
                "SUMMARIZE_THINKING": "true",
                "SUMMARIZER_MODEL": "stale-model",
                "SUMMARIZER_PROMPT": "stale-prompt",
            },
            clear=True,
        ):
            run_phase1.configure_condition_environment(condition, {})

            self.assertEqual(os.environ["RETENTION_STRATEGY"], "strip_all")
            self.assertEqual(os.environ["SUMMARIZE_THINKING"], "false")
            self.assertNotIn("SUMMARIZER_MODEL", os.environ)
            self.assertNotIn("SUMMARIZER_PROMPT", os.environ)

    def test_condition_config_summarize_thinking_field(self) -> None:
        condition = run_phase1.ConditionConfig(
            name="summary_window3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )
        self.assertTrue(condition.summarize_thinking)

        condition_off = run_phase1.ConditionConfig(
            name="strip_all",
            enable_thinking=True,
            retention_strategy="strip_all",
        )
        self.assertFalse(condition_off.summarize_thinking)

    def test_apply_smoke_selection_keeps_first_slice(self) -> None:
        models = [
            run_phase1.ModelConfig("repo/a", "a.gguf", "model-a"),
            run_phase1.ModelConfig("repo/b", "b.gguf", "model-b"),
        ]
        conditions = [
            run_phase1.ConditionConfig("thinking_off", False, "strip_all"),
            run_phase1.ConditionConfig("retain_all", True, "retain_all"),
        ]

        smoke_models, smoke_conditions, smoke_tasks = run_phase1.apply_smoke_selection(
            models,
            conditions,
            ["task-1", "task-2"],
        )

        self.assertEqual([model.short_name for model in smoke_models], ["model-a"])
        self.assertEqual(
            [condition.name for condition in smoke_conditions], ["thinking_off"]
        )
        self.assertEqual(smoke_tasks, ["task-1"])

    def test_validate_runtime_environment_requires_openrouter_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llama_server = Path(tmpdir) / "llama-server"
            llama_server.write_text("", encoding="utf-8")
            llama_server.chmod(0o755)

            with mock.patch.object(
                run_phase1, "DEFAULT_LLAMA_SERVER", str(llama_server)
            ):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "OPENROUTER_API_KEY not set",
                    ):
                        run_phase1.validate_runtime_environment()

    def test_resolve_llama_server_bin_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            default_server = Path(tmpdir) / "llama-server"
            default_server.write_text("", encoding="utf-8")
            default_server.chmod(0o755)

            with mock.patch.object(
                run_phase1, "DEFAULT_LLAMA_SERVER", str(default_server)
            ):
                with mock.patch.dict(
                    os.environ,
                    {"LLAMA_SERVER_BIN": str(default_server.parent / "missing")},
                    clear=True,
                ):
                    self.assertEqual(
                        run_phase1.resolve_llama_server_bin(),
                        str(default_server),
                    )

    def test_resolve_llama_server_bin_errors_when_missing_everywhere(self) -> None:
        with mock.patch.object(
            run_phase1, "DEFAULT_LLAMA_SERVER", "/tmp/definitely-missing-llama-server"
        ):
            with mock.patch.dict(
                os.environ,
                {"LLAMA_SERVER_BIN": "/tmp/also-missing-llama-server"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    "executable file",
                ):
                    run_phase1.resolve_llama_server_bin()

    def test_resolve_llama_server_bin_rejects_non_executable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            default_server = Path(tmpdir) / "llama-server"
            default_server.write_text("", encoding="utf-8")

            with mock.patch.object(
                run_phase1, "DEFAULT_LLAMA_SERVER", str(default_server)
            ):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "Set LLAMA_SERVER_BIN to an executable file",
                    ):
                        run_phase1.resolve_llama_server_bin()

    def test_abort_on_thinking_contamination_rewrites_summary_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            analysis_path = run_dir / "thinking_analysis.jsonl"
            analysis_path.write_text(
                json.dumps({"raw_thinking_tokens_approx": 5}) + "\n",
                encoding="utf-8",
            )
            summary = {"condition": "thinking_off", "contaminated": False}

            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "contamination detected"):
                    run_phase1.abort_on_thinking_contamination(run_dir, summary)

            written_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(written_summary["contaminated"])

    def test_abort_on_thinking_contamination_supports_legacy_field_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            analysis_path = run_dir / "thinking_analysis.jsonl"
            analysis_path.write_text(
                json.dumps({"thinking_tokens_approx": 5}) + "\n",
                encoding="utf-8",
            )
            summary = {"condition": "thinking_off", "contaminated": False}

            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "contamination detected"):
                    run_phase1.abort_on_thinking_contamination(run_dir, summary)

    def test_main_skips_completed_conditions_and_runs_missing(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        conditions = [
            run_phase1.ConditionConfig("completed", False, "strip_all"),
            run_phase1.ConditionConfig("missing", True, "retain_all"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            completed_dir = results_root / "model-a_completed_20260101T000000Z"
            self._write_run_fixture(
                completed_dir,
                {"task_ids": ["task-1"], "num_simulations": 1, "status": "complete"},
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=conditions,
                    execute_condition_run=mock.Mock(
                        return_value={"model": model.short_name, "condition": "missing"}
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertEqual(
                execute_condition_run.call_args.kwargs["condition"].name, "missing"
            )
            output = stdout.getvalue()
            self.assertIn("Resuming: skipping 1 completed conditions", output)
            self.assertIn("- model-a / completed", output)
            self.assertIn("Skipping completed model-a / completed", output)

    def test_main_fresh_ignores_completed_conditions(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("completed", False, "strip_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            completed_dir = results_root / "model-a_completed_20260101T000000Z"
            self._write_run_fixture(
                completed_dir,
                {"task_ids": ["task-1"], "num_simulations": 1, "status": "complete"},
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args(fresh=True))

            execute_condition_run.assert_called_once()
            output = stdout.getvalue()
            self.assertIn("Fresh run: ignoring existing results", output)
            self.assertNotIn("Skipping completed", output)

    def test_main_does_not_skip_mismatched_subset_runs(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("completed", False, "strip_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            subset_dir = results_root / "model-a_completed_20260101T000000Z"
            self._write_run_fixture(
                subset_dir,
                {
                    "task_ids": ["task-smoke"],
                    "num_simulations": 1,
                    "status": "complete",
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertIn(
                "Starting fresh: 0 completed conditions found",
                stdout.getvalue(),
            )

    def test_main_does_not_skip_contaminated_runs(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("thinking_off", False, "strip_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            contaminated_dir = results_root / "model-a_thinking_off_20260101T000000Z"
            self._write_run_fixture(
                contaminated_dir,
                {
                    "task_ids": ["task-1"],
                    "num_simulations": 1,
                    "status": "complete",
                    "contaminated": True,
                },
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertIn(
                "Starting fresh: 0 completed conditions found",
                stdout.getvalue(),
            )

    def test_main_does_not_skip_running_runs(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("running", True, "retain_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            running_dir = results_root / "model-a_running_20260101T000000Z"
            self._write_run_fixture(
                running_dir,
                {"task_ids": ["task-1"], "num_simulations": 1, "status": "running"},
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertIn(
                "Starting fresh: 0 completed conditions found",
                stdout.getvalue(),
            )

    def test_main_does_not_skip_failed_runs(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("failed", True, "retain_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            failed_dir = results_root / "model-a_failed_20260101T000000Z"
            self._write_run_fixture(
                failed_dir,
                {"task_ids": ["task-1"], "num_simulations": 1, "status": "failed"},
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertIn(
                "Starting fresh: 0 completed conditions found",
                stdout.getvalue(),
            )

    def test_main_skips_legacy_completed_runs_with_valid_results(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        conditions = [
            run_phase1.ConditionConfig("completed", False, "strip_all"),
            run_phase1.ConditionConfig("missing", True, "retain_all"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            completed_dir = results_root / "model-a_completed_20260101T000000Z"
            self._write_run_fixture(
                completed_dir,
                {"task_ids": ["task-1"], "num_simulations": 1},
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=conditions,
                    execute_condition_run=mock.Mock(
                        return_value={"model": model.short_name, "condition": "missing"}
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertEqual(
                execute_condition_run.call_args.kwargs["condition"].name, "missing"
            )
            self.assertIn(
                "Resuming: skipping 1 completed conditions", stdout.getvalue()
            )

    def test_main_does_not_skip_runs_with_only_short_conversations(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        condition = run_phase1.ConditionConfig("completed", False, "strip_all")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            run_dir = results_root / "model-a_completed_20260101T000000Z"
            self._write_run_fixture(
                run_dir,
                {"task_ids": ["task-1"], "num_simulations": 1, "status": "complete"},
                message_counts=[5],
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self._patch_main_dependencies(
                    results_root=results_root,
                    models=[model],
                    conditions=[condition],
                    execute_condition_run=mock.Mock(
                        return_value={
                            "model": model.short_name,
                            "condition": condition.name,
                        }
                    ),
                ) as execute_condition_run:
                    run_phase1.main(self._main_args())

            execute_condition_run.assert_called_once()
            self.assertIn(
                "Starting fresh: 0 completed conditions found",
                stdout.getvalue(),
            )

    def test_execute_condition_run_with_cleanup_preserves_results_and_marks_failed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "model-a_condition-a_20260101T000000Z"

            def fail_run(**_: object) -> dict[str, object]:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "results.json").write_text("{}", encoding="utf-8")
                raise RuntimeError("boom")

            stdout = StringIO()
            with mock.patch.object(
                run_phase1,
                "execute_condition_run",
                side_effect=fail_run,
            ):
                with redirect_stdout(stdout):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        run_phase1.execute_condition_run_with_cleanup(
                            run_dir=run_dir,
                            experiment={},
                            user_llm="user/model",
                            model=run_phase1.ModelConfig("repo/a", "a.gguf", "model-a"),
                            condition=run_phase1.ConditionConfig(
                                "condition-a", True, "retain_all"
                            ),
                            task_ids=["task-1"],
                            port=8080,
                        )

            self.assertTrue(run_dir.exists())
            written_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_summary["status"], "failed")
            self.assertEqual(written_summary["error"], "boom")
            self.assertIn(f"Preserved failed run: {run_dir.name}", stdout.getvalue())

    def test_cleanup_partial_run_deletes_directory_without_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "model-a_condition-a_20260101T000000Z"
            run_dir.mkdir(parents=True, exist_ok=True)

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_phase1.cleanup_partial_run(run_dir, RuntimeError("boom"))

            self.assertFalse(run_dir.exists())
            self.assertIn(
                f"Cleaned up partial run: {run_dir.name}",
                stdout.getvalue(),
            )

    def test_cleanup_partial_run_deletes_summary_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "model-a_condition-a_20260101T000000Z"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_phase1.cleanup_partial_run(run_dir, RuntimeError("boom"))

            self.assertFalse(run_dir.exists())
            self.assertIn(
                f"Cleaned up partial run: {run_dir.name}",
                stdout.getvalue(),
            )

    def test_exit_if_shutdown_requested_exits_cleanly_between_conditions(self) -> None:
        run_phase1._shutdown_requested = True

        stdout = StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as excinfo:
                run_phase1.exit_if_shutdown_requested(["model-a / condition-a"])

        self.assertEqual(excinfo.exception.code, 0)
        self.assertIn(
            "Graceful shutdown complete. Completed this session:", stdout.getvalue()
        )
        self.assertIn("- model-a / condition-a", stdout.getvalue())

    def test_request_graceful_shutdown_second_signal_forces_keyboard_interrupt(
        self,
    ) -> None:
        stdout = StringIO()
        default_handler = mock.Mock(side_effect=KeyboardInterrupt)

        with mock.patch.object(run_phase1.signal, "signal") as signal_mock:
            with mock.patch.object(
                run_phase1.signal,
                "default_int_handler",
                default_handler,
            ):
                with redirect_stdout(stdout):
                    run_phase1.request_graceful_shutdown(signal.SIGINT, None)
                    with self.assertRaises(KeyboardInterrupt):
                        run_phase1.request_graceful_shutdown(signal.SIGINT, None)

        self.assertTrue(run_phase1._shutdown_requested)
        signal_mock.assert_called_with(signal.SIGINT, default_handler)
        default_handler.assert_called_once_with(signal.SIGINT, None)
        self.assertIn("Graceful shutdown requested.", stdout.getvalue())

    @staticmethod
    def _main_args(**overrides: object) -> argparse.Namespace:
        values = {
            "config": "unused.json",
            "model": [],
            "condition": [],
            "dry_run": False,
            "smoke": False,
            "fresh": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _write_run_fixture(
        run_dir: Path,
        summary: dict[str, object],
        *,
        message_counts: list[int] | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        RunPhase1Test._write_results_fixture(
            run_dir,
            message_counts if message_counts is not None else [6],
        )

    @staticmethod
    def _write_results_fixture(run_dir: Path, message_counts: list[int]) -> None:
        simulations = []
        for index, count in enumerate(message_counts):
            simulations.append(
                {
                    "task_id": f"task-{index}",
                    "messages": [
                        {
                            "role": "assistant" if message_index % 2 else "user",
                            "content": f"message-{index}-{message_index}",
                        }
                        for message_index in range(count)
                    ],
                }
            )
        (run_dir / "results.json").write_text(
            json.dumps({"simulations": simulations}), encoding="utf-8"
        )

    @contextmanager
    def _patch_main_dependencies(
        self,
        *,
        results_root: Path,
        models: list[run_phase1.ModelConfig],
        conditions: list[run_phase1.ConditionConfig],
        execute_condition_run: mock.Mock,
    ):
        config = {
            "experiment": {
                "benchmark": "bench",
                "domain": "telecom",
                "task_split": "test",
                "trials": 1,
            },
            "llama": {"port": 8080},
            "user_sim": {"provider": "openrouter", "model": "mimo"},
        }
        process = mock.Mock()
        process.poll.return_value = None
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.multiple(
                    run_phase1,
                    RESULTS_ROOT=results_root,
                    load_config=mock.Mock(return_value=config),
                    load_models=mock.Mock(return_value=models),
                    load_conditions=mock.Mock(return_value=conditions),
                    load_task_ids=mock.Mock(return_value=["task-1"]),
                    print_plan=mock.Mock(),
                    validate_runtime_environment=mock.Mock(
                        return_value="/tmp/llama-server"
                    ),
                    resolve_model_path=mock.Mock(return_value="/tmp/model.gguf"),
                    build_llama_command=mock.Mock(return_value=["llama-server"]),
                    wait_for_server=mock.Mock(),
                    stop_process=mock.Mock(),
                    configure_condition_environment=mock.Mock(),
                    execute_condition_run=execute_condition_run,
                )
            )
            stack.enter_context(
                mock.patch("scripts.run_phase1.subprocess.Popen", return_value=process)
            )
            stack.enter_context(
                mock.patch(
                    "scripts.run_phase1.signal.getsignal",
                    return_value=signal.default_int_handler,
                )
            )
            stack.enter_context(mock.patch("scripts.run_phase1.signal.signal"))
            yield execute_condition_run


if __name__ == "__main__":
    unittest.main()
