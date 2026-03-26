from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
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
            name="window_3",
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
            name="window_3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )

        records = run_phase1.build_thinking_records(results, condition)

        self.assertEqual(records[0]["raw_thinking_chars"], 0)
        self.assertEqual(records[0]["summary_chars"], len("condensed note"))
        self.assertIsNone(records[0]["summarizer_input_tokens"])
        self.assertIsNone(records[0]["summarizer_output_tokens"])

    def test_configure_condition_environment_sets_summarizer_env_vars(self) -> None:
        config = {
            "summarizer": {
                "model": "groq/openai/gpt-oss-20b",
                "prompt": "Distill: {thinking_text}",
            }
        }
        condition = run_phase1.ConditionConfig(
            name="window_3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=True,
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            run_phase1.configure_condition_environment(condition, config)

            self.assertEqual(os.environ["RETENTION_STRATEGY"], "window_3")
            self.assertEqual(os.environ["SUMMARIZE_THINKING"], "true")
            self.assertEqual(os.environ["SUMMARIZER_MODEL"], "groq/openai/gpt-oss-20b")
            self.assertEqual(
                os.environ["SUMMARIZER_PROMPT"], "Distill: {thinking_text}"
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
            name="window_3",
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

    def test_validate_runtime_environment_requires_groq_api_key(self) -> None:
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
                        "GROQ_API_KEY not set",
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


if __name__ == "__main__":
    unittest.main()
