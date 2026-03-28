from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from scripts import run_phase1, view_results


class ViewResultsTest(unittest.TestCase):
    def test_main_counts_only_completed_runs(self) -> None:
        model = run_phase1.ModelConfig("repo/a", "a.gguf", "model-a")
        conditions = [
            run_phase1.ConditionConfig("complete", True, "retain_all"),
            run_phase1.ConditionConfig("running", True, "retain_all"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            self._write_run_fixture(
                results_root / "model-a_complete_20260101T000000Z",
                status="complete",
            )
            self._write_run_fixture(
                results_root / "model-a_running_20260101T000000Z",
                status="running",
            )
            self._write_run_fixture(
                results_root / "model-a_failed_20260101T000000Z",
                status="failed",
            )
            self._write_run_fixture(
                results_root / "model-a_other_20260101T000000Z",
                status="complete",
                condition="other",
            )
            self._write_run_fixture(
                results_root / "model-b_running_20260101T000000Z",
                status="running",
                model="model-b",
                condition="running",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                with mock.patch.multiple(
                    view_results,
                    load_config=mock.Mock(
                        return_value={"experiment": {"name": "demo"}}
                    ),
                    resolve_results_root=mock.Mock(return_value=results_root),
                    load_models=mock.Mock(return_value=[model]),
                    load_conditions=mock.Mock(return_value=conditions),
                ):
                    view_results.main(["--config", "unused.json"])

        output = stdout.getvalue()
        self.assertIn("1/2 configurations completed", output)
        self.assertIn("1 run(s) in progress", output)
        self.assertNotIn("2/2 configurations completed", output)

    @staticmethod
    def _write_run_fixture(
        run_dir: Path,
        *,
        status: str,
        model: str = "model-a",
        condition: str | None = None,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if condition is None:
            condition = run_dir.name.split("_")[1]
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "condition": condition,
                    "status": status,
                    "full_reward_count": 1,
                    "num_simulations": 1,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "task_id": "task-1",
                            "duration": 1,
                            "messages": [
                                {"role": "user", "content": "u"},
                                {"role": "assistant", "content": "a"},
                                {"role": "user", "content": "u2"},
                                {"role": "assistant", "content": "a2"},
                                {"role": "user", "content": "u3"},
                                {"role": "assistant", "content": "a3"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
