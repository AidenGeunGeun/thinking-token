from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from scripts import run_phase1, verify_pipeline


@dataclass
class MockMessage:
    role: str
    content: str | None = None
    raw_data: dict[str, Any] | None = None


class VerifyPipelineTest(unittest.TestCase):
    def test_strip_all_uses_raw_data_to_confirm_thinking_generated(self) -> None:
        condition = run_phase1.ConditionConfig(
            name="strip_all",
            enable_thinking=True,
            retention_strategy="strip_all",
            summarize_thinking=False,
        )
        public_messages = [
            MockMessage(
                role="assistant",
                content="Visible answer only",
                raw_data={
                    "choices": [{"message": {"reasoning": "private chain of thought"}}]
                },
            )
        ]
        internal_messages = [
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Visible answer only"},
        ]

        report = verify_pipeline.evaluate_invariants(
            condition,
            public_messages,
            internal_messages,
        )

        self.assertEqual(report.invariants["INV-1"].status, "PASS")
        self.assertEqual(report.invariants["INV-2"].status, "PASS")
        self.assertEqual(report.invariants["INV-5"].status, "PASS")
        self.assertEqual(report.invariants["INV-7"].status, "PASS")

    def test_summary_quality_rejects_headers_and_bullets(self) -> None:
        condition = run_phase1.ConditionConfig(
            name="summary_retain",
            enable_thinking=True,
            retention_strategy="retain_all",
            summarize_thinking=True,
        )
        internal_messages = [
            {
                "role": "assistant",
                "content": "<think_summary>## Plan\n- first step</think_summary>Visible answer",
            }
        ]
        public_messages = [MockMessage(role="assistant", content="Visible answer")]

        report = verify_pipeline.evaluate_invariants(
            condition,
            public_messages,
            internal_messages,
        )

        self.assertEqual(report.invariants["INV-4"].status, "PASS")
        self.assertEqual(report.invariants["INV-6"].status, "FAIL")
        self.assertIn("markdown headers", report.invariants["INV-6"].evidence)
        self.assertIn("bullet points", report.invariants["INV-6"].evidence)

    def test_raw_condition_skips_internal_tag_check_when_no_thinking_generated(
        self,
    ) -> None:
        condition = run_phase1.ConditionConfig(
            name="raw_window3",
            enable_thinking=True,
            retention_strategy="window_3",
            summarize_thinking=False,
        )
        public_messages = [MockMessage(role="assistant", content="Visible answer")]
        internal_messages = [{"role": "assistant", "content": "Visible answer"}]

        report = verify_pipeline.evaluate_invariants(
            condition,
            public_messages,
            internal_messages,
        )

        self.assertEqual(report.invariants["INV-3"].status, "skip")
        self.assertEqual(report.invariants["INV-7"].status, "FAIL")

    def test_strip_all_uses_snapshot_meta_when_internal_view_is_clean(self) -> None:
        condition = run_phase1.ConditionConfig(
            name="strip_all",
            enable_thinking=True,
            retention_strategy="strip_all",
            summarize_thinking=False,
        )
        public_messages = [MockMessage(role="assistant", content="Visible answer")]
        internal_messages = [{"role": "assistant", "content": "Visible answer"}]
        snapshot_meta = [
            {
                "content": "<think>hidden reasoning</think>Visible answer",
                "has_think": True,
                "has_think_summary": False,
                "raw_thinking_chars": len("hidden reasoning"),
            }
        ]

        report = verify_pipeline.evaluate_invariants(
            condition,
            public_messages,
            internal_messages,
            snapshot_meta,
        )

        self.assertEqual(report.invariants["INV-5"].status, "PASS")
        self.assertEqual(report.invariants["INV-7"].status, "PASS")


if __name__ == "__main__":
    unittest.main()
