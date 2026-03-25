from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.thinking import (
    apply_retention_strategy,
    extract_thinking,
    identify_turn_boundaries,
    strip_thinking,
)


@dataclass
class MockMessage:
    role: str
    content: str | None = None


class ThinkingUtilsTest(unittest.TestCase):
    def test_strip_thinking_removes_all_blocks(self) -> None:
        content = "<think>draft</think>Visible<think>notes</think> done"
        self.assertEqual(strip_thinking(content), "Visible done")

    def test_extract_thinking_concatenates_blocks(self) -> None:
        content = "Hello<think>first</think> world<think>second</think>!"
        thinking, visible = extract_thinking(content)
        self.assertEqual(thinking, "first\n\nsecond")
        self.assertEqual(visible, "Hello world!")

    def test_identify_turn_boundaries_uses_user_messages(self) -> None:
        messages = [
            MockMessage("system", "policy"),
            MockMessage("user", "u1"),
            MockMessage("assistant", "a1"),
            MockMessage("user", "u2"),
        ]
        self.assertEqual(identify_turn_boundaries(messages), [1, 3])

    def test_strip_all_does_not_mutate_input(self) -> None:
        messages = [
            MockMessage("user", "u1"),
            MockMessage("assistant", "<think>secret</think>reply"),
            MockMessage("tool", "<think>tool</think>raw"),
        ]
        retained = apply_retention_strategy(messages, "strip_all")

        self.assertEqual(messages[1].content, "<think>secret</think>reply")
        self.assertEqual(retained[1].content, "reply")
        self.assertEqual(retained[2].content, "<think>tool</think>raw")
        self.assertIsNot(retained, messages)
        self.assertIsNot(retained[1], messages[1])

    def test_window_strategy_keeps_only_recent_turns(self) -> None:
        messages = [
            MockMessage("user", "u1"),
            MockMessage("assistant", "<think>t1</think>a1"),
            MockMessage("user", "u2"),
            MockMessage("assistant", "<think>t2</think>a2"),
            MockMessage("user", "u3"),
            MockMessage("assistant", "<think>t3</think>a3"),
            MockMessage("user", "u4"),
            MockMessage("assistant", "<think>t4</think>a4"),
        ]

        retained = apply_retention_strategy(messages, "window_3")

        self.assertEqual(retained[1].content, "a1")
        self.assertEqual(retained[3].content, "<think>t2</think>a2")
        self.assertEqual(retained[5].content, "<think>t3</think>a3")
        self.assertEqual(retained[7].content, "<think>t4</think>a4")

    def test_window_strategy_uses_assistant_turns_during_prompting(self) -> None:
        messages = [
            MockMessage("user", "u1"),
            MockMessage("assistant", "<think>t1</think>a1"),
            MockMessage("user", "u2"),
            MockMessage("assistant", "<think>t2</think>a2"),
            MockMessage("user", "u3"),
            MockMessage("assistant", "<think>t3</think>a3"),
            MockMessage("user", "u4"),
        ]

        retained = apply_retention_strategy(messages, "window_3")

        self.assertEqual(retained[1].content, "<think>t1</think>a1")
        self.assertEqual(retained[3].content, "<think>t2</think>a2")
        self.assertEqual(retained[5].content, "<think>t3</think>a3")


if __name__ == "__main__":
    unittest.main()
