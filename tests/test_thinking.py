from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.thinking import (
    apply_retention_strategy,
    extract_thinking,
    identify_turn_boundaries,
    replace_thinking_with_summary,
    strip_all_thinking_tags,
    strip_think_summary,
    strip_thinking,
    summarize_thinking,
)


@dataclass
class MockMessage:
    role: str
    content: str | None = None


class ThinkingUtilsTest(unittest.TestCase):
    def test_strip_thinking_removes_all_blocks(self) -> None:
        content = "<think>draft</think>Visible<think>notes</think> done"
        self.assertEqual(strip_thinking(content), "Visible done")

    def test_strip_think_summary_removes_summary_blocks(self) -> None:
        content = "<think_summary>short note</think_summary>Visible text"
        self.assertEqual(strip_think_summary(content), "Visible text")

    def test_strip_all_thinking_tags_removes_both(self) -> None:
        content = "<think>raw</think>Middle<think_summary>summary</think_summary>End"
        self.assertEqual(strip_all_thinking_tags(content), "MiddleEnd")

    def test_extract_thinking_concatenates_blocks(self) -> None:
        content = "Hello<think>first</think> world<think>second</think>!"
        thinking, visible = extract_thinking(content)
        self.assertEqual(thinking, "first\n\nsecond")
        self.assertEqual(visible, "Hello world!")

    def test_replace_thinking_with_summary(self) -> None:
        content = "<think>long reasoning</think>The answer is 42"
        result = replace_thinking_with_summary(content, "Reasoned about math")

        self.assertIn("<think_summary>Reasoned about math</think_summary>", result)
        self.assertIn("The answer is 42", result)
        self.assertNotIn("<think>", result)

    def test_replace_thinking_with_summary_no_thinking(self) -> None:
        content = "Just plain text"
        self.assertEqual(replace_thinking_with_summary(content, "summary"), content)

    @patch("src.thinking.litellm")
    def test_summarize_thinking_calls_litellm(self, mock_litellm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  Summary result  "
        mock_litellm.completion.return_value = mock_response

        result = summarize_thinking(
            "long reasoning text",
            "groq/openai/gpt-oss-20b",
            "Distill: {thinking_text}",
        )

        self.assertEqual(result, "Summary result")
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args
        self.assertIn(
            "long reasoning text", call_kwargs.kwargs["messages"][0]["content"]
        )

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

    def test_apply_retention_strategy_strips_summaries_from_old_turns(self) -> None:
        messages = [
            MockMessage("user", "u1"),
            MockMessage("assistant", "<think_summary>note1</think_summary>a1"),
            MockMessage("user", "u2"),
            MockMessage("assistant", "<think_summary>note2</think_summary>a2"),
        ]

        retained = apply_retention_strategy(messages, "window_1")

        self.assertEqual(retained[1].content, "a1")
        self.assertEqual(retained[3].content, "<think_summary>note2</think_summary>a2")


if __name__ == "__main__":
    unittest.main()
