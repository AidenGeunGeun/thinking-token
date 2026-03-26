"""Tests for ThinkingRetentionAgent summarization integration."""

from __future__ import annotations

import importlib
import os
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch


@dataclass
class MockMessage:
    role: str
    content: str | None = None
    tool_calls: list[Any] | None = None
    raw_data: dict[str, Any] | None = None
    is_audio: bool = False


class AgentSummarizationTest(unittest.TestCase):
    """Test _maybe_summarize_thinking method."""

    def _make_agent(self, env_overrides: dict[str, str] | None = None):
        """Create a ThinkingRetentionAgent with mocked tau2 dependencies."""
        env = {
            "RETENTION_STRATEGY": "retain_all",
            "SUMMARIZE_THINKING": "true",
            "SUMMARIZER_MODEL": "groq/openai/gpt-oss-20b",
            "SUMMARIZER_PROMPT": "Distill: {thinking_text}",
        }
        if env_overrides:
            env.update(env_overrides)

        with patch.dict(os.environ, env, clear=False):
            agent_module = importlib.import_module("src.agent")
            with patch.object(agent_module, "_TAU2_IMPORT_ERROR", None):
                agent = agent_module.ThinkingRetentionAgent(
                    tools=[],
                    domain_policy="test policy",
                    llm="openai/test-model",
                    llm_args={},
                )
        return agent

    def test_summarize_replaces_thinking(self):
        agent = self._make_agent()
        msg = MockMessage(
            role="assistant",
            content="<think>long reasoning</think>The answer is 42",
        )

        with patch("src.agent.summarize_thinking", return_value="Short summary"):
            result = agent._maybe_summarize_thinking(msg)

        self.assertIn("<think_summary>Short summary</think_summary>", result.content)
        self.assertNotIn("<think>", result.content)
        self.assertIn("The answer is 42", result.content)

    def test_summarize_disabled_returns_unchanged(self):
        agent = self._make_agent({"SUMMARIZE_THINKING": "false"})
        msg = MockMessage(
            role="assistant",
            content="<think>reasoning</think>response",
        )
        result = agent._maybe_summarize_thinking(msg)
        self.assertEqual(result.content, "<think>reasoning</think>response")

    def test_no_thinking_returns_unchanged(self):
        agent = self._make_agent()
        msg = MockMessage(role="assistant", content="just a reply")
        result = agent._maybe_summarize_thinking(msg)
        self.assertEqual(result.content, "just a reply")

    def test_summarizer_failure_strips_thinking(self):
        agent = self._make_agent()
        msg = MockMessage(
            role="assistant",
            content="<think>reasoning</think>The answer",
        )

        with patch(
            "src.agent.summarize_thinking", side_effect=RuntimeError("API down")
        ):
            result = agent._maybe_summarize_thinking(msg)

        self.assertNotIn("<think>", result.content)
        self.assertNotIn("<think_summary>", result.content)
        self.assertIn("The answer", result.content)

    def test_empty_thinking_returns_unchanged(self):
        agent = self._make_agent()
        msg = MockMessage(
            role="assistant",
            content="<think></think>The answer",
        )
        result = agent._maybe_summarize_thinking(msg)
        self.assertEqual(result.content, "<think></think>The answer")

    def test_none_content_returns_unchanged(self):
        agent = self._make_agent()
        msg = MockMessage(role="assistant", content=None)
        result = agent._maybe_summarize_thinking(msg)
        self.assertIsNone(result.content)


if __name__ == "__main__":
    unittest.main()
