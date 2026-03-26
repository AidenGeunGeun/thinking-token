"""Tests for ThinkingRetentionAgent summarization integration."""

from __future__ import annotations

import importlib
import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
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
            "SUMMARIZER_MODEL": "openrouter/xiaomi/mimo-v2-flash",
            "SUMMARIZER_PROMPT": (
                "Customer: {user_message}\n"
                "Reasoning: {thinking_text}\n"
                "Response: {response_text}"
            ),
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
        agent.tools = []
        agent.llm = "openai/test-model"
        agent.llm_args = {}
        return agent

    def _make_state(
        self,
        messages: list[Any] | None = None,
        system_messages: list[Any] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            messages=list(messages or []),
            system_messages=list(system_messages or []),
            tools=[],
        )

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

    def test_summarize_passes_last_user_message_and_visible_response(self):
        agent = self._make_agent()
        agent._internal_messages = [
            MockMessage(role="user", content="Can you check why I was charged twice?")
        ]
        msg = MockMessage(
            role="assistant",
            content="<think>Investigate duplicate charge</think>I found the duplicate charge.",
        )

        with patch(
            "src.agent.summarize_thinking", return_value="Short summary"
        ) as mock_sum:
            agent._maybe_summarize_thinking(msg)

        mock_sum.assert_called_once_with(
            "Investigate duplicate charge",
            "openrouter/xiaomi/mimo-v2-flash",
            "Customer: {user_message}\nReasoning: {thinking_text}\nResponse: {response_text}",
            user_message="Can you check why I was charged twice?",
            response_text="I found the duplicate charge.",
        )

    def test_summarize_stringifies_flattened_tool_turn_context(self):
        agent = self._make_agent()
        agent._internal_messages = [
            MockMessage(role="assistant", content="Prior answer"),
            SimpleNamespace(role="tool", content={"account_id": "12345"}),
            SimpleNamespace(role="tool", content="Plan status: active"),
        ]
        msg = MockMessage(
            role="assistant",
            content="<think>Review tool output</think>Your account is active.",
        )

        with patch(
            "src.agent.summarize_thinking", return_value="Short summary"
        ) as mock_sum:
            agent._maybe_summarize_thinking(msg)

        mock_sum.assert_called_once_with(
            "Review tool output",
            "openrouter/xiaomi/mimo-v2-flash",
            "Customer: {user_message}\nReasoning: {thinking_text}\nResponse: {response_text}",
            user_message='{"account_id": "12345"}\n\nPlan status: active',
            response_text="Your account is active.",
        )

    def test_summarize_stringifies_multitool_context(self):
        agent = self._make_agent()
        agent_module = importlib.import_module("src.agent")
        multi_tool_message = agent_module.MultiToolMessage()
        multi_tool_message.tool_messages = [
            SimpleNamespace(role="tool", content={"account_id": "12345"}),
            SimpleNamespace(role="tool", content="Plan status: active"),
        ]
        agent._internal_messages = [multi_tool_message]
        msg = MockMessage(
            role="assistant",
            content="<think>Review tool output</think>Your account is active.",
        )

        with patch(
            "src.agent.summarize_thinking", return_value="Short summary"
        ) as mock_sum:
            agent._maybe_summarize_thinking(msg)

        mock_sum.assert_called_once_with(
            "Review tool output",
            "openrouter/xiaomi/mimo-v2-flash",
            "Customer: {user_message}\nReasoning: {thinking_text}\nResponse: {response_text}",
            user_message='{"account_id": "12345"}\n\nPlan status: active',
            response_text="Your account is active.",
        )

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

    def test_tool_call_message_skips_summarization(self):
        agent = self._make_agent()
        msg = MockMessage(
            role="assistant",
            content="<think>reasoning about tool</think>",
            tool_calls=[{"name": "get_customer", "arguments": "{}"}],
        )
        result = agent._maybe_summarize_thinking(msg)
        # Should be unchanged — tool-call messages are never summarized
        self.assertEqual(result.content, "<think>reasoning about tool</think>")

    def test_restore_thinking_blocks_skips_tool_calls(self):
        """_restore_thinking_blocks should not inject thinking into tool-call messages."""
        import importlib

        agent_module = importlib.import_module("src.agent")
        msg = MockMessage(
            role="assistant",
            content=None,
            tool_calls=[{"name": "get_customer", "arguments": "{}"}],
            raw_data={
                "choices": [{"message": {"reasoning": "deep thoughts about tools"}}]
            },
        )
        result = agent_module._restore_thinking_blocks(msg)
        # Content should remain None — don't inject thinking into tool-call messages
        self.assertIsNone(result.content)

    def test_generate_next_message_internal_history_keeps_summary(self):
        agent = self._make_agent()
        state = self._make_state(
            system_messages=[MockMessage(role="system", content="policy")]
        )
        incoming = MockMessage(role="user", content="What is 6 * 7?")
        generated = MockMessage(
            role="assistant",
            content="<think>long reasoning</think>The answer is 42",
        )

        with (
            patch("src.agent.generate", return_value=generated),
            patch("src.agent.summarize_thinking", return_value="Short summary"),
        ):
            returned = agent._generate_next_message(incoming, state)

        self.assertEqual(len(agent._internal_messages), 2)
        self.assertIs(agent._internal_messages[0], incoming)
        self.assertEqual(
            agent._internal_messages[1].content,
            "<think_summary>Short summary</think_summary>\nThe answer is 42",
        )
        self.assertIsNot(agent._internal_messages[1], returned)

    def test_generate_next_message_seeds_internal_history_from_existing_state(self):
        agent = self._make_agent({"SUMMARIZE_THINKING": "false"})
        prior_assistant = MockMessage(role="assistant", content="Hi! How can I help?")
        state = self._make_state(
            messages=[prior_assistant],
            system_messages=[MockMessage(role="system", content="policy")],
        )
        incoming = MockMessage(role="user", content="I need help")
        generated = MockMessage(role="assistant", content="Sure")

        with patch("src.agent.generate", return_value=generated) as mock_generate:
            returned = agent._generate_next_message(incoming, state)

        prompt_messages = mock_generate.call_args.kwargs["messages"]

        self.assertEqual(
            [msg.content for msg in prompt_messages],
            ["policy", "Hi! How can I help?", "I need help"],
        )
        self.assertIs(agent._internal_messages[0], prior_assistant)
        self.assertIs(agent._internal_messages[1], incoming)
        self.assertEqual(returned.content, "Sure")

    def test_generate_next_message_returns_message_without_any_thinking_tags(self):
        agent = self._make_agent({"SUMMARIZE_THINKING": "false"})
        state = self._make_state()
        incoming = MockMessage(role="user", content="Hello")
        generated = MockMessage(
            role="assistant",
            content="<think>reasoning</think>Visible<think_summary>note</think_summary>",
        )

        with patch("src.agent.generate", return_value=generated):
            returned = agent._generate_next_message(incoming, state)

        self.assertEqual(returned.content, "Visible")
        self.assertNotIn("<think>", returned.content)
        self.assertNotIn("<think_summary>", returned.content)

    def test_generate_next_message_strips_thinking_from_tool_call_content(self):
        agent = self._make_agent({"SUMMARIZE_THINKING": "false"})
        state = self._make_state()
        incoming = MockMessage(role="user", content="Find the account")
        generated = MockMessage(
            role="assistant",
            content="<think>lookup plan</think>Calling tool",
            tool_calls=[{"name": "get_customer", "arguments": "{}"}],
        )

        with patch("src.agent.generate", return_value=generated):
            returned = agent._generate_next_message(incoming, state)

        self.assertEqual(
            agent._internal_messages[1].content,
            "<think>lookup plan</think>Calling tool",
        )
        self.assertEqual(returned.content, "Calling tool")
        self.assertEqual(returned.tool_calls, generated.tool_calls)

    def test_generate_next_message_strip_all_keeps_prompt_and_returned_views_clean(
        self,
    ):
        agent = self._make_agent(
            {"RETENTION_STRATEGY": "strip_all", "SUMMARIZE_THINKING": "false"}
        )
        state = self._make_state(
            system_messages=[MockMessage(role="system", content="policy")]
        )
        first_user = MockMessage(role="user", content="First question")
        second_user = MockMessage(role="user", content="Second question")
        first_generated = MockMessage(
            role="assistant",
            content="<think>private reasoning</think>First answer",
        )
        second_generated = MockMessage(role="assistant", content="Second answer")

        with patch(
            "src.agent.generate", side_effect=[first_generated, second_generated]
        ) as mock_generate:
            first_returned = agent._generate_next_message(first_user, state)
            state.messages.append(first_returned)
            second_returned = agent._generate_next_message(second_user, state)

        second_prompt_messages = mock_generate.call_args_list[1].kwargs["messages"]
        assistant_messages = [
            msg
            for msg in second_prompt_messages
            if getattr(msg, "role", None) == "assistant"
        ]

        self.assertEqual(first_returned.content, "First answer")
        self.assertEqual(
            agent._internal_messages[1].content,
            "<think>private reasoning</think>First answer",
        )
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0].content, "First answer")
        self.assertEqual(second_returned.content, "Second answer")

    def test_generate_next_message_retain_all_keeps_raw_thinking_in_internal_prompt_view(
        self,
    ):
        agent = self._make_agent(
            {"RETENTION_STRATEGY": "retain_all", "SUMMARIZE_THINKING": "false"}
        )
        state = self._make_state(
            system_messages=[MockMessage(role="system", content="policy")]
        )
        first_user = MockMessage(role="user", content="First question")
        second_user = MockMessage(role="user", content="Second question")
        first_generated = MockMessage(
            role="assistant",
            content="<think>private reasoning</think>First answer",
        )
        second_generated = MockMessage(role="assistant", content="Second answer")

        with patch(
            "src.agent.generate", side_effect=[first_generated, second_generated]
        ) as mock_generate:
            first_returned = agent._generate_next_message(first_user, state)
            state.messages.append(first_returned)
            agent._generate_next_message(second_user, state)

        second_prompt_messages = mock_generate.call_args_list[1].kwargs["messages"]
        assistant_messages = [
            msg
            for msg in second_prompt_messages
            if getattr(msg, "role", None) == "assistant"
        ]

        self.assertEqual(
            agent._internal_messages[1].content,
            "<think>private reasoning</think>First answer",
        )
        self.assertEqual(first_returned.content, "First answer")
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(
            assistant_messages[0].content,
            "<think>private reasoning</think>First answer",
        )

    def test_generate_next_message_preserves_internal_history_across_state_swap(self):
        agent = self._make_agent(
            {"RETENTION_STRATEGY": "retain_all", "SUMMARIZE_THINKING": "false"}
        )
        system_messages = [MockMessage(role="system", content="policy")]
        first_state = self._make_state(system_messages=system_messages)
        first_user = MockMessage(role="user", content="First question")
        second_user = MockMessage(role="user", content="Second question")
        first_generated = MockMessage(
            role="assistant",
            content="<think>private reasoning</think>First answer",
        )
        second_generated = MockMessage(role="assistant", content="Second answer")

        with patch(
            "src.agent.generate", side_effect=[first_generated, second_generated]
        ) as mock_generate:
            first_returned = agent._generate_next_message(first_user, first_state)
            first_state.messages.append(first_returned)
            second_state = self._make_state(
                messages=list(first_state.messages),
                system_messages=system_messages,
            )
            agent._generate_next_message(second_user, second_state)

        second_prompt_messages = mock_generate.call_args_list[1].kwargs["messages"]
        assistant_messages = [
            msg
            for msg in second_prompt_messages
            if getattr(msg, "role", None) == "assistant"
        ]

        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(
            assistant_messages[0].content,
            "<think>private reasoning</think>First answer",
        )


if __name__ == "__main__":
    unittest.main()
