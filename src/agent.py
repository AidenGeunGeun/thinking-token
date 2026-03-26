"""Custom tau2-bench agent with thinking token retention strategies."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from tau2.agent.llm_agent import LLMAgent, LLMAgentState  # type: ignore[import-not-found]
    from tau2.agent.base_agent import ValidAgentInputMessage  # type: ignore[import-not-found]
    from tau2.data_model.message import (  # type: ignore[import-not-found]
        AssistantMessage,
        MultiToolMessage,
        UserMessage,
    )
    from tau2.utils.llm_utils import generate  # type: ignore[import-not-found]

    _TAU2_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without tau2 installed

    class LLMAgent:  # pragma: no cover
        def __init__(self, **kwargs: Any):
            pass

    class LLMAgentState:  # pragma: no cover
        messages: list[Any]
        system_messages: list[Any]

    class AssistantMessage:  # pragma: no cover
        content: str | None
        raw_data: dict[str, Any] | None

    class MultiToolMessage:  # pragma: no cover
        tool_messages: list[Any]

    class UserMessage:  # pragma: no cover
        is_audio: bool = False

    ValidAgentInputMessage = Any

    def generate(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "tau2-bench is required to use ThinkingRetentionAgent"
        ) from exc

    _TAU2_IMPORT_ERROR = exc

from .thinking import (
    apply_retention_strategy,
    extract_thinking,
    replace_thinking_with_summary,
    summarize_thinking,
)

logger = logging.getLogger(__name__)


def _coerce_reasoning_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    return None


def _extract_reasoning(raw_data: dict[str, Any] | None) -> str | None:
    if not isinstance(raw_data, dict):
        return None
    choices = raw_data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        reasoning = _coerce_reasoning_text(message.get(key))
        if reasoning:
            return reasoning
    return None


def _restore_thinking_blocks(message: Any) -> Any:
    content = message.content or ""
    if "<think>" in content:
        return message

    reasoning = _extract_reasoning(message.raw_data)
    if not reasoning:
        return message

    if content:
        message.content = f"<think>{reasoning}</think>\n{content}"
    else:
        message.content = f"<think>{reasoning}</think>"
    return message


class ThinkingRetentionAgent(LLMAgent):  # type: ignore[misc]
    """LLMAgent that applies thinking retention strategies between turns."""

    def __init__(
        self,
        *,
        tools: list[Any],
        domain_policy: str,
        llm: str,
        llm_args: dict[str, Any] | None = None,
        **_: Any,
    ):
        if _TAU2_IMPORT_ERROR is not None:
            raise ImportError(
                "tau2-bench is required to instantiate ThinkingRetentionAgent"
            ) from _TAU2_IMPORT_ERROR
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.retention_strategy = os.environ.get("RETENTION_STRATEGY", "strip_all")
        self.summarize_thinking = (
            os.environ.get("SUMMARIZE_THINKING", "").lower() == "true"
        )
        self.summarizer_model = os.environ.get("SUMMARIZER_MODEL", "")
        self.summarizer_prompt = os.environ.get("SUMMARIZER_PROMPT", "")

    def _maybe_summarize_thinking(self, assistant_message: Any) -> Any:
        """Replace raw <think> blocks with <think_summary> if enabled."""
        if not self.summarize_thinking:
            return assistant_message

        content = getattr(assistant_message, "content", None)
        if not content or "<think>" not in content:
            return assistant_message

        thinking_text, stripped = extract_thinking(content)
        if not thinking_text:
            return assistant_message

        try:
            summary = summarize_thinking(
                thinking_text,
                self.summarizer_model,
                self.summarizer_prompt,
            )
            assistant_message.content = replace_thinking_with_summary(content, summary)
        except Exception:
            logger.warning(
                "Summarizer failed; stripping thinking from this message",
                exc_info=True,
            )
            assistant_message.content = stripped

        return assistant_message

    def _generate_next_message(self, message: Any, state: Any) -> Any:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        prompt_messages = state.system_messages + apply_retention_strategy(
            state.messages,
            self.retention_strategy,
        )
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=prompt_messages,
            call_name="agent_response",
            **self.llm_args,
        )
        assistant_message = _restore_thinking_blocks(assistant_message)
        return self._maybe_summarize_thinking(assistant_message)
