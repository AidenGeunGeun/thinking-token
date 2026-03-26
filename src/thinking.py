"""Utilities for extracting and stripping thinking tokens from model outputs."""

from __future__ import annotations

import copy
import re
from typing import Any

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_SUMMARY_PATTERN = re.compile(r"<think_summary>(.*?)</think_summary>", re.DOTALL)
WINDOW_PATTERN = re.compile(r"window_(\d+)")

litellm = None


def strip_thinking(content: str) -> str:
    """Remove all <think>...</think> blocks from content, preserving surrounding text."""
    return THINK_PATTERN.sub("", content)


def strip_think_summary(content: str) -> str:
    """Remove all <think_summary>...</think_summary> blocks from content."""
    return THINK_SUMMARY_PATTERN.sub("", content)


def strip_all_thinking_tags(content: str) -> str:
    """Remove all <think> and <think_summary> blocks from content."""
    return strip_think_summary(strip_thinking(content))


def replace_thinking_with_summary(content: str, summary: str) -> str:
    """Replace <think> blocks with a single summary block."""
    if THINK_PATTERN.search(content) is None:
        return content

    visible_content = strip_thinking(content)
    summary_block = f"<think_summary>{summary}</think_summary>"
    if not visible_content:
        return summary_block
    return f"{summary_block}\n{visible_content}"


def extract_thinking(content: str) -> tuple[str, str]:
    """Return thinking text and content with thinking removed."""
    thinking_blocks = [match.strip() for match in THINK_PATTERN.findall(content)]
    thinking_text = "\n\n".join(block for block in thinking_blocks if block)
    return thinking_text, strip_thinking(content)


def count_thinking_tokens_approx(content: str) -> int:
    """Approximate token count for thinking content."""
    return len(content) // 4


def summarize_thinking(thinking_text: str, model: str, prompt_template: str) -> str:
    """Send raw thinking text to an external model for summarization.

    Args:
        thinking_text: Raw extracted thinking content (no tags).
        model: LiteLLM model string, e.g. "groq/openai/gpt-oss-20b".
        prompt_template: Template with {thinking_text} placeholder.

    Returns:
        Summarized text (no tags).

    Raises:
        Exception: If the litellm call fails (caller should handle fallback).
    """
    global litellm

    if litellm is None:
        import litellm as litellm_module  # type: ignore[import-not-found]

        litellm = litellm_module

    prompt = prompt_template.replace("{thinking_text}", thinking_text)
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _tau2_user_message_type() -> type[Any] | None:
    try:
        from tau2.data_model.message import UserMessage  # type: ignore[import-not-found]

        return UserMessage
    except ImportError:
        return None


def _role_of(message: Any) -> str | None:
    return getattr(message, "role", None)


def _is_user_message(message: Any) -> bool:
    user_message_type = _tau2_user_message_type()
    if user_message_type is not None and isinstance(message, user_message_type):
        return True
    return _role_of(message) == "user" or message.__class__.__name__ == "UserMessage"


def _is_assistant_message(message: Any) -> bool:
    return (
        _role_of(message) == "assistant"
        or message.__class__.__name__ == "AssistantMessage"
    )


def identify_turn_boundaries(messages: list[Any]) -> list[int]:
    """Return message indices where user messages appear."""
    return [
        index for index, message in enumerate(messages) if _is_user_message(message)
    ]


def _parse_window_size(strategy: str) -> int | None:
    match = WINDOW_PATTERN.fullmatch(strategy)
    if match is None:
        return None
    return int(match.group(1))


def apply_retention_strategy(messages: list[Any], strategy: str) -> list[Any]:
    """Apply a retention strategy to a deep copy of a message history."""
    if strategy == "retain_all":
        return copy.deepcopy(messages)

    window_size = _parse_window_size(strategy)
    if strategy != "strip_all" and window_size is None:
        raise ValueError(f"Unsupported retention strategy: {strategy}")

    retained_messages = copy.deepcopy(messages)
    user_indices = identify_turn_boundaries(retained_messages)
    user_index_set = set(user_indices)

    turn_by_index: dict[int, int | None] = {}
    current_turn = -1
    for index, message in enumerate(retained_messages):
        if index in user_index_set:
            current_turn += 1
        turn_by_index[index] = current_turn if current_turn >= 0 else None

    keep_turns: set[int] | None = None
    if window_size is not None:
        assistant_turn_values: list[int] = []
        for index, message in enumerate(retained_messages):
            turn_index = turn_by_index[index]
            if _is_assistant_message(message) and isinstance(turn_index, int):
                assistant_turn_values.append(turn_index)
        assistant_turns = sorted(set(assistant_turn_values))
        keep_turns = set(assistant_turns[-window_size:])

    for index, message in enumerate(retained_messages):
        if not _is_assistant_message(message):
            continue

        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue

        should_strip = strategy == "strip_all"
        if keep_turns is not None:
            turn_index = turn_by_index[index]
            should_strip = turn_index is None or turn_index not in keep_turns

        if should_strip:
            message.content = strip_all_thinking_tags(content)

    return retained_messages
