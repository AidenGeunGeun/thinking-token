"""Register ThinkingRetentionAgent with tau2-bench's agent factory."""

from __future__ import annotations

from tau2.registry import registry  # type: ignore[import-not-found]

from .agent import ThinkingRetentionAgent


def create_thinking_retention_agent(
    *, tools, domain_policy, llm=None, llm_args=None, **kwargs
):
    if llm is None:
        raise ValueError("thinking_retention requires an agent LLM model")
    return ThinkingRetentionAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
        **kwargs,
    )


def register() -> None:
    if "thinking_retention" in registry.get_agents():
        return
    registry.register_agent_factory(
        create_thinking_retention_agent, "thinking_retention"
    )


register()
