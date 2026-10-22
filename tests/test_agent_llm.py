"""Tests for the LLM-backed agent path using a scripted, offline chat model.

No API keys or network access are needed: the fake model emits one tool call
followed by a final answer, which exercises the LangChain 1.x ``create_agent``
loop (tool execution + final-text extraction) end to end.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ.setdefault("LLM_PROVIDER", "none")

from app.agents.health_agent import HealthAssistantAgent, extract_final_text

TOOL_NAME = "bmi_calculator"
TOOL_ARGS: dict[str, Any] = {"weight_kg": 70, "height_cm": 175}
FINAL_ANSWER = "Your BMI is about 22.9, which is in the normal range."


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that accepts ``bind_tools`` (returns itself)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolModel:  # type: ignore[override]
        return self


def _scripted_model() -> FakeToolModel:
    return FakeToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": TOOL_NAME, "args": TOOL_ARGS, "id": "call_1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content=[{"type": "text", "text": FINAL_ANSWER}]),
            ]
        )
    )


def test_agent_executes_tool_and_returns_final_text() -> None:
    agent = HealthAssistantAgent(llm=_scripted_model())
    assert agent._executor is not None

    result = agent._executor.invoke({"messages": [HumanMessage(content="What is my BMI at 70 kg and 175 cm?")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == TOOL_NAME
    assert "22.9" in str(tool_messages[0].content)
    assert extract_final_text(result) == FINAL_ANSWER


def test_chat_uses_llm_reply() -> None:
    agent = HealthAssistantAgent(llm=_scripted_model())
    reply = agent.chat("What is my BMI at 70 kg and 175 cm?")
    assert reply.startswith(FINAL_ANSWER)


def test_chat_falls_back_to_keyword_router_when_model_fails() -> None:
    # An exhausted script makes the fake model raise on first use.
    agent = HealthAssistantAgent(llm=FakeToolModel(messages=iter([])))
    reply = agent.chat("What is my BMI at 70 kg and 175 cm?")
    assert isinstance(reply, str)
    assert reply.strip()


def test_extract_final_text_edge_cases() -> None:
    assert extract_final_text({}) == ""
    assert extract_final_text({"messages": []}) == ""
    assert extract_final_text({"messages": [HumanMessage(content="hi")]}) == ""
    assert extract_final_text({"messages": [AIMessage(content="  plain  ")]}) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert extract_final_text({"messages": [blocks]}) == "ab"
