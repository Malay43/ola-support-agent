"""Autogen review stage after the CrewAI draft (Task 14).

2-agent RoundRobinGroupChat (Policy-Compliance-Reviewer -> Final-Editor), bounded with max_turns=2.
The Final-Editor emits a Pydantic ReviewVerdict (output_content_type=...), which requires the team
to register custom_message_types=[StructuredMessage[ReviewVerdict]].
Under MOCK_LLM the model behind both agents is a deterministic rule-based client (no key, no network).
"""
import asyncio
import json
import re
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, Mapping, Optional

import config
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (ChatCompletionClient, CreateResult, LLMMessage, ModelInfo,
                                 RequestUsage)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

import governance
from guardrails import unsupported_sentences
from mock_llm import FALLBACK, split_sentences
from schemas import ReviewVerdict

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"
REVIEWER_SYS = ("You are the Policy-Compliance-Reviewer. Check every sentence of the DRAFT ANSWER against the "
                "RETRIEVED CONTEXT and list any claim the context does not support.")
EDITOR_SYS = ("You are the Final-Editor. Using the reviewer's findings, either approve the draft unchanged or "
              "remove the unsupported claims, and return a structured verdict (approved, final_answer, reason).")


def _text(m: LLMMessage) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _parse_task(messages: Sequence[LLMMessage]) -> tuple[str, str]:
    full = "\n".join(_text(m) for m in messages if m.type == "UserMessage")
    d = re.search(r"DRAFT ANSWER:\n(.*?)\nRETRIEVED CONTEXT:\n(.*)", full, re.S)
    if not d:
        return "", ""
    ctx = d.group(2).split("\nREVIEW:", 1)[0]
    ctx = re.split(r"\n(?=REVIEW: )", ctx)[0]
    return d.group(1).strip(), ctx.strip()


class MockReviewClient(ChatCompletionClient):
    """Deterministic stand-in for an LLM: rule-based reviewer + editor."""

    def __init__(self):
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.calls = 0

    # ---- logic
    def _respond(self, messages: Sequence[LLMMessage], json_output) -> str:
        system = next((_text(m) for m in messages if m.type == "SystemMessage"), "")
        draft, context = _parse_task(messages)
        bad = unsupported_sentences(draft, context) if draft else []
        if "Policy-Compliance-Reviewer" in system:
            if bad:
                return "REVIEW: REVISE\nUNSUPPORTED: " + " || ".join(bad)
            return "REVIEW: APPROVE\nEvery sentence is supported by the retrieved context."
        # Final-Editor
        convo = "\n".join(_text(m) for m in messages if m.type == "UserMessage")
        revise = "REVIEW: REVISE" in convo
        if revise and bad:
            kept = [s for s in split_sentences(draft) if s not in bad]
            final = " ".join(kept) if kept else FALLBACK
            verdict = ReviewVerdict(approved=False, final_answer=final,
                                    reason=f"Removed {len(bad)} unsupported claim(s): " + " | ".join(bad))
        else:
            verdict = ReviewVerdict(approved=True, final_answer=draft,
                                    reason="All claims are supported by the retrieved context; draft approved unchanged.")
        return verdict.model_dump_json()

    # ---- ChatCompletionClient interface
    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = [],
                     tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
                     json_output: Optional[bool | type[BaseModel]] = None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        self.calls += 1
        out = self._respond(messages, json_output)
        p = governance.estimate_tokens(" ".join(_text(m) for m in messages))
        c = governance.estimate_tokens(out)
        meter = governance.current_meter()
        if meter is not None:
            meter.add(p + c)
        self._usage = RequestUsage(prompt_tokens=self._usage.prompt_tokens + p,
                                   completion_tokens=self._usage.completion_tokens + c)
        return CreateResult(finish_reason="stop", content=out, usage=RequestUsage(prompt_tokens=p, completion_tokens=c),
                            cached=False)

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                            extra_create_args={}, cancellation_token=None) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                                extra_create_args=extra_create_args, cancellation_token=cancellation_token)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return governance.estimate_tokens(" ".join(_text(m) for m in messages))

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 100_000

    @property
    def capabilities(self):  # type: ignore[override]
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown",
                         structured_output=True)


def _make_client() -> ChatCompletionClient:
    if config.MOCK_LLM:
        return MockReviewClient()
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # optional: pip install autogen-ext[openai]
    import os

    return OpenAIChatCompletionClient(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))


def build_team() -> RoundRobinGroupChat:
    client = _make_client()
    reviewer = AssistantAgent(REVIEWER_NAME, model_client=client, system_message=REVIEWER_SYS)
    editor = AssistantAgent(EDITOR_NAME, model_client=client, system_message=EDITOR_SYS,
                            output_content_type=ReviewVerdict)
    return RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,  # real constructor parameter (NOT max_iterations): each agent speaks once
        custom_message_types=[StructuredMessage[ReviewVerdict]],  # required for the editor's structured output
    )


async def _review_async(draft: str, context: str) -> tuple[ReviewVerdict, list[dict]]:
    team = build_team()
    task = f"DRAFT ANSWER:\n{draft}\nRETRIEVED CONTEXT:\n{context}"
    result = await team.run(task=task)
    transcript = [{"source": m.source, "type": type(m).__name__, "content": _text(m)[:400]} for m in result.messages]
    last = result.messages[-1]
    verdict = last.content if isinstance(last, StructuredMessage) else ReviewVerdict.model_validate_json(_text(last))
    return verdict, transcript


def review_answer(draft: str, context: str) -> tuple[ReviewVerdict, list[dict]]:
    """Synchronous wrapper (call from a worker thread when inside an event loop)."""
    return asyncio.run(_review_async(draft, context))
