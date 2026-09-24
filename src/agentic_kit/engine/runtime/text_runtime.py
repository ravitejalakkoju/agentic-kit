"""The agent loop: prompt, model, tools, reply.

The loop is bounded on purpose. A model that keeps reaching for tools is not
making progress, so after MAX_TOOL_ROUNDS it is asked once more with none.

The prompt and output checkpoints live here because this is the only place the
assembled prompt and the drafted reply exist.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...domain.models import ConversationState, Message, Role, SopDefinition, TurnRequest
from ...domain.tools import ToolCall
from ...errors import ProviderError
from ...ports.llm import LlmPort, LlmReply, LlmRequest, ToolExchange
from ..guardrails import Checkpoint, Finding, GuardrailResponder, Guardrails
from ..prompt.builder import PromptBuilder
from ..tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 3
GAVE_UP_REPLY = "I could not finish looking that up. Let me get a teammate to help."


@dataclass(frozen=True, slots=True)
class RuntimeReply:
    text: str
    blocked_by: Finding | None = None
    """Set when a guardrail replaced the model's answer with a safe one."""
    tool_calls: tuple[ToolCall, ...] = ()
    """Every tool the model used this turn, in the order it asked for them."""


class TextRuntime:
    def __init__(
        self,
        llm: LlmPort,
        prompts: PromptBuilder,
        guardrails: Guardrails,
        responder: GuardrailResponder,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._guardrails = guardrails
        self._responder = responder
        self._tools = tools or ToolRegistry()

    async def run(
        self, *, sop: SopDefinition, conversation: ConversationState, request: TurnRequest
    ) -> RuntimeReply:
        offered = self._tools.definitions(request)
        prompt = await self._prompts.build(sop=sop, request=request, tools=offered)

        blocked = self._check(Checkpoint.PROMPT, request, prompt.system)
        if blocked is not None:
            return blocked

        messages = (*conversation.history, Message(role=Role.USER, text=request.text))
        reply, used = await self._converse(
            LlmRequest(system=prompt.system, messages=messages, tools=offered), request
        )

        if not reply.text.strip():
            raise ProviderError("model returned an empty reply")

        checked = self._check(Checkpoint.OUTPUT, request, reply.text)
        return checked or RuntimeReply(text=reply.text, tool_calls=used)

    async def _converse(
        self, ask: LlmRequest, request: TurnRequest
    ) -> tuple[LlmReply, tuple[ToolCall, ...]]:
        """Trade tool calls with the model until it answers or runs out of rounds."""
        exchanges: list[ToolExchange] = []
        used: list[ToolCall] = []

        for _ in range(MAX_TOOL_ROUNDS):
            reply = await self._llm.complete(
                LlmRequest(
                    system=ask.system,
                    messages=ask.messages,
                    tools=ask.tools,
                    exchanges=tuple(exchanges),
                )
            )
            if not reply.wants_tools:
                return reply, tuple(used)

            used.extend(reply.tool_calls)
            exchanges.append(await self._run_tools(reply.tool_calls, request))

        return await self._final_answer(ask, exchanges), tuple(used)

    async def _run_tools(self, calls: tuple[ToolCall, ...], request: TurnRequest) -> ToolExchange:
        results = await asyncio.gather(
            *(self._tools.execute(call, request) for call in calls),
        )
        return ToolExchange(
            calls=calls,
            results=tuple(
                (call.id, result.for_model()) for call, result in zip(calls, results, strict=True)
            ),
        )

    async def _final_answer(self, ask: LlmRequest, exchanges: list[ToolExchange]) -> LlmReply:
        """One last ask with no tools, so the model has to answer with what it has."""
        reply = await self._llm.complete(
            LlmRequest(
                system=ask.system,
                messages=ask.messages,
                exchanges=tuple(exchanges),
            )
        )
        return reply if reply.text.strip() else LlmReply(text=GAVE_UP_REPLY)

    def _check(
        self, checkpoint: Checkpoint, request: TurnRequest, text: str
    ) -> RuntimeReply | None:
        verdict = self._guardrails.check(checkpoint, request, text)
        failure = verdict.failure
        if verdict.passed or failure is None:
            return None
        return RuntimeReply(text=self._responder.reply_for(failure), blocked_by=failure)
