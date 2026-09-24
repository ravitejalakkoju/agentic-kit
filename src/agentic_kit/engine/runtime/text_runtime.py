"""The agent loop: prompt, model, tools, reply.

The loop is bounded on purpose. A model that keeps reaching for tools is not
making progress, so after MAX_TOOL_ROUNDS it is asked once more with none.

The prompt and output checkpoints live here because this is the only place the
assembled prompt and the drafted reply exist.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace

from ...domain.memory import MemoryUpdate, WorkingMemory
from ...domain.models import ConversationState, Message, Role, SopDefinition, TurnRequest
from ...domain.tools import ToolCall
from ...errors import ProviderError
from ...ports.llm import LlmPort, LlmReply, LlmRequest, ToolExchange
from ..guardrails import Checkpoint, Finding, GuardrailResponder, Guardrails
from ..prompt.builder import PromptBuilder
from ..tools.registry import ToolRegistry

logger = logging.getLogger("agentic_kit.runtime")

MAX_TOOL_ROUNDS = 3
GAVE_UP_REPLY = "I could not finish looking that up. Let me get a teammate to help."


@dataclass(frozen=True, slots=True)
class RuntimeReply:
    text: str
    blocked_by: Finding | None = None
    """Set when a guardrail replaced the model's answer with a safe one."""
    tool_calls: tuple[ToolCall, ...] = ()
    """Every tool the model used this turn, in the order it asked for them."""
    memory: MemoryUpdate = field(default_factory=MemoryUpdate)
    """What those tools learned, screened but not yet committed."""
    finished: bool = False
    """A tool reported the procedure done, so the next message routes afresh."""


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
        memory = conversation.memory
        offered = self._tools.definitions(request)
        prompt = await self._prompts.build(sop=sop, request=request, tools=offered, memory=memory)

        blocked = self._check(Checkpoint.PROMPT, request, prompt.system)
        if blocked is not None:
            return blocked

        messages = (*conversation.history, Message(role=Role.USER, text=request.text))
        reply, used, learned, finished = await self._converse(
            LlmRequest(system=prompt.system, messages=messages, tools=offered), request, memory
        )

        if not reply.text.strip():
            raise ProviderError("model returned an empty reply")

        # A blocked reply still keeps what the tools found; only the words were wrong.
        checked = self._check(Checkpoint.OUTPUT, request, reply.text)
        if checked is not None:
            return replace(checked, memory=learned, finished=finished)
        return RuntimeReply(text=reply.text, tool_calls=used, memory=learned, finished=finished)

    async def _converse(
        self, ask: LlmRequest, request: TurnRequest, memory: WorkingMemory
    ) -> tuple[LlmReply, tuple[ToolCall, ...], MemoryUpdate, bool]:
        """Trade tool calls with the model until it answers or runs out of rounds."""
        exchanges: list[ToolExchange] = []
        used: list[ToolCall] = []
        learned = MemoryUpdate()
        finished = False

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
                return reply, tuple(used), learned, finished

            used.extend(reply.tool_calls)
            exchange, update, done = await self._run_tools(reply.tool_calls, request, memory)
            exchanges.append(exchange)
            learned = learned.then(update)
            finished = finished or done

        return await self._final_answer(ask, exchanges), tuple(used), learned, finished

    async def _run_tools(
        self, calls: tuple[ToolCall, ...], request: TurnRequest, memory: WorkingMemory
    ) -> tuple[ToolExchange, MemoryUpdate, bool]:
        results = await asyncio.gather(
            *(self._tools.execute(call, request, memory) for call in calls),
        )
        paired = tuple(zip(calls, results, strict=True))
        exchange = ToolExchange(
            calls=calls,
            results=tuple((call.id, result.for_model()) for call, result in paired),
        )
        update = MemoryUpdate.from_tools((call.name, result) for call, result in paired)
        finished = any(result.finishes for result in results)
        return exchange, self._screen(request, update), finished

    def _screen(self, request: TurnRequest, update: MemoryUpdate) -> MemoryUpdate:
        """Drop facts carrying text that should never reach a later prompt.

        A fact joins the system prompt on every turn after this one, so a
        poisoned lookup would outlive the turn that fetched it. Losing the fact
        costs a repeated question; the reply still goes out.
        """
        poisoned = {
            fact.key
            for fact in update.facts
            if not self._guardrails.check(Checkpoint.MEMORY, fact.value, request=request).passed
        }
        if not poisoned:
            return update
        logger.warning("dropped facts that failed memory screening: %s", sorted(poisoned))
        return update.without(poisoned)

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
        verdict = self._guardrails.check(checkpoint, text, request=request)
        failure = verdict.failure
        if verdict.passed or failure is None:
            return None
        return RuntimeReply(text=self._responder.reply_for(failure), blocked_by=failure)
