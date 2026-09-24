"""The agent loop: prompt, model, reply.

Guardrails and tool calls belong here when they exist, which keeps the graph
above unaware of how a reply is produced.
"""

from __future__ import annotations

from ...domain.models import ConversationState, Message, Role, SopDefinition, TurnRequest
from ...errors import ProviderError
from ...ports.llm import LlmPort
from ..prompt.builder import PromptBuilder


class TextRuntime:
    def __init__(self, llm: LlmPort, prompts: PromptBuilder) -> None:
        self._llm = llm
        self._prompts = prompts

    async def run(
        self, *, sop: SopDefinition, conversation: ConversationState, request: TurnRequest
    ) -> str:
        prompt = await self._prompts.build(sop=sop, request=request)
        messages = [*conversation.history, Message(role=Role.USER, text=request.text)]
        reply = await self._llm.complete(system=prompt.system, messages=messages)
        if not reply.strip():
            raise ProviderError("model returned an empty reply")
        return reply
