"""The agent loop: prompt, model, reply.

Guardrails and tool calls belong here when they exist, which keeps the graph
above unaware of how a reply is produced.
"""

from __future__ import annotations

from ...domain.models import ConversationState, Message, Role, SopDefinition
from ...errors import ProviderError
from ...ports.llm import LlmPort
from ..prompt.assembler import build_system_prompt


class TextRuntime:
    def __init__(self, llm: LlmPort) -> None:
        self._llm = llm

    async def run(self, *, sop: SopDefinition, conversation: ConversationState, text: str) -> str:
        messages = [*conversation.history, Message(role=Role.USER, text=text)]
        reply = await self._llm.complete(system=build_system_prompt(sop), messages=messages)
        if not reply.strip():
            raise ProviderError("model returned an empty reply")
        return reply
