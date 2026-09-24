from __future__ import annotations

from ...domain.models import Message, Role, RunRecord, TurnStatus
from ...errors import EngineError
from ...ports.stores import ConversationStore, RunStore
from ..graph.state import GraphState, NodeKey, Outcome


class PersistStateNode:
    """Writes the turn down: history, active procedure, and a run record."""

    key = NodeKey.PERSIST_STATE
    outcomes = frozenset({Outcome.CONTINUE})

    def __init__(self, conversations: ConversationStore, runs: RunStore) -> None:
        self._conversations = conversations
        self._runs = runs

    async def run(self, state: GraphState) -> Outcome:
        conversation, result = state.conversation, state.result
        if conversation is None or result is None:
            raise EngineError("persistence reached without a conversation and a result")

        conversation.append(Message(role=Role.USER, text=state.request.text))
        if state.reply:
            conversation.append(Message(role=Role.AGENT, text=state.reply))

        if state.sop is not None:
            conversation.active_sop_id = state.sop.sop_id
            conversation.active_agent_id = state.sop.agent_id

        if result.status is TurnStatus.RESPONDED:
            conversation.attempts = 0

        await self._conversations.save(conversation)

        run = RunRecord(
            conversation_id=conversation.conversation_id,
            agent_id=conversation.active_agent_id,
            sop_id=conversation.active_sop_id,
            status=result.status,
            outcome=result.outcome,
        )
        await self._runs.append(run)
        result.run_id = run.run_id

        return Outcome.CONTINUE
