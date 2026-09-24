from __future__ import annotations

from ...domain.models import Message, Role, RunRecord, TurnStatus
from ...errors import EngineError
from ...ports.stores import ConversationStore, RunStore
from ..graph.state import GraphState, NodeKey, Outcome


class PersistStateNode:
    """Writes the turn down: history, active procedure, what it learned, and a run record.

    The run record is built before the save so every fact can name the turn
    that taught it. Every path to a reply comes through here, so a turn whose
    answer was blocked still keeps what its tools found out.
    """

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

        run = RunRecord(
            conversation_id=conversation.conversation_id,
            agent_id=conversation.active_agent_id,
            sop_id=conversation.active_sop_id,
            status=result.status,
            outcome=result.outcome,
        )
        conversation.memory.apply(state.memory, run_id=run.run_id)

        await self._conversations.save(conversation)
        await self._runs.append(run)
        result.run_id = run.run_id

        return Outcome.CONTINUE
