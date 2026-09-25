from __future__ import annotations

from ...domain.models import FlowKind, Message, Role, RunRecord, TurnStatus
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

        # History is a record of who said what. An event describes what
        # happened rather than what anyone said, so only the reply goes down,
        # and it reads perfectly well on its own.
        if state.request.kind is FlowKind.CONVERSATION:
            conversation.append(Message(role=Role.USER, text=state.request.text))
        if state.reply:
            conversation.append(Message(role=Role.AGENT, text=state.reply))

        if state.finished:
            # The procedure is done, not the conversation: clearing it means the
            # next message is routed afresh rather than refused.
            conversation.active_sop_id = None
            conversation.active_agent_id = None
        elif state.sop is not None:
            conversation.active_sop_id = state.sop.sop_id
            conversation.active_agent_id = state.sop.agent_id

        if result.status is TurnStatus.RESPONDED:
            conversation.attempts = 0

        # Named from the turn rather than the conversation, so a procedure that
        # just finished is still on record as the one that ran.
        run = RunRecord(
            conversation_id=conversation.conversation_id,
            agent_id=state.sop.agent_id if state.sop else conversation.active_agent_id,
            sop_id=state.sop.sop_id if state.sop else conversation.active_sop_id,
            status=result.status,
            outcome=result.outcome,
        )
        conversation.memory.apply(state.memory, run_id=run.run_id)

        await self._conversations.save(conversation)
        await self._runs.append(run)
        result.run_id = run.run_id

        return Outcome.CONTINUE
