from __future__ import annotations

from ...domain.models import ConversationState, TurnOutcome, TurnResult, TurnStatus
from ...ports.stores import ConversationStore
from ..graph.state import GraphState, NodeKey, Outcome


class LoadStateNode:
    """Fetches or starts the conversation, and rejects turns that cannot run."""

    key = NodeKey.LOAD_STATE
    outcomes = frozenset({Outcome.CONTINUE, Outcome.NO_MATCH, Outcome.ENDED})

    def __init__(self, conversations: ConversationStore) -> None:
        self._conversations = conversations

    async def run(self, state: GraphState) -> Outcome:
        request = state.request
        if not request.text.strip() or not request.customer_id.strip():
            state.result = TurnResult(
                status=TurnStatus.NOOP,
                outcome=TurnOutcome.NO_MATCH,
                reason="a turn needs a customer and some text",
            )
            return Outcome.NO_MATCH

        conversation = await self._conversations.get(request.conversation_id)
        if conversation is None:
            conversation = ConversationState(
                conversation_id=request.conversation_id,
                customer_id=request.customer_id,
            )
        state.conversation = conversation

        if conversation.closed:
            state.result = TurnResult(
                status=TurnStatus.ENDED,
                outcome=TurnOutcome.RUN_ENDED,
                reason="conversation is closed",
            )
            return Outcome.ENDED

        if conversation.human_handoff_requested:
            state.result = TurnResult(
                status=TurnStatus.NOOP,
                outcome=TurnOutcome.USER_REQUESTED_HUMAN,
                reason="waiting for a human teammate",
            )
            return Outcome.NO_MATCH

        return Outcome.CONTINUE
