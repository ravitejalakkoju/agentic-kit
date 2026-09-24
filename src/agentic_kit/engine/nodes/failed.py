from __future__ import annotations

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome

RETRY_REPLY = "Something went wrong on my side. Could you say that again?"
GIVE_UP_REPLY = "I am handing this to a human teammate."


class FailedNode:
    """Handles a turn the model could not complete, and gives up if it keeps happening.

    The counter lives on the conversation, so it survives across turns and a
    single flaky model call does not escalate. Whether it has gone on too long
    is `ConversationState.stalled`, which a blocked reply consults as well.
    """

    key = NodeKey.FAILED
    outcomes = frozenset({Outcome.CONTINUE})

    async def run(self, state: GraphState) -> Outcome:
        conversation = state.conversation
        if conversation is None:
            raise EngineError("failure handling reached without a conversation")

        conversation.record_stall()
        exhausted = conversation.stalled
        reason = state.result.reason if state.result else None

        if exhausted:
            conversation.human_handoff_requested = True

        state.reply = GIVE_UP_REPLY if exhausted else RETRY_REPLY
        state.result = TurnResult(
            status=TurnStatus.HANDOFF if exhausted else TurnStatus.FAILED,
            outcome=TurnOutcome.MAX_ATTEMPTS if exhausted else TurnOutcome.FAILED,
            reply=state.reply,
            reason=reason,
        )
        return Outcome.CONTINUE
