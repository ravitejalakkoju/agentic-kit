from __future__ import annotations

import logging

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome
from .failed import GIVE_UP_REPLY

logger = logging.getLogger("agentic_kit.routing")


class ReviewResultNode:
    """Looks at what the turn produced before it is written down.

    Everything that reaches here has a reply, so the question is not whether
    the model answered but whether the answer was any use. A reply the output
    checkpoint had to replace is a turn the customer got nothing from, and
    without someone counting those it can repeat for as long as they keep
    asking. `FailedNode` counts the same thing on the path where the model
    never answered at all, which is why the rule lives on the conversation
    rather than in either node.
    """

    key = NodeKey.REVIEW_RESULT
    outcomes = frozenset({Outcome.CONTINUE, Outcome.HANDOFF})

    async def run(self, state: GraphState) -> Outcome:
        conversation, result = state.conversation, state.result
        if conversation is None or result is None:
            raise EngineError("review reached without a conversation and a result")

        if state.finished and state.sop is not None:
            logger.info("procedure %s finished", state.sop.sop_id)

        if result.status is not TurnStatus.BLOCKED:
            return Outcome.CONTINUE

        conversation.record_stall()
        if not conversation.stalled:
            return Outcome.CONTINUE

        state.reply = GIVE_UP_REPLY
        state.result = TurnResult(
            status=TurnStatus.HANDOFF,
            outcome=TurnOutcome.MAX_ATTEMPTS,
            reply=state.reply,
            reason=result.reason,
        )
        return Outcome.HANDOFF
