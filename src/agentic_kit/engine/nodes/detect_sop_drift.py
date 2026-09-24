from __future__ import annotations

import logging

from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome

logger = logging.getLogger("agentic_kit.routing")


class DetectSopDriftNode:
    """Notices when the customer has moved on to something else.

    Works off the scores the previous node already produced, so a change of
    subject costs a comparison rather than a second opinion from a model. The
    margin is what stops a passing mention from moving the conversation.
    """

    key = NodeKey.DETECT_SOP_DRIFT
    outcomes = frozenset({Outcome.CONTINUE, Outcome.SWITCHED})

    async def run(self, state: GraphState) -> Outcome:
        active = state.sop
        if active is None:
            raise EngineError("drift detection reached without a procedure")

        better = state.matches.beats(active.sop_id)
        if better is None:
            return Outcome.CONTINUE

        logger.info(
            "conversation moved from %s to %s (%.3f against %.3f)",
            active.sop_id,
            better.sop.sop_id,
            better.score,
            state.matches.score_of(active.sop_id),
        )
        state.sop = better.sop
        return Outcome.SWITCHED
