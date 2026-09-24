from __future__ import annotations

from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome


class HandoffNode:
    """Marks the conversation as needing a human.

    Every path that gives up on automation passes through here, so there is one
    place that records the flag.
    """

    key = NodeKey.HANDOFF
    outcomes = frozenset({Outcome.CONTINUE})

    async def run(self, state: GraphState) -> Outcome:
        if state.conversation is None or state.result is None:
            raise EngineError("handoff reached without a conversation and a result")

        state.conversation.human_handoff_requested = True
        return Outcome.CONTINUE
