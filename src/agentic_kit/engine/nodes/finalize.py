from __future__ import annotations

from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome


class FinalizeNode:
    """The single exit. Every path ends here with a result already on the state."""

    key = NodeKey.FINALIZE
    outcomes = frozenset({Outcome.CONTINUE})

    async def run(self, state: GraphState) -> Outcome:
        if state.result is None:
            raise EngineError("finalize reached without a result")
        return Outcome.CONTINUE
