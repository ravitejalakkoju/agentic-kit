from __future__ import annotations

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError
from ...ports.stores import SopCatalog
from ..graph.state import GraphState, NodeKey, Outcome


class RouteNode:
    """Picks the procedure for this turn, or refuses to run one.

    A conversation stays on its active procedure. Choosing between several
    procedures is the catalog's job, so a classifier can arrive without
    touching this node.
    """

    key = NodeKey.ROUTE
    outcomes = frozenset({Outcome.CONTINUE, Outcome.NO_MATCH})

    def __init__(self, catalog: SopCatalog) -> None:
        self._catalog = catalog

    async def run(self, state: GraphState) -> Outcome:
        conversation = state.conversation
        if conversation is None:
            raise EngineError("routing reached without a conversation")

        sop = None
        if conversation.active_sop_id:
            sop = await self._catalog.get(conversation.active_sop_id)
        if sop is None:
            sop = await self._catalog.match(state.request.text)
        if sop is None:
            state.result = TurnResult(
                status=TurnStatus.NOOP,
                outcome=TurnOutcome.NO_MATCH,
                reason="no procedure matched",
            )
            return Outcome.NO_MATCH

        state.sop = sop
        return Outcome.CONTINUE
