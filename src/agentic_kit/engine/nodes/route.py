from __future__ import annotations

from ...errors import EngineError
from ...ports.stores import SopCatalog
from ..graph.state import GraphState, NodeKey, Outcome


class RouteNode:
    """Picks up whatever procedure the conversation was already on.

    Deliberately does no choosing. "What were we doing" is a lookup and "what
    should we do now" is a judgement, and keeping them in separate nodes is
    what lets the judgement change without this one being touched.
    """

    key = NodeKey.ROUTE
    outcomes = frozenset({Outcome.CONTINUE})

    def __init__(self, catalog: SopCatalog) -> None:
        self._catalog = catalog

    async def run(self, state: GraphState) -> Outcome:
        conversation = state.conversation
        if conversation is None:
            raise EngineError("routing reached without a conversation")

        if conversation.active_sop_id:
            state.sop = await self._catalog.get(conversation.active_sop_id)
        return Outcome.CONTINUE
