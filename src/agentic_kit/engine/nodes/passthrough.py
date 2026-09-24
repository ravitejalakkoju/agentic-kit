from __future__ import annotations

from ..graph.state import GraphState, NodeKey, Outcome


class PassThroughNode:
    """A node that is on the graph but has no behavior yet.

    Procedure selection, drift detection, request building, and result review
    are real stages of the flow; they are stubbed so that implementing one is a
    change to a single node rather than a change to the topology.
    """

    outcomes = frozenset({Outcome.CONTINUE})

    def __init__(self, key: NodeKey) -> None:
        self.key = key

    async def run(self, state: GraphState) -> Outcome:
        return Outcome.CONTINUE
