"""What every node in the graph looks like."""

from __future__ import annotations

from typing import Protocol

from .state import GraphState, NodeKey, Outcome


class Node(Protocol):
    key: NodeKey
    outcomes: frozenset[Outcome]
    """Every outcome this node can return. The edge map must cover all of them."""

    async def run(self, state: GraphState) -> Outcome: ...
