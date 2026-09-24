"""Runs the graph.

The executor knows about nodes and edges in general, and about no node in
particular. Adding or reordering a node is an edit to the node set and the edge
map, never to this loop.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...domain.models import TurnRequest, TurnResult
from ...errors import EngineError
from .edges import EDGES, ENTRY, Edge, validate_graph
from .node import Node
from .observer import GraphObserver, LoggingObserver
from .state import GraphState, NodeKey


class GraphExecutor:
    def __init__(
        self,
        nodes: Mapping[NodeKey, Node],
        edges: Mapping[Edge, NodeKey | None] = EDGES,
        observer: GraphObserver | None = None,
        entry: NodeKey = ENTRY,
    ) -> None:
        validate_graph(nodes, edges, entry)
        self._nodes = dict(nodes)
        self._edges = edges
        self._observer = observer or LoggingObserver()
        self._entry = entry

    async def run(self, request: TurnRequest) -> TurnResult:
        state = GraphState(request=request)
        await self._observer.execution_started(state)

        key: NodeKey | None = self._entry
        while key is not None:
            node = self._nodes[key]
            try:
                outcome = await node.run(state)
            except EngineError as error:
                await self._observer.execution_failed(state, error)
                raise
            except Exception as error:
                await self._observer.execution_failed(state, error)
                raise EngineError(f"node {key} raised {error!r}") from error

            await self._observer.node_completed(key, state, outcome)

            if (key, outcome) not in self._edges:
                raise EngineError(f"no edge from {key} on {outcome}")
            key = self._edges[(key, outcome)]

        if state.result is None:
            raise EngineError("graph finished without a result")

        await self._observer.execution_completed(state, state.result)
        return state.result
