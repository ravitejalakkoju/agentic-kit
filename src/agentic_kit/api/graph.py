"""The topology, served so the flow can be read without opening the source."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..engine.graph.edges import EDGES, ENTRY
from ..engine.graph.state import NodeKey, Outcome

router = APIRouter(prefix="/v1", tags=["graph"])


class GraphEdge(BaseModel):
    source: NodeKey
    outcome: Outcome
    target: NodeKey | None


class GraphView(BaseModel):
    entry: NodeKey
    edges: list[GraphEdge]


@router.get("/graph", summary="Show the conversation graph")
async def get_graph() -> GraphView:
    edges = [
        GraphEdge(source=source, outcome=outcome, target=target)
        for (source, outcome), target in EDGES.items()
    ]
    return GraphView(entry=ENTRY, edges=edges)
