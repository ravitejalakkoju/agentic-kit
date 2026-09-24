"""The graph topology, as data.

Nodes decide an outcome and nothing else; this table decides where that outcome
goes. Keeping it here means the whole conversation flow is readable, testable,
and renderable in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ...errors import GraphValidationError
from .node import Node
from .state import NodeKey, Outcome

Edge = tuple[NodeKey, Outcome]

ENTRY: Final[NodeKey] = NodeKey.LOAD_STATE

EDGES: Final[Mapping[Edge, NodeKey | None]] = MappingProxyType(
    {
        (NodeKey.LOAD_STATE, Outcome.CONTINUE): NodeKey.GUARD_INPUT,
        (NodeKey.LOAD_STATE, Outcome.NO_MATCH): NodeKey.FINALIZE,
        (NodeKey.LOAD_STATE, Outcome.ENDED): NodeKey.FINALIZE,
        (NodeKey.GUARD_INPUT, Outcome.CONTINUE): NodeKey.ROUTE,
        (NodeKey.GUARD_INPUT, Outcome.BLOCKED): NodeKey.PERSIST_STATE,
        (NodeKey.GUARD_INPUT, Outcome.HANDOFF): NodeKey.HANDOFF,
        (NodeKey.ROUTE, Outcome.CONTINUE): NodeKey.SELECT_SOP,
        (NodeKey.SELECT_SOP, Outcome.CONTINUE): NodeKey.DETECT_SOP_DRIFT,
        (NodeKey.SELECT_SOP, Outcome.CLARIFY): NodeKey.PERSIST_STATE,
        (NodeKey.SELECT_SOP, Outcome.NO_MATCH): NodeKey.FINALIZE,
        (NodeKey.DETECT_SOP_DRIFT, Outcome.CONTINUE): NodeKey.BUILD_RUNTIME_REQUEST,
        (NodeKey.DETECT_SOP_DRIFT, Outcome.SWITCHED): NodeKey.BUILD_RUNTIME_REQUEST,
        (NodeKey.BUILD_RUNTIME_REQUEST, Outcome.CONTINUE): NodeKey.RUN_RUNTIME,
        (NodeKey.RUN_RUNTIME, Outcome.CONTINUE): NodeKey.REVIEW_RESULT,
        (NodeKey.RUN_RUNTIME, Outcome.HANDOFF): NodeKey.HANDOFF,
        (NodeKey.RUN_RUNTIME, Outcome.FAILED): NodeKey.FAILED,
        (NodeKey.REVIEW_RESULT, Outcome.CONTINUE): NodeKey.PERSIST_STATE,
        (NodeKey.REVIEW_RESULT, Outcome.HANDOFF): NodeKey.HANDOFF,
        (NodeKey.HANDOFF, Outcome.CONTINUE): NodeKey.PERSIST_STATE,
        (NodeKey.FAILED, Outcome.CONTINUE): NodeKey.PERSIST_STATE,
        (NodeKey.PERSIST_STATE, Outcome.CONTINUE): NodeKey.FINALIZE,
        (NodeKey.FINALIZE, Outcome.CONTINUE): None,
    }
)


def validate_graph(
    nodes: Mapping[NodeKey, Node],
    edges: Mapping[Edge, NodeKey | None] = EDGES,
    entry: NodeKey = ENTRY,
) -> None:
    """Fail loudly at startup rather than mid-turn on an unusual path."""
    problems: list[str] = []

    if entry not in nodes:
        problems.append(f"entry node {entry} is not registered")

    for key, node in nodes.items():
        for outcome in node.outcomes:
            if (key, outcome) not in edges:
                problems.append(f"{key} can return {outcome} but has no edge for it")

    for (source, outcome), target in edges.items():
        if source not in nodes:
            problems.append(f"edge from unknown node {source}")
        elif outcome not in nodes[source].outcomes:
            problems.append(f"edge {source} -> {outcome} is unreachable, the node never returns it")
        if target is not None and target not in nodes:
            problems.append(f"edge {source} -> {outcome} points at unknown node {target}")

    orphans = sorted(set(nodes) - _reachable_from(entry, edges))
    problems.extend(f"node {key} is unreachable from {entry}" for key in orphans)

    if problems:
        raise GraphValidationError("; ".join(problems))


def _reachable_from(entry: NodeKey, edges: Mapping[Edge, NodeKey | None]) -> set[NodeKey]:
    reached = {entry}
    pending = [entry]
    while pending:
        source = pending.pop()
        for (edge_source, _), target in edges.items():
            if edge_source is source and target is not None and target not in reached:
                reached.add(target)
                pending.append(target)
    return reached
