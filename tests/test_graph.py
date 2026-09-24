from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentic_kit.composition import Components
from agentic_kit.engine.graph.edges import EDGES, ENTRY, validate_graph
from agentic_kit.engine.graph.state import GraphState, NodeKey, Outcome
from agentic_kit.errors import GraphValidationError


@dataclass
class StubNode:
    key: NodeKey
    outcomes: frozenset[Outcome] = frozenset({Outcome.CONTINUE})

    async def run(self, state: GraphState) -> Outcome:
        return Outcome.CONTINUE


def test_composed_graph_passes_validation(components: Components) -> None:
    """`build` validates the node set against the edge table; reaching here means it passed."""
    assert components.engine is not None


def test_input_screening_sits_between_loading_and_routing() -> None:
    assert EDGES[(NodeKey.LOAD_STATE, Outcome.CONTINUE)] is NodeKey.GUARD_INPUT
    assert EDGES[(NodeKey.GUARD_INPUT, Outcome.CONTINUE)] is NodeKey.ROUTE


def test_a_blocked_turn_is_still_written_down() -> None:
    assert EDGES[(NodeKey.GUARD_INPUT, Outcome.BLOCKED)] is NodeKey.PERSIST_STATE
    assert EDGES[(NodeKey.GUARD_INPUT, Outcome.HANDOFF)] is NodeKey.HANDOFF


def test_every_edge_target_is_a_known_node() -> None:
    targets = {target for target in EDGES.values() if target is not None}
    assert targets <= set(NodeKey)


def test_missing_edge_for_a_declared_outcome_is_rejected() -> None:
    nodes = {ENTRY: StubNode(ENTRY, frozenset({Outcome.CONTINUE, Outcome.FAILED}))}
    edges = {(ENTRY, Outcome.CONTINUE): None}

    with pytest.raises(GraphValidationError, match="has no edge"):
        validate_graph(nodes, edges)


def test_edge_for_an_outcome_the_node_never_returns_is_rejected() -> None:
    nodes = {ENTRY: StubNode(ENTRY)}
    edges = {(ENTRY, Outcome.CONTINUE): None, (ENTRY, Outcome.FAILED): None}

    with pytest.raises(GraphValidationError, match="unreachable, the node never returns it"):
        validate_graph(nodes, edges)


def test_edge_to_an_unregistered_node_is_rejected() -> None:
    nodes = {ENTRY: StubNode(ENTRY)}
    edges = {(ENTRY, Outcome.CONTINUE): NodeKey.FINALIZE}

    with pytest.raises(GraphValidationError, match="unknown node"):
        validate_graph(nodes, edges)


def test_node_unreachable_from_entry_is_rejected() -> None:
    nodes = {ENTRY: StubNode(ENTRY), NodeKey.FINALIZE: StubNode(NodeKey.FINALIZE)}
    edges = {(ENTRY, Outcome.CONTINUE): None, (NodeKey.FINALIZE, Outcome.CONTINUE): None}

    with pytest.raises(GraphValidationError, match="unreachable from"):
        validate_graph(nodes, edges)
