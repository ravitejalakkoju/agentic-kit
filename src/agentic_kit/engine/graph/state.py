"""The vocabulary of the graph: node names, outcomes, and the state they pass along."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.models import ConversationState, SopDefinition, TurnRequest, TurnResult


class NodeKey(StrEnum):
    LOAD_STATE = "load_state"
    GUARD_INPUT = "guard_input"
    ROUTE = "route"
    SELECT_SOP = "select_sop"
    DETECT_SOP_DRIFT = "detect_sop_drift"
    BUILD_RUNTIME_REQUEST = "build_runtime_request"
    RUN_RUNTIME = "run_runtime"
    REVIEW_RESULT = "review_result"
    PERSIST_STATE = "persist_state"
    FINALIZE = "finalize"
    HANDOFF = "handoff"
    FAILED = "failed"


class Outcome(StrEnum):
    """What a node decided. The edge map turns this into the next node."""

    CONTINUE = "continue"
    NO_MATCH = "no_match"
    BLOCKED = "blocked"
    HANDOFF = "handoff"
    ENDED = "ended"
    FAILED = "failed"


@dataclass(slots=True)
class GraphState:
    """Mutable working set for one turn. Nodes read and write it in place."""

    request: TurnRequest
    conversation: ConversationState | None = None
    sop: SopDefinition | None = None
    reply: str | None = None
    result: TurnResult | None = None
