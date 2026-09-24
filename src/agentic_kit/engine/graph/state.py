"""The vocabulary of the graph: node names, outcomes, and the state they pass along."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ...domain.memory import MemoryUpdate
from ...domain.models import ConversationState, SopDefinition, TurnRequest, TurnResult
from ..routing.matcher import Ranking


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
    SWITCHED = "switched"
    """The conversation changed procedure. It goes where CONTINUE goes, and is
    its own outcome so the change shows up in the edge map and the logs rather
    than only in what the customer notices."""
    CLARIFY = "clarify"
    """Two procedures fit equally well, so the turn asks instead of guessing."""


@dataclass(slots=True)
class GraphState:
    """Mutable working set for one turn. Nodes read and write it in place."""

    request: TurnRequest
    conversation: ConversationState | None = None
    sop: SopDefinition | None = None
    reply: str | None = None
    result: TurnResult | None = None
    memory: MemoryUpdate = field(default_factory=MemoryUpdate)
    """What this turn's tools learned, waiting to be committed when the turn is written down."""
    matches: Ranking = field(default_factory=Ranking)
    """How the catalog ranked against this message. Scored once, read by two nodes."""
    finished: bool = False
    """The agent said the procedure is done, so the next message routes afresh."""
