"""Execution tracing. The executor reports; it does not decide what to do with it."""

from __future__ import annotations

import logging
from typing import Protocol

from ...domain.models import TurnResult
from .state import GraphState, NodeKey, Outcome

logger = logging.getLogger("agentic_kit.graph")


class GraphObserver(Protocol):
    async def execution_started(self, state: GraphState) -> None: ...

    async def node_completed(self, key: NodeKey, state: GraphState, outcome: Outcome) -> None: ...

    async def execution_completed(self, state: GraphState, result: TurnResult) -> None: ...

    async def execution_failed(self, state: GraphState, error: Exception) -> None: ...


class LoggingObserver:
    """Default observer. Spans and metrics can replace it without touching the executor."""

    async def execution_started(self, state: GraphState) -> None:
        logger.debug("turn started", extra=_context(state))

    async def node_completed(self, key: NodeKey, state: GraphState, outcome: Outcome) -> None:
        logger.debug("node %s -> %s", key, outcome, extra=_context(state))

    async def execution_completed(self, state: GraphState, result: TurnResult) -> None:
        logger.info("turn %s / %s", result.status, result.outcome, extra=_context(state))

    async def execution_failed(self, state: GraphState, error: Exception) -> None:
        logger.exception("turn failed: %s", error, extra=_context(state))


def _context(state: GraphState) -> dict[str, str | None]:
    return {
        "conversation_id": state.request.conversation_id,
        "sop_id": state.sop.sop_id if state.sop else None,
    }
