"""The entry point into the engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ..domain.models import FlowKind, TurnOutcome, TurnRequest, TurnResult, TurnStatus


class Flow(Protocol):
    async def handle(self, request: TurnRequest) -> TurnResult: ...


class AiEngine:
    """Resolves a flow for the requested kind.

    Only conversation is implemented; event, callback, and task are part of the
    engine's shape and answer with a no-op until they exist.
    """

    def __init__(self, flows: Mapping[FlowKind, Flow]) -> None:
        self._flows = dict(flows)

    async def handle(self, kind: FlowKind, request: TurnRequest) -> TurnResult:
        flow = self._flows.get(kind)
        if flow is None:
            return TurnResult(
                status=TurnStatus.NOOP,
                outcome=TurnOutcome.NO_MATCH,
                reason=f"no flow registered for {kind}",
            )
        return await flow.handle(request)
