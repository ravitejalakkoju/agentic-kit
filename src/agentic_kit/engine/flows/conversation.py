from __future__ import annotations

from ...domain.models import TurnRequest, TurnResult
from ..graph.executor import GraphExecutor


class ConversationFlow:
    """The conversation kind, which is the graph."""

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def handle(self, request: TurnRequest) -> TurnResult:
        return await self._executor.run(request)
