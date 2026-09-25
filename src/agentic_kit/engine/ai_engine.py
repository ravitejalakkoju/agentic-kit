"""The entry point into the engine."""

from __future__ import annotations

from ..domain.models import TurnRequest, TurnResult
from .graph.executor import GraphExecutor


class AiEngine:
    """The front door. Every kind of turn runs the same graph.

    There was a flow per kind here once, on the assumption that a turn nobody
    typed would need a pipeline of its own. It does not: an event differs from
    a question in which procedure answers it and whose words get written down,
    and both of those are decisions inside nodes that already existed. A seam
    with one thing behind it is a seam that has not been earned, and it comes
    back cheaply on the day a kind genuinely needs a different path.
    """

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def handle(self, request: TurnRequest) -> TurnResult:
        return await self._executor.run(request)
