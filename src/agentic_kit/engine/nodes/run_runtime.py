from __future__ import annotations

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError, ProviderError
from ..graph.state import GraphState, NodeKey, Outcome
from ..runtime.text_runtime import TextRuntime


class RunRuntimeNode:
    """Hands the turn to the agent loop and classifies what came back."""

    key = NodeKey.RUN_RUNTIME
    outcomes = frozenset({Outcome.CONTINUE, Outcome.FAILED})

    def __init__(self, runtime: TextRuntime) -> None:
        self._runtime = runtime

    async def run(self, state: GraphState) -> Outcome:
        if state.conversation is None or state.sop is None:
            raise EngineError("runtime reached without a conversation and a procedure")

        try:
            reply = await self._runtime.run(
                sop=state.sop,
                conversation=state.conversation,
                text=state.request.text,
            )
        except ProviderError as error:
            state.result = TurnResult(
                status=TurnStatus.FAILED,
                outcome=TurnOutcome.FAILED,
                reason=str(error),
            )
            return Outcome.FAILED

        state.reply = reply
        state.result = TurnResult(
            status=TurnStatus.RESPONDED,
            outcome=TurnOutcome.RESPONDED,
            reply=reply,
        )
        return Outcome.CONTINUE
