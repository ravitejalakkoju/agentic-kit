from __future__ import annotations

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError, ProviderError
from ..graph.state import GraphState, NodeKey, Outcome
from ..guardrails import Action
from ..runtime.text_runtime import RuntimeReply, TextRuntime


class RunRuntimeNode:
    """Hands the turn to the agent loop and classifies what came back."""

    key = NodeKey.RUN_RUNTIME
    outcomes = frozenset({Outcome.CONTINUE, Outcome.HANDOFF, Outcome.FAILED})

    def __init__(self, runtime: TextRuntime) -> None:
        self._runtime = runtime

    async def run(self, state: GraphState) -> Outcome:
        if state.conversation is None or state.sop is None:
            raise EngineError("runtime reached without a conversation and a procedure")

        try:
            reply = await self._runtime.run(
                sop=state.sop,
                conversation=state.conversation,
                request=state.request,
            )
        except ProviderError as error:
            state.result = TurnResult(
                status=TurnStatus.FAILED,
                outcome=TurnOutcome.FAILED,
                reason=str(error),
            )
            return Outcome.FAILED

        state.reply = reply.text
        state.result = self._result_for(reply)
        return Outcome.HANDOFF if state.result.status is TurnStatus.HANDOFF else Outcome.CONTINUE

    def _result_for(self, reply: RuntimeReply) -> TurnResult:
        blocked = reply.blocked_by
        if blocked is None:
            return TurnResult(
                status=TurnStatus.RESPONDED,
                outcome=TurnOutcome.RESPONDED,
                reply=reply.text,
            )
        return TurnResult(
            status=(TurnStatus.HANDOFF if blocked.action is Action.HANDOFF else TurnStatus.BLOCKED),
            outcome=TurnOutcome.POLICY_BLOCK,
            reply=reply.text,
            reason=blocked.message,
        )
