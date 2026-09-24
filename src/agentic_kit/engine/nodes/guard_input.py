from __future__ import annotations

from ...domain.models import TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome
from ..guardrails import Action, Checkpoint, GuardrailResponder, Guardrails


class GuardInputNode:
    """Screens the customer's message before the engine spends anything on it.

    Sitting ahead of routing means a blocked turn never selects a procedure or
    reaches the model, and the graph shows where that decision is made.
    """

    key = NodeKey.GUARD_INPUT
    outcomes = frozenset({Outcome.CONTINUE, Outcome.BLOCKED, Outcome.HANDOFF})

    def __init__(self, guardrails: Guardrails, responder: GuardrailResponder) -> None:
        self._guardrails = guardrails
        self._responder = responder

    async def run(self, state: GraphState) -> Outcome:
        if state.conversation is None:
            raise EngineError("input screening reached without a conversation")

        verdict = self._guardrails.check(Checkpoint.INPUT, state.request, state.request.text)
        failure = verdict.failure
        if verdict.passed or failure is None:
            return Outcome.CONTINUE

        handing_off = failure.action is Action.HANDOFF
        state.reply = self._responder.reply_for(failure)
        state.result = TurnResult(
            status=TurnStatus.HANDOFF if handing_off else TurnStatus.BLOCKED,
            outcome=TurnOutcome.POLICY_BLOCK,
            reply=state.reply,
            reason=failure.message,
        )
        return Outcome.HANDOFF if handing_off else Outcome.BLOCKED
