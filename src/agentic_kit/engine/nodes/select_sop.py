from __future__ import annotations

from ...domain.models import FlowKind, TurnOutcome, TurnResult, TurnStatus
from ...errors import EngineError
from ..graph.state import GraphState, NodeKey, Outcome
from ..routing.matcher import SopMatcher

CLARIFY_REPLY = "I can help with {first}, or with {second}. Which of those is it?"


class SelectSopNode:
    """Chooses the procedure for a conversation that is not already on one.

    Ranks the catalog whether or not a choice is needed, because drift wants
    the same numbers a moment later and asking twice would mean embedding the
    same sentence twice. An event skips all of that: it arrives with a name,
    and the procedure that claims the name has already made the choice.
    """

    key = NodeKey.SELECT_SOP
    outcomes = frozenset({Outcome.CONTINUE, Outcome.CLARIFY, Outcome.NO_MATCH})

    def __init__(self, matcher: SopMatcher) -> None:
        self._matcher = matcher

    async def run(self, state: GraphState) -> Outcome:
        if state.conversation is None:
            raise EngineError("selection reached without a conversation")

        if state.request.kind is FlowKind.EVENT:
            return await self._whoever_claims(state)

        state.matches = await self._matcher.rank(state.request.text)
        if state.sop is not None:
            return Outcome.CONTINUE

        leaders = state.matches.leaders
        if not leaders:
            return await self._fall_back(state)
        if state.matches.ambiguous:
            return self._ask_which(state)

        state.sop = leaders[0].sop
        return Outcome.CONTINUE

    async def _whoever_claims(self, state: GraphState) -> Outcome:
        """An event names itself, so nothing here is ranked.

        Decided before whatever the conversation was already on, because an
        event is not the customer changing the subject. Something happened,
        one procedure said it answers for it, and that settles it.
        """
        event = state.request.event
        state.sop = await self._matcher.for_event(event) if event else None
        if state.sop is not None:
            return Outcome.CONTINUE

        state.result = TurnResult(
            status=TurnStatus.NOOP,
            outcome=TurnOutcome.NO_MATCH,
            reason=f"no procedure answers {event}",
        )
        return Outcome.NO_MATCH

    async def _fall_back(self, state: GraphState) -> Outcome:
        """Nothing matched, which is what "hello" looks like to a matcher."""
        state.sop = await self._matcher.catch_all()
        if state.sop is not None:
            return Outcome.CONTINUE

        state.result = TurnResult(
            status=TurnStatus.NOOP,
            outcome=TurnOutcome.NO_MATCH,
            reason="no procedure matched, and none of them takes unmatched turns",
        )
        return Outcome.NO_MATCH

    def _ask_which(self, state: GraphState) -> Outcome:
        """Two procedures fit as well as each other.

        Asked here rather than after the fact, because this is the only moment
        the engine knows the choice was a coin toss. A turn later it would look
        like a settled decision the customer had gone along with.
        """
        first, second = state.matches.leaders[:2]
        state.reply = CLARIFY_REPLY.format(
            first=first.sop.description, second=second.sop.description
        )
        state.result = TurnResult(
            status=TurnStatus.RESPONDED,
            outcome=TurnOutcome.RESPONDED,
            reply=state.reply,
            reason="two procedures matched equally well",
        )
        return Outcome.CLARIFY
