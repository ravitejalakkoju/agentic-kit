"""Which procedure a turn belongs to, and what happens when that changes."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agentic_kit.adapters.hash_embedder import HashEmbedder
from agentic_kit.adapters.memory import InMemorySopCatalog
from agentic_kit.composition import Components, build
from agentic_kit.domain.models import (
    MAX_ATTEMPTS,
    ConversationState,
    Personality,
    SopDefinition,
    TurnOutcome,
    TurnStatus,
)
from agentic_kit.engine.guardrails import (
    DEFAULT_PROFILE,
    Checkpoint,
    CheckpointPolicy,
    Finding,
    GuardrailContext,
    Level,
    Mode,
    default_detectors,
)
from agentic_kit.engine.nodes.failed import GIVE_UP_REPLY
from agentic_kit.engine.routing.matcher import AMBIGUITY, MARGIN, Match, Ranking, SopMatcher
from agentic_kit.seed import DEFAULT_SOPS, DELIVERY_SOP, RETURNS_SOP, SUPPORT_SOP
from agentic_kit.settings import Settings

from .fakes import ScriptedLlm, calls, send, tool_call

RETURN_ASK = "I want to return a jacket"
DELIVERY_ASK = "my parcel never turned up"
ORDER_ASK = "Where is my order?"


@pytest.fixture
def matcher() -> SopMatcher:
    return SopMatcher(InMemorySopCatalog(DEFAULT_SOPS), HashEmbedder())


def a_sop(sop_id: str, **fields) -> SopDefinition:
    return SopDefinition(
        sop_id=sop_id,
        agent_id=f"{sop_id}_agent",
        description=sop_id,
        instructions="Do the thing.",
        personality=Personality(name="Ava", identity="support", tone="warm"),
        **fields,
    )


def ranking(*scored: tuple[str, float], floor: float = 0.1) -> Ranking:
    return Ranking(tuple(Match(a_sop(name), score) for name, score in scored), floor=floor)


# --- ranking the catalog ----------------------------------------------------


async def test_a_question_lands_on_the_procedure_written_for_it(matcher: SopMatcher) -> None:
    for ask, expected in (
        (ORDER_ASK, SUPPORT_SOP),
        (RETURN_ASK, RETURNS_SOP),
        (DELIVERY_ASK, DELIVERY_SOP),
    ):
        best = (await matcher.rank(ask)).best
        assert best is not None and best.sop.sop_id == expected.sop_id, ask


async def test_a_question_the_catalog_has_nothing_to_say_about_matches_nothing(
    matcher: SopMatcher,
) -> None:
    assert (await matcher.rank("what is the capital of France")).leaders == ()


async def test_a_greeting_matches_nothing_which_is_why_there_is_a_catch_all(
    matcher: SopMatcher,
) -> None:
    assert (await matcher.rank("hello there")).leaders == ()
    catch_all = await matcher.catch_all()
    assert catch_all is not None and catch_all.sop_id == SUPPORT_SOP.sop_id


async def test_the_best_example_counts_not_the_average_of_them_all() -> None:
    """One sharp match should not be diluted by the examples it says nothing like."""
    focused = a_sop("focused", examples=["I want to return a jacket"])
    padded = a_sop(
        "padded",
        examples=["I want to return a jacket", "billing", "warranty", "gift cards", "vouchers"],
    )
    matcher = SopMatcher(InMemorySopCatalog([focused, padded]), HashEmbedder())

    scores = {m.sop.sop_id: m.score for m in (await matcher.rank(RETURN_ASK)).matches}

    assert scores["padded"] == pytest.approx(scores["focused"])


async def test_an_empty_catalog_ranks_nothing() -> None:
    matcher = SopMatcher(InMemorySopCatalog([]), HashEmbedder())

    assert (await matcher.rank(ORDER_ASK)) == Ranking(floor=HashEmbedder().min_score)


async def test_the_catalog_is_embedded_once_however_many_turns_ask() -> None:
    class CountingEmbedder(HashEmbedder):
        calls = 0

        async def embed(self, texts):
            type(self).calls += 1
            return await super().embed(texts)

    matcher = SopMatcher(InMemorySopCatalog(DEFAULT_SOPS), CountingEmbedder())
    await matcher.rank(ORDER_ASK)
    await matcher.rank(RETURN_ASK)

    # One for the whole catalog, then one per utterance.
    assert CountingEmbedder.calls == 3


# --- what the scores mean ---------------------------------------------------


def test_a_procedure_under_the_floor_is_not_a_leader() -> None:
    assert ranking(("returns", 0.05), floor=0.1).leaders == ()


def test_the_procedure_in_progress_is_kept_when_nothing_clearly_beats_it() -> None:
    close = ranking(("returns", 0.40), ("orders", 0.40 - MARGIN / 2))

    assert close.beats("orders") is None


def test_a_procedure_that_clearly_fits_better_wins() -> None:
    clear = ranking(("returns", 0.60), ("orders", 0.60 - MARGIN * 2))
    winner = clear.beats("orders")

    assert winner is not None and winner.sop.sop_id == "returns"


def test_the_leader_never_beats_itself() -> None:
    assert ranking(("orders", 0.9), ("returns", 0.1)).beats("orders") is None


def test_two_procedures_too_close_to_call_are_ambiguous() -> None:
    assert ranking(("returns", 0.40), ("delivery", 0.40 - AMBIGUITY / 2)).ambiguous
    assert not ranking(("returns", 0.40), ("delivery", 0.40 - AMBIGUITY * 2)).ambiguous


def test_one_procedure_on_its_own_is_never_ambiguous() -> None:
    assert not ranking(("returns", 0.40), ("delivery", 0.01)).ambiguous


# --- choosing, staying, moving ----------------------------------------------


async def test_a_new_conversation_lands_on_the_matching_procedure(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Happy to help with that return.")

    await send(components, RETURN_ASK)

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == RETURNS_SOP.sop_id
    assert conversation.active_agent_id == RETURNS_SOP.agent_id


async def test_a_greeting_is_answered_by_the_catch_all(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Hello, how can I help?")

    result = await send(components, "hello there")

    assert result.status is TurnStatus.RESPONDED
    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == SUPPORT_SOP.sop_id


async def test_a_follow_up_on_the_same_subject_stays_put(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Happy to help.", "Any time this week.")

    await send(components, RETURN_ASK)
    await send(components, "when can I send it back")

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == RETURNS_SOP.sop_id


async def test_changing_the_subject_moves_the_conversation_to_another_teammate(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Happy to help with that return.", "Let me look into the parcel.")

    await send(components, RETURN_ASK)
    await send(components, DELIVERY_ASK)

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == DELIVERY_SOP.sop_id
    assert conversation.active_agent_id == DELIVERY_SOP.agent_id

    assert [run.sop_id for run in components.runs.records] == [
        RETURNS_SOP.sop_id,
        DELIVERY_SOP.sop_id,
    ]


async def test_the_new_teammate_is_the_one_the_model_is_told_to_be(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Happy to help with that return.", "Let me look into the parcel.")

    await send(components, RETURN_ASK)
    await send(components, DELIVERY_ASK)

    first, second = llm.calls
    assert RETURNS_SOP.personality.name in first.system
    assert DELIVERY_SOP.personality.name in second.system
    assert DELIVERY_SOP.instructions in second.system


async def test_a_question_that_is_two_things_at_once_is_asked_about_rather_than_guessed(
    components: Components, llm: ScriptedLlm
) -> None:
    result = await send(components, "I need a refund for a damaged parcel")

    assert result.status is TurnStatus.RESPONDED
    assert RETURNS_SOP.description in result.reply
    assert DELIVERY_SOP.description in result.reply
    assert llm.calls == [], "asking which is which needs no model"


async def test_nothing_matched_and_no_catch_all_is_a_turn_that_does_not_run(
    settings: Settings, llm: ScriptedLlm
) -> None:
    components = build(settings, llm, sops=[RETURNS_SOP])

    result = await send(components, "what is the capital of France")

    assert result.status is TurnStatus.NOOP
    assert result.outcome is TurnOutcome.NO_MATCH
    assert llm.calls == []


# --- a turn that went nowhere -----------------------------------------------


class RefusesLongReplies:
    """The kind of enforcing output check a deployment adds on top of the defaults."""

    id = "refuses_long_replies"
    checkpoints = frozenset({Checkpoint.OUTPUT})

    def evaluate(self, context: GuardrailContext) -> Finding | None:
        if len(context.text) < 40:
            return None
        return Finding(
            detector_id=self.id,
            checkpoint=context.checkpoint,
            level=Level.FAIL,
            message="the reply runs on",
        )


def strict_about_replies(settings: Settings, llm: ScriptedLlm) -> Components:
    profile = replace(
        DEFAULT_PROFILE,
        checkpoints={
            **DEFAULT_PROFILE.checkpoints,
            Checkpoint.OUTPUT: CheckpointPolicy(Mode.ENFORCE, (RefusesLongReplies.id,)),
        },
    )
    return build(settings, llm, profile, detectors=[*default_detectors(), RefusesLongReplies()])


TOO_LONG = "I am afraid this answer goes on for rather a long while indeed."


async def test_a_reply_that_keeps_being_refused_ends_up_with_a_human(
    settings: Settings, llm: ScriptedLlm
) -> None:
    """FailedNode counts a model that errored; this counts one that answered unusably."""
    components = strict_about_replies(settings, llm)
    llm.queue(*[TOO_LONG] * MAX_ATTEMPTS)

    results = [await send(components, ORDER_ASK) for _ in range(MAX_ATTEMPTS)]

    assert [r.status for r in results[:-1]] == [TurnStatus.BLOCKED] * (MAX_ATTEMPTS - 1)
    assert results[-1].status is TurnStatus.HANDOFF
    assert results[-1].outcome is TurnOutcome.MAX_ATTEMPTS
    assert results[-1].reply == GIVE_UP_REPLY

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.human_handoff_requested is True


async def test_one_answer_that_lands_clears_the_count(settings: Settings, llm: ScriptedLlm) -> None:
    components = strict_about_replies(settings, llm)
    llm.queue(TOO_LONG, "It ships tomorrow.")

    await send(components, ORDER_ASK)
    await send(components, ORDER_ASK)

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.attempts == 0


def test_a_conversation_counts_both_kinds_of_wasted_turn() -> None:
    conversation = ConversationState(conversation_id="conv-1", customer_id="cust-1")

    for _ in range(MAX_ATTEMPTS - 1):
        conversation.record_stall()
    assert not conversation.stalled

    conversation.record_stall()
    assert conversation.stalled


# --- finishing a procedure --------------------------------------------------


async def test_finishing_a_procedure_lets_the_next_message_route_afresh(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(
        calls(tool_call("finish_procedure", summary="return arranged", confirmed=True)),
        "All sorted, thanks!",
        "Let me look into that parcel.",
    )

    await send(components, RETURN_ASK)
    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id is None, "the procedure ended, not the conversation"

    await send(components, DELIVERY_ASK)
    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == DELIVERY_SOP.sop_id


async def test_a_finished_procedure_is_still_named_on_the_run_that_ran_it(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(
        calls(tool_call("finish_procedure", summary="done", confirmed=True)),
        "All sorted.",
    )

    await send(components, RETURN_ASK)

    [run] = components.runs.records
    assert run.sop_id == RETURNS_SOP.sop_id
    assert run.agent_id == RETURNS_SOP.agent_id


async def test_finishing_needs_the_customer_asked_first(
    components: Components, llm: ScriptedLlm
) -> None:
    """A write tool the model reaches for unprompted is refused, like any other."""
    llm.queue(
        calls(tool_call("finish_procedure", summary="assuming we are done")),
        "Is there anything else?",
    )

    await send(components, RETURN_ASK)

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.active_sop_id == RETURNS_SOP.sop_id


async def test_the_conversation_is_not_closed_by_finishing(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(
        calls(tool_call("finish_procedure", summary="done", confirmed=True)),
        "All sorted.",
        "Of course, what else?",
    )

    await send(components, RETURN_ASK)
    result = await send(components, "actually one more thing")

    assert result.status is TurnStatus.RESPONDED
