"""One turn through the engine, end to end, with a scripted model."""

from __future__ import annotations

from agentic_kit.composition import Components
from agentic_kit.domain.models import (
    HISTORY_LIMIT,
    FlowKind,
    Role,
    TurnContext,
    TurnOutcome,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agentic_kit.engine.nodes.failed import GIVE_UP_REPLY, MAX_ATTEMPTS, RETRY_REPLY
from agentic_kit.engine.nodes.route import POLICY_REPLY
from agentic_kit.errors import ProviderError
from agentic_kit.seed import SUPPORT_SOP

from .fakes import ScriptedLlm


async def send(
    components: Components,
    text: str,
    *,
    conversation_id: str = "conv-1",
    customer_id: str = "cust-1",
    context: TurnContext | None = None,
) -> TurnResult:
    request = TurnRequest(
        conversation_id=conversation_id,
        customer_id=customer_id,
        text=text,
        context=context or TurnContext(),
    )
    return await components.engine.handle(FlowKind.CONVERSATION, request)


async def test_order_named_in_the_turn_reaches_the_model_prompt(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("It has shipped.")

    await send(components, "Where is my order?", context=TurnContext(order_id="1001"))

    [call] = llm.calls
    assert '"order_id": "1001"' in call.system
    assert '"status": "shipped"' in call.system
    assert "Priya Sharma" in call.system


async def test_reply_is_returned_and_the_turn_is_persisted(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Your order ships tomorrow.")

    result = await send(components, "Where is my order?")

    assert result.status is TurnStatus.RESPONDED
    assert result.outcome is TurnOutcome.RESPONDED
    assert result.reply == "Your order ships tomorrow."
    assert result.run_id is not None

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert [(m.role, m.text) for m in conversation.history] == [
        (Role.USER, "Where is my order?"),
        (Role.AGENT, "Your order ships tomorrow."),
    ]
    assert conversation.active_sop_id == SUPPORT_SOP.sop_id
    assert conversation.active_agent_id == SUPPORT_SOP.agent_id

    [run] = components.runs.records
    assert run.run_id == result.run_id
    assert run.sop_id == SUPPORT_SOP.sop_id


async def test_model_receives_the_sop_prompt_and_prior_history(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("Which order?", "Found it.")

    await send(components, "Where is my order?")
    await send(components, "Order 42")

    first, second = llm.calls
    assert SUPPORT_SOP.personality.name in first.system
    assert SUPPORT_SOP.instructions in first.system
    assert [m.text for m in second.messages] == ["Where is my order?", "Which order?", "Order 42"]


async def test_policy_phrase_hands_off_without_calling_the_model(
    components: Components, llm: ScriptedLlm
) -> None:
    result = await send(components, "Someone tried to hack account access")

    assert result.status is TurnStatus.HANDOFF
    assert result.outcome is TurnOutcome.POLICY_BLOCK
    assert result.reply == POLICY_REPLY
    assert llm.calls == []

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.human_handoff_requested is True


async def test_blank_text_is_a_noop_and_nothing_is_saved(
    components: Components, llm: ScriptedLlm
) -> None:
    result = await send(components, "   ")

    assert result.status is TurnStatus.NOOP
    assert result.outcome is TurnOutcome.NO_MATCH
    assert llm.calls == []
    assert await components.conversations.get("conv-1") is None
    assert components.runs.records == []


async def test_closed_conversation_ends_the_turn(components: Components, llm: ScriptedLlm) -> None:
    llm.queue("Hello.")
    await send(components, "Hi")
    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    conversation.closed = True
    await components.conversations.save(conversation)

    result = await send(components, "Are you there?")

    assert result.status is TurnStatus.ENDED
    assert result.outcome is TurnOutcome.RUN_ENDED
    assert len(llm.calls) == 1


async def test_provider_failure_asks_the_customer_to_retry(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(ProviderError("timeout"))

    result = await send(components, "Where is my order?")

    assert result.status is TurnStatus.FAILED
    assert result.outcome is TurnOutcome.FAILED
    assert result.reply == RETRY_REPLY
    assert result.reason == "timeout"

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.attempts == 1
    assert conversation.human_handoff_requested is False


async def test_empty_model_reply_counts_as_a_failure(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue("   ")

    result = await send(components, "Where is my order?")

    assert result.status is TurnStatus.FAILED
    assert result.reason == "model returned an empty reply"


async def test_repeated_failures_hand_off_to_a_human(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(*(ProviderError("down") for _ in range(MAX_ATTEMPTS)))

    results = [await send(components, "Hello?") for _ in range(MAX_ATTEMPTS)]

    assert [r.status for r in results[:-1]] == [TurnStatus.FAILED] * (MAX_ATTEMPTS - 1)
    assert results[-1].status is TurnStatus.HANDOFF
    assert results[-1].outcome is TurnOutcome.USER_REQUESTED_HUMAN
    assert results[-1].reply == GIVE_UP_REPLY

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.human_handoff_requested is True


async def test_a_successful_reply_resets_the_failure_count(
    components: Components, llm: ScriptedLlm
) -> None:
    llm.queue(ProviderError("blip"), "Back now.")

    await send(components, "Hello?")
    await send(components, "Hello again?")

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert conversation.attempts == 0


async def test_history_is_capped(components: Components, llm: ScriptedLlm) -> None:
    turns = HISTORY_LIMIT
    llm.queue(*(f"reply {i}" for i in range(turns)))

    for i in range(turns):
        await send(components, f"message {i}")

    conversation = await components.conversations.get("conv-1")
    assert conversation is not None
    assert len(conversation.history) == HISTORY_LIMIT
    assert conversation.history[-1].text == f"reply {turns - 1}"


async def test_conversations_are_isolated(components: Components, llm: ScriptedLlm) -> None:
    llm.queue("To A.", "To B.")

    await send(components, "Hi from A", conversation_id="a")
    await send(components, "Hi from B", conversation_id="b")

    assert len(llm.calls[1].messages) == 1
    conversation_a = await components.conversations.get("a")
    assert conversation_a is not None
    assert [m.text for m in conversation_a.history] == ["Hi from A", "To A."]
