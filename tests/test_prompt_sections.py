from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentic_kit.domain.crm import Order
from agentic_kit.domain.memory import Fact, PendingInput, WorkingMemory
from agentic_kit.domain.models import Personality, TurnContext, TurnRequest
from agentic_kit.domain.tools import Safety, ToolDefinition
from agentic_kit.engine.guardrails.profile import DEFAULT_RULES
from agentic_kit.engine.prompt.builder import PromptBuilder
from agentic_kit.engine.prompt.context import ContextBag, ContextPipeline, Resource
from agentic_kit.engine.prompt.policy import DEFAULT_RESPONSE_STRATEGY, PromptPolicy
from agentic_kit.engine.prompt.sections import (
    DEFAULT_SECTIONS,
    ExampleResponsesSection,
    GuardrailSection,
    PersonalitySection,
    PromptInput,
    RuntimeContextSection,
    SopSection,
    ToolCatalogSection,
    WorkingMemorySection,
)
from agentic_kit.seed import SAMPLE_ORDERS, SUPPORT_SOP

REQUEST = TurnRequest(
    conversation_id="conv-1",
    customer_id="cust-1",
    text="Where is my order?",
    context=TurnContext(email="priya@example.com", order_id="1001"),
)


A_TOOL = ToolDefinition(
    name="track_order", description="Track an order.", parameters={}, safety=Safety.READ
)


def memory_with(
    facts: dict[str, str] | None = None, waiting: dict[str, str] | None = None
) -> WorkingMemory:
    return WorkingMemory(
        facts={
            key: Fact(key=key, value=value, source="track_order")
            for key, value in (facts or {}).items()
        },
        pending={
            key: PendingInput(key=key, prompt=text, source="track_order")
            for key, text in (waiting or {}).items()
        },
    )


REMEMBERED = memory_with({"order_id": "1001"})


def prompt_input(
    *,
    sop=SUPPORT_SOP,
    context: ContextBag | None = None,
    rules: tuple[str, ...] = DEFAULT_RULES,
    tools: tuple[ToolDefinition, ...] = (A_TOOL,),
    memory: WorkingMemory = REMEMBERED,
) -> PromptInput:
    policy = PromptPolicy.for_turn(sop, REQUEST, rules)
    return PromptInput(
        request=REQUEST,
        policy=policy,
        context=context or ContextBag(),
        tools=tools,
        memory=memory,
    )


def test_default_sections_have_unique_keys_and_priorities() -> None:
    assert len({s.key for s in DEFAULT_SECTIONS}) == len(DEFAULT_SECTIONS)
    assert len({s.priority for s in DEFAULT_SECTIONS}) == len(DEFAULT_SECTIONS)


def test_policy_falls_back_to_the_default_response_strategy() -> None:
    assert PromptPolicy.for_turn(SUPPORT_SOP, REQUEST).response_strategy == (
        DEFAULT_RESPONSE_STRATEGY
    )


def test_sop_response_strategy_wins_over_the_default() -> None:
    sop = SUPPORT_SOP.model_copy(update={"response_strategy": "Be brief."})

    assert PromptPolicy.for_turn(sop, REQUEST).response_strategy == "Be brief."


def test_personality_renders_each_tone_as_a_bullet() -> None:
    content = PersonalitySection().render(prompt_input())

    assert content is not None
    assert "- warm\n- direct\n- never padded with filler" in content


def test_personality_without_tone_is_left_out() -> None:
    sop = SUPPORT_SOP.model_copy(
        update={"personality": Personality(name="Ava", identity="support", tone=" , ")}
    )

    assert PersonalitySection().render(prompt_input(sop=sop)) is None


def test_guardrails_are_left_out_when_there_are_no_rules() -> None:
    assert GuardrailSection().render(prompt_input(rules=())) is None


def test_sop_section_carries_the_instructions() -> None:
    assert SUPPORT_SOP.instructions in SopSection().render(prompt_input())


def test_tool_rules_are_left_out_when_no_tool_is_offered() -> None:
    assert ToolCatalogSection().render(prompt_input(tools=())) is None


def test_tool_rules_do_not_repeat_the_schemas() -> None:
    content = ToolCatalogSection().render(prompt_input())

    assert content is not None
    assert "track_order" not in content


def test_examples_are_left_out_when_the_sop_has_none() -> None:
    sop = SUPPORT_SOP.model_copy(update={"example_responses": []})

    assert ExampleResponsesSection().render(prompt_input(sop=sop)) is None


def test_runtime_context_shows_the_customer_without_lookup_ids() -> None:
    content = RuntimeContextSection().render(prompt_input())

    assert '"email": "priya@example.com"' in content
    assert "order_id" not in content
    assert "Contact:" not in content


def test_runtime_context_renders_each_resource_as_a_json_block() -> None:
    [order] = SAMPLE_ORDERS
    context = ContextBag(resources=(Resource("order", order),))

    content = RuntimeContextSection().render(prompt_input(context=context))

    assert "Order:\n```json" in content
    assert '"status": "shipped"' in content


def test_working_memory_names_the_tool_behind_each_fact() -> None:
    content = WorkingMemorySection().render(prompt_input())

    assert content is not None
    assert "- order_id: 1001 (from track_order)" in content


def test_working_memory_is_left_out_when_nothing_is_known() -> None:
    assert WorkingMemorySection().render(prompt_input(memory=WorkingMemory())) is None


def test_working_memory_repeats_the_tools_own_words_for_what_is_missing() -> None:
    memory = memory_with(waiting={"order_id": "Ask the customer for their order reference."})

    content = WorkingMemorySection().render(prompt_input(memory=memory))

    assert content is not None
    assert "Still waiting on:\n- Ask the customer for their order reference." in content
    assert "Known from this conversation" not in content


def test_the_superseded_archive_is_never_shown_to_the_model() -> None:
    memory = memory_with({"order_id": "1002"})
    memory.superseded = [Fact(key="order_id", value="1001", source="track_order")]

    content = WorkingMemorySection().render(prompt_input(memory=memory))

    assert content is not None
    assert "1002" in content
    assert "1001" not in content


@dataclass
class FixedSection:
    key: str
    priority: int
    body: str | None
    title: str = "T"

    def render(self, prompt: PromptInput) -> str | None:
        return self.body


async def test_builder_orders_by_priority_and_drops_empty_sections() -> None:
    sections = [
        FixedSection("late", 50, "b"),
        FixedSection("empty", 20, None),
        FixedSection("early", 10, "a"),
    ]
    builder = PromptBuilder(ContextPipeline([]), sections)

    prompt = await builder.build(sop=SUPPORT_SOP, request=REQUEST)

    assert [s.key for s in prompt.sections] == ["early", "late"]
    assert prompt.system == "T:\na\n\nT:\nb"


@pytest.mark.parametrize("section", DEFAULT_SECTIONS, ids=lambda s: s.key)
def test_every_default_section_renders_for_the_seeded_sop(section) -> None:
    order = Order(
        order_id="1",
        status="new",
        amount=1,
        currency="INR",
        payment_status="paid",
        customer_name="x",
    )
    context = ContextBag(resources=(Resource("order", order),))

    assert section.render(prompt_input(context=context))
