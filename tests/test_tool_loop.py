"""The loop that trades tool calls with the model until it answers."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from agentic_kit.composition import build_crm, build_prompts, build_tools
from agentic_kit.domain.models import ConversationState, TurnContext, TurnRequest
from agentic_kit.domain.tools import Safety, ToolResult
from agentic_kit.engine.guardrails import DEFAULT_PROFILE, GuardrailResponder, Guardrails
from agentic_kit.engine.guardrails.detectors import default_detectors
from agentic_kit.engine.runtime.text_runtime import (
    GAVE_UP_REPLY,
    MAX_TOOL_ROUNDS,
    TextRuntime,
)
from agentic_kit.engine.tools.registry import ToolRegistry
from agentic_kit.ports.tools import ToolContext
from agentic_kit.seed import SUPPORT_SOP

from .fakes import ScriptedLlm, calls, tool_call

REQUEST = TurnRequest(
    conversation_id="conv-1",
    customer_id="cust-1",
    text="Where is my order?",
    context=TurnContext(order_id="1001"),
)
CONVERSATION = ConversationState(conversation_id="conv-1", customer_id="cust-1")


def runtime_with(llm: ScriptedLlm, tools: ToolRegistry | None = None) -> TextRuntime:
    return TextRuntime(
        llm,
        build_prompts(),
        Guardrails(default_detectors(), DEFAULT_PROFILE),
        GuardrailResponder(),
        tools if tools is not None else build_tools(build_crm()),
    )


async def run(runtime: TextRuntime):
    return await runtime.run(sop=SUPPORT_SOP, conversation=CONVERSATION, request=REQUEST)


async def test_a_turn_with_no_tool_call_is_one_round(llm: ScriptedLlm) -> None:
    llm.queue("It has shipped.")

    reply = await run(runtime_with(llm))

    assert reply.text == "It has shipped."
    assert reply.tool_calls == ()
    assert len(llm.calls) == 1


async def test_a_tool_result_is_fed_back_and_answered(llm: ScriptedLlm) -> None:
    llm.queue(calls(tool_call("track_order", order_id="1001")), "Your order has shipped.")

    reply = await run(runtime_with(llm))

    assert reply.text == "Your order has shipped."
    assert [call.name for call in reply.tool_calls] == ["track_order"]

    second = llm.calls[1]
    [exchange] = second.exchanges
    assert '"status": "shipped"' in exchange.results[0][1]


async def test_the_model_can_use_two_rounds(llm: ScriptedLlm) -> None:
    llm.queue(
        calls(tool_call("lookup_contact")),
        calls(tool_call("track_order", order_id="1001")),
        "Priya, your order has shipped.",
    )

    reply = await run(runtime_with(llm))

    assert reply.text == "Priya, your order has shipped."
    assert [call.name for call in reply.tool_calls] == ["lookup_contact", "track_order"]
    assert len(llm.calls[2].exchanges) == 2


async def test_several_tools_in_one_round_all_run(llm: ScriptedLlm) -> None:
    llm.queue(
        calls(tool_call("track_order", order_id="1001"), tool_call("lookup_contact")),
        "Here is everything.",
    )

    reply = await run(runtime_with(llm))

    assert len(reply.tool_calls) == 2
    [exchange] = llm.calls[1].exchanges
    assert len(exchange.results) == 2


async def test_tools_in_one_round_run_at_the_same_time(llm: ScriptedLlm) -> None:
    class SlowArgs(BaseModel):
        pass

    class SlowTool:
        name = "slow"
        description = "Takes its time."
        safety = Safety.READ
        args_model = SlowArgs
        remembers = ()

        def is_available(self, request: TurnRequest) -> bool:
            return True

        async def execute(self, args: SlowArgs, context: ToolContext) -> ToolResult:
            await asyncio.sleep(0.05)
            return ToolResult.ok("done")

    llm.queue(calls(tool_call("slow"), tool_call("slow"), tool_call("slow")), "All done.")
    runtime = runtime_with(llm, ToolRegistry([SlowTool()]))

    started = asyncio.get_running_loop().time()
    await run(runtime)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.12, "three 50ms tools should overlap, not queue up"


async def test_a_failing_tool_still_reaches_an_answer(llm: ScriptedLlm) -> None:
    llm.queue(calls(tool_call("track_order", order_id="does-not-exist")), "I could not find it.")

    reply = await run(runtime_with(llm))

    assert reply.text == "I could not find it."
    [exchange] = llm.calls[1].exchanges
    assert '"status": "not_found"' in exchange.results[0][1]


async def test_an_unknown_tool_is_reported_back_to_the_model(llm: ScriptedLlm) -> None:
    llm.queue(calls(tool_call("no_such_tool")), "Sorry, I cannot do that.")

    reply = await run(runtime_with(llm))

    assert reply.text == "Sorry, I cannot do that."
    [exchange] = llm.calls[1].exchanges
    assert '"status": "invalid_input"' in exchange.results[0][1]


async def test_a_write_tool_needs_confirming_before_it_runs(llm: ScriptedLlm) -> None:
    crm = build_crm()
    llm.queue(
        calls(tool_call("add_ticket_note", note="chased", ticket_id="T-501")),
        "Shall I add a note to your ticket?",
    )

    await run(runtime_with(llm, build_tools(crm)))

    assert crm.ticket_notes.notes == {}
    [exchange] = llm.calls[1].exchanges
    assert '"status": "needs_confirmation"' in exchange.results[0][1]


async def test_a_confirmed_write_goes_through(llm: ScriptedLlm) -> None:
    crm = build_crm()
    llm.queue(
        calls(tool_call("add_ticket_note", note="chased", ticket_id="T-501", confirmed=True)),
        "I have added the note.",
    )

    await run(runtime_with(llm, build_tools(crm)))

    assert crm.ticket_notes.notes["T-501"] == ["chased"]


async def test_the_loop_gives_up_and_asks_once_without_tools(llm: ScriptedLlm) -> None:
    llm.queue(*(calls(tool_call("lookup_contact")) for _ in range(MAX_TOOL_ROUNDS)))
    llm.queue("Here is what I have so far.")

    reply = await run(runtime_with(llm))

    assert reply.text == "Here is what I have so far."
    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert llm.calls[-1].tools == (), "the last ask offers no tools"
    assert len(reply.tool_calls) == MAX_TOOL_ROUNDS


async def test_a_model_that_never_stops_gets_a_safe_reply(llm: ScriptedLlm) -> None:
    llm.queue(*(calls(tool_call("lookup_contact")) for _ in range(MAX_TOOL_ROUNDS + 1)))

    reply = await run(runtime_with(llm))

    assert reply.text == GAVE_UP_REPLY


async def test_tool_traffic_never_enters_the_conversation_history(llm: ScriptedLlm) -> None:
    llm.queue(calls(tool_call("track_order", order_id="1001")), "It has shipped.")

    await run(runtime_with(llm))

    assert all(request.messages == llm.calls[0].messages for request in llm.calls)


async def test_the_prompt_mentions_tool_rules_when_tools_are_offered(llm: ScriptedLlm) -> None:
    llm.queue("It has shipped.")

    await run(runtime_with(llm))

    assert "Tool Use:" in llm.calls[0].system


async def test_no_tool_rules_when_there_are_no_tools(llm: ScriptedLlm) -> None:
    llm.queue("It has shipped.")

    await run(runtime_with(llm, ToolRegistry()))

    assert "Tool Use:" not in llm.calls[0].system
    assert llm.calls[0].tools == ()
