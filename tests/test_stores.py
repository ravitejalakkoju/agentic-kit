"""The persistence adapters obey the same read and write contract."""

from __future__ import annotations

from pathlib import Path

from agentic_kit.adapters.memory import InMemoryRunStore
from agentic_kit.adapters.sqlite import open_sqlite
from agentic_kit.domain.memory import Fact, PendingInput, WorkingMemory
from agentic_kit.domain.models import (
    ConversationState,
    FlowKind,
    Message,
    Role,
    RunRecord,
    TurnOutcome,
    TurnStatus,
)


def run(run_id: str, conversation_id: str = "conv-1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        conversation_id=conversation_id,
        agent_id="agent-1",
        sop_id="sop-1",
        kind=FlowKind.EVENT,
        event="order.shipped",
        status=TurnStatus.NOOP,
        outcome=TurnOutcome.NO_MATCH,
        reason="nothing claimed the event",
    )


async def test_in_memory_run_reads_match_what_was_appended() -> None:
    store = InMemoryRunStore()
    first = run("run-1")
    second = run("run-2", "conv-2")
    await store.append(first)
    await store.append(second)

    assert await store.get("run-1") == first
    assert await store.get("missing") is None
    assert await store.list_for("conv-1") == [first]


async def test_sqlite_round_trips_a_conversation_and_its_runs(tmp_path: Path) -> None:
    conversations, runs = open_sqlite(str(tmp_path / "agentic-kit.db"))
    conversation = ConversationState(
        conversation_id="conv-1",
        customer_id="cust-1",
        active_agent_id="agent-1",
        active_sop_id="sop-1",
        history=[
            Message(role=Role.USER, text="Where is my order?"),
            Message(role=Role.AGENT, text="It shipped."),
        ],
        memory=WorkingMemory(
            facts={
                "order_id": Fact(key="order_id", value="1001", source="track_order")
            },
            pending={
                "ticket_id": PendingInput(
                    key="ticket_id", prompt="Which ticket?", source="lookup_ticket"
                )
            },
        ),
    )
    first = run("run-1")
    second = run("run-2")

    await conversations.save(conversation)
    await runs.append(first)
    await runs.append(second)

    assert await conversations.get("conv-1") == conversation
    assert await runs.get("run-1") == first
    assert await runs.get("missing") is None
    assert await runs.list_for("conv-1") == [first, second]


async def test_a_second_sqlite_store_sees_the_first_store_s_writes(tmp_path: Path) -> None:
    path = str(tmp_path / "agentic-kit.db")
    conversations, runs = open_sqlite(path)
    conversation = ConversationState(conversation_id="conv-1", customer_id="cust-1")
    saved_run = run("run-1")
    await conversations.save(conversation)
    await runs.append(saved_run)

    restarted_conversations, restarted_runs = open_sqlite(path)

    assert await restarted_conversations.get("conv-1") == conversation
    assert await restarted_runs.get("run-1") == saved_run
