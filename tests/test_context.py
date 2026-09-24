from __future__ import annotations

from collections.abc import Iterable

import pytest

from agentic_kit.adapters.sample_crm import InMemoryContactReader, InMemoryRecordReader
from agentic_kit.domain.crm import Contact
from agentic_kit.domain.memory import Fact, WorkingMemory
from agentic_kit.domain.models import TurnContext, TurnRequest
from agentic_kit.engine.prompt.context import (
    ContactCollector,
    ContextBag,
    ContextPipeline,
    ResourceCollector,
)
from agentic_kit.seed import SAMPLE_CONTACTS, SAMPLE_ORDERS

BLANK = WorkingMemory()


def turn(customer_id: str = "unknown", **context: str) -> TurnRequest:
    return TurnRequest(
        conversation_id="conv-1",
        customer_id=customer_id,
        text="hi",
        context=TurnContext(**context),
    )


def remembering(**facts: str) -> WorkingMemory:
    return WorkingMemory(
        facts={key: Fact(key=key, value=value, source="a_tool") for key, value in facts.items()}
    )


@pytest.fixture
def contacts() -> ContactCollector:
    return ContactCollector(InMemoryContactReader(SAMPLE_CONTACTS))


@pytest.fixture
def orders() -> ResourceCollector:
    reader = InMemoryRecordReader(SAMPLE_ORDERS, id_of=lambda order: order.order_id)
    return ResourceCollector("order", reader, id_of=lambda ctx: ctx.order_id, fact_key="order_id")


@pytest.mark.parametrize(
    "request_",
    [turn("cust-1"), turn(email="priya@example.com"), turn(phone="+919800000001")],
    ids=["customer_id", "email", "phone"],
)
async def test_contact_is_found_by_any_identity(
    contacts: ContactCollector, request_: TurnRequest
) -> None:
    bag = await contacts.collect(request_, BLANK)

    assert bag.contact is not None
    assert bag.contact.name == "Priya Sharma"


async def test_unknown_customer_has_no_contact(contacts: ContactCollector) -> None:
    assert (await contacts.collect(turn(), BLANK)).contact is None


async def test_order_is_attached_when_the_turn_names_one(orders: ResourceCollector) -> None:
    bag = await orders.collect(turn(order_id="1001"), BLANK)

    [resource] = bag.resources
    assert resource.kind == "order"
    assert resource.record.model_dump()["status"] == "shipped"


@pytest.mark.parametrize("context", [{}, {"order_id": "does-not-exist"}])
async def test_missing_or_unknown_order_adds_nothing(
    orders: ResourceCollector, context: dict[str, str]
) -> None:
    assert (await orders.collect(turn(**context), BLANK)).resources == ()


class BrokenContactReader:
    async def find(self, identities: Iterable[str]) -> Contact | None:
        raise ConnectionError("crm is down")


async def test_a_failing_collector_is_skipped_not_fatal(orders: ResourceCollector) -> None:
    pipeline = ContextPipeline([ContactCollector(BrokenContactReader()), orders])

    bag = await pipeline.collect(turn("cust-1", order_id="1001"), BLANK)

    assert bag.contact is None
    assert [resource.kind for resource in bag.resources] == ["order"]


async def test_pipeline_merges_every_collector(
    contacts: ContactCollector, orders: ResourceCollector
) -> None:
    bag = await ContextPipeline([contacts, orders]).collect(turn("cust-1", order_id="1001"), BLANK)

    assert bag.contact is not None
    assert len(bag.resources) == 1


def test_merge_keeps_the_earlier_contact_when_the_later_has_none() -> None:
    contact = Contact(contact_id="c", name="C")

    assert ContextBag(contact=contact).merge(ContextBag()).contact == contact


async def test_the_order_named_only_by_memory_is_still_fetched(orders: ResourceCollector) -> None:
    bag = await orders.collect(turn(), remembering(order_id="1001"))

    [resource] = bag.resources
    assert resource.record.model_dump()["order_id"] == "1001"


async def test_the_turn_beats_memory_when_both_name_an_order(orders: ResourceCollector) -> None:
    bag = await orders.collect(turn(order_id="1001"), remembering(order_id="does-not-exist"))

    [resource] = bag.resources
    assert resource.record.model_dump()["order_id"] == "1001"


async def test_the_record_is_read_again_rather_than_remembered(orders: ResourceCollector) -> None:
    """Remembering the id and re-reading the record is how status stays current."""
    memory = remembering(order_id="1001")

    first = await orders.collect(turn(), memory)
    second = await orders.collect(turn(), memory)

    assert first.resources[0].record == second.resources[0].record
    assert memory.facts.keys() == {"order_id"}, "the record itself was never stored"


async def test_a_remembered_phone_finds_the_contact(contacts: ContactCollector) -> None:
    bag = await contacts.collect(turn(), remembering(customer_phone="+919800000001"))

    assert bag.contact is not None
    assert bag.contact.name == "Priya Sharma"
