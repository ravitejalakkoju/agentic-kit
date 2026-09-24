"""The four tools, against the seeded sample data."""

from __future__ import annotations

import pytest

from agentic_kit.composition import Crm, build_crm
from agentic_kit.domain.models import TurnContext, TurnRequest
from agentic_kit.domain.tools import ToolStatus
from agentic_kit.engine.tools.catalog import (
    AddTicketNote,
    AddTicketNoteArgs,
    LookupContact,
    LookupContactArgs,
    LookupTicket,
    LookupTicketArgs,
    TrackOrder,
    TrackOrderArgs,
)
from agentic_kit.ports.tools import ToolContext


@pytest.fixture
def crm() -> Crm:
    return build_crm()


def context(**turn_context: str) -> ToolContext:
    return ToolContext(
        request=TurnRequest(
            conversation_id="conv-1",
            customer_id="cust-1",
            text="hi",
            context=TurnContext(**turn_context),
        )
    )


async def test_track_order_finds_a_seeded_order(crm: Crm) -> None:
    result = await TrackOrder(crm.orders).execute(TrackOrderArgs(order_id="1001"), context())

    assert result.succeeded
    assert result.data["order"]["status"] == "shipped"


async def test_track_order_falls_back_to_the_order_in_the_chat(crm: Crm) -> None:
    result = await TrackOrder(crm.orders).execute(TrackOrderArgs(), context(order_id="1001"))

    assert result.succeeded


async def test_track_order_asks_for_a_reference_when_there_is_none(crm: Crm) -> None:
    result = await TrackOrder(crm.orders).execute(TrackOrderArgs(), context())

    assert result.status is ToolStatus.INVALID_INPUT


async def test_track_order_reports_a_missing_order(crm: Crm) -> None:
    result = await TrackOrder(crm.orders).execute(TrackOrderArgs(order_id="nope"), context())

    assert result.status is ToolStatus.NOT_FOUND


async def test_lookup_ticket_finds_a_seeded_ticket(crm: Crm) -> None:
    result = await LookupTicket(crm.tickets).execute(LookupTicketArgs(), context(ticket_id="T-501"))

    assert result.succeeded
    assert result.data["ticket"]["status"] == "open"


async def test_lookup_contact_uses_the_customer_on_the_turn(crm: Crm) -> None:
    result = await LookupContact(crm.contacts).execute(LookupContactArgs(), context())

    assert result.succeeded
    assert result.data["contact"]["name"] == "Priya Sharma"


async def test_lookup_contact_can_search_by_email(crm: Crm) -> None:
    result = await LookupContact(crm.contacts).execute(
        LookupContactArgs(email="priya@example.com"), context()
    )

    assert result.succeeded


async def test_lookup_contact_reports_nobody(crm: Crm) -> None:
    request = TurnRequest(conversation_id="c", customer_id="stranger", text="hi")

    result = await LookupContact(crm.contacts).execute(
        LookupContactArgs(), ToolContext(request=request)
    )

    assert result.status is ToolStatus.NOT_FOUND


async def test_add_ticket_note_writes_the_note(crm: Crm) -> None:
    args = AddTicketNoteArgs(note="Customer chased delivery.", confirmed=True)

    result = await AddTicketNote(crm.ticket_notes).execute(args, context(ticket_id="T-501"))

    assert result.succeeded
    assert crm.ticket_notes.notes["T-501"] == ["Customer chased delivery."]


async def test_add_ticket_note_reports_a_missing_ticket(crm: Crm) -> None:
    args = AddTicketNoteArgs(note="x", ticket_id="T-999", confirmed=True)

    result = await AddTicketNote(crm.ticket_notes).execute(args, context())

    assert result.status is ToolStatus.NOT_FOUND
    assert crm.ticket_notes.notes == {}
