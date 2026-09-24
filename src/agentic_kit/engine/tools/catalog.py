"""The tools an order-support agent actually needs.

Each one reads or writes through a port, so the same tool works against sample
data today and a real system later. Descriptions are written for the model: they
say when to reach for the tool, not how it is built.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...domain.crm import Order, Ticket
from ...domain.models import TurnRequest
from ...domain.tools import Safety, ToolResult, ToolStatus
from ...ports.readers import ContactReader, RecordReader, TicketWriter
from ...ports.tools import ConfirmableArgs, ToolContext


class TrackOrderArgs(BaseModel):
    order_id: str | None = Field(
        default=None,
        description="The order reference. Leave empty to use the order already in this chat.",
    )


class TrackOrder:
    name = "track_order"
    description = (
        "Look up the current status of a customer order. "
        "Use this before telling a customer anything about where their order is."
    )
    safety = Safety.READ
    args_model = TrackOrderArgs

    def __init__(self, orders: RecordReader[Order]) -> None:
        self._orders = orders

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: TrackOrderArgs, context: ToolContext) -> ToolResult:
        order_id = args.order_id or context.request.context.order_id
        if not order_id:
            return ToolResult.rejected(
                ToolStatus.INVALID_INPUT, "Ask the customer for their order reference first."
            )

        order = await self._orders.get(order_id)
        if order is None:
            return ToolResult.missing(f"No order matches {order_id}.")

        return ToolResult.ok(f"Order {order_id} was found.", order=order.model_dump(mode="json"))


class LookupTicketArgs(BaseModel):
    ticket_id: str | None = Field(
        default=None,
        description="The ticket reference. Leave empty to use the ticket already in this chat.",
    )


class LookupTicket:
    name = "lookup_ticket"
    description = "Read a support ticket's status and subject."
    safety = Safety.READ
    args_model = LookupTicketArgs

    def __init__(self, tickets: RecordReader[Ticket]) -> None:
        self._tickets = tickets

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: LookupTicketArgs, context: ToolContext) -> ToolResult:
        ticket_id = args.ticket_id or context.request.context.ticket_id
        if not ticket_id:
            return ToolResult.rejected(
                ToolStatus.INVALID_INPUT, "Ask the customer which ticket they mean."
            )

        ticket = await self._tickets.get(ticket_id)
        if ticket is None:
            return ToolResult.missing(f"No ticket matches {ticket_id}.")

        return ToolResult.ok(
            f"Ticket {ticket_id} was found.", ticket=ticket.model_dump(mode="json")
        )


class LookupContactArgs(BaseModel):
    email: str | None = Field(default=None, description="The customer's email address.")
    phone: str | None = Field(default=None, description="The customer's phone number.")


class LookupContact:
    name = "lookup_contact"
    description = (
        "Find the customer's contact record. "
        "Use this when you need their name, tags, or the details on file."
    )
    safety = Safety.READ
    args_model = LookupContactArgs

    def __init__(self, contacts: ContactReader) -> None:
        self._contacts = contacts

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: LookupContactArgs, context: ToolContext) -> ToolResult:
        turn = context.request
        identities = [
            value
            for value in (
                args.email,
                args.phone,
                turn.customer_id,
                turn.context.email,
                turn.context.phone,
            )
            if value
        ]
        contact = await self._contacts.find(identities)
        if contact is None:
            return ToolResult.missing("No contact record matches this customer.")

        return ToolResult.ok("Contact record found.", contact=contact.model_dump(mode="json"))


class AddTicketNoteArgs(ConfirmableArgs):
    note: str = Field(description="What to record on the ticket, in one or two sentences.")
    ticket_id: str | None = Field(
        default=None,
        description="The ticket reference. Leave empty to use the ticket already in this chat.",
    )


class AddTicketNote:
    name = "add_ticket_note"
    description = (
        "Add a note to a support ticket so a human teammate can pick the conversation up. "
        "This changes the ticket, so confirm with the customer before calling it."
    )
    safety = Safety.WRITE
    args_model = AddTicketNoteArgs

    def __init__(self, tickets: TicketWriter) -> None:
        self._tickets = tickets

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: AddTicketNoteArgs, context: ToolContext) -> ToolResult:
        ticket_id = args.ticket_id or context.request.context.ticket_id
        if not ticket_id:
            return ToolResult.rejected(
                ToolStatus.INVALID_INPUT, "Ask the customer which ticket the note belongs to."
            )

        if not await self._tickets.add_note(ticket_id, args.note):
            return ToolResult.missing(f"No ticket matches {ticket_id}, so nothing was written.")

        return ToolResult.ok(f"The note was added to ticket {ticket_id}.")
