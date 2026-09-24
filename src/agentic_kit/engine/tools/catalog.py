"""The tools an order-support agent actually needs.

Each one reads or writes through a port, so the same tool works against sample
data today and a real system later. Descriptions are written for the model: they
say when to reach for the tool, not how it is built.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...domain.crm import Order, Ticket
from ...domain.knowledge import Passage
from ...domain.models import TurnRequest
from ...domain.tools import Safety, ToolResult
from ...ports.readers import ContactReader, RecordReader, TicketWriter
from ...ports.tools import ConfirmableArgs, ToolContext
from ..knowledge.base import KnowledgeBase


class TrackOrderArgs(BaseModel):
    order_id: str | None = Field(
        default=None,
        description="The order reference. Leave empty to use the order already in this chat.",
    )


class TrackOrder:
    """Reads an order. Remembers which order, never what state it was in.

    A remembered status reads as current on a later turn, which is how an agent
    ends up describing a delivery that has since moved on.
    """

    name = "track_order"
    description = (
        "Look up the current status of a customer order. "
        "Use this before telling a customer anything about where their order is."
    )
    safety = Safety.READ
    args_model = TrackOrderArgs
    remembers = ("order_id",)

    def __init__(self, orders: RecordReader[Order]) -> None:
        self._orders = orders

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: TrackOrderArgs, context: ToolContext) -> ToolResult:
        order_id = context.resolve("order_id", args.order_id, context.request.context.order_id)
        if not order_id:
            return ToolResult.needs_input(
                "order_id", "Ask the customer for their order reference first."
            )

        order = await self._orders.get(order_id)
        if order is None:
            return ToolResult.missing(f"No order matches {order_id}.")

        return ToolResult.ok(
            f"Order {order_id} was found.", order=order.model_dump(mode="json")
        ).remembering(order_id=order_id)


class LookupTicketArgs(BaseModel):
    ticket_id: str | None = Field(
        default=None,
        description="The ticket reference. Leave empty to use the ticket already in this chat.",
    )


class LookupTicket:
    """Reads a ticket. Remembers which ticket, never its state, for the same reason."""

    name = "lookup_ticket"
    description = "Read a support ticket's status and subject."
    safety = Safety.READ
    args_model = LookupTicketArgs
    remembers = ("ticket_id",)

    def __init__(self, tickets: RecordReader[Ticket]) -> None:
        self._tickets = tickets

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: LookupTicketArgs, context: ToolContext) -> ToolResult:
        ticket_id = context.resolve("ticket_id", args.ticket_id, context.request.context.ticket_id)
        if not ticket_id:
            return ToolResult.needs_input("ticket_id", "Ask the customer which ticket they mean.")

        ticket = await self._tickets.get(ticket_id)
        if ticket is None:
            return ToolResult.missing(f"No ticket matches {ticket_id}.")

        return ToolResult.ok(
            f"Ticket {ticket_id} was found.", ticket=ticket.model_dump(mode="json")
        ).remembering(ticket_id=ticket_id)


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
    remembers = ("customer_email", "customer_phone")

    def __init__(self, contacts: ContactReader) -> None:
        self._contacts = contacts

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: LookupContactArgs, context: ToolContext) -> ToolResult:
        turn = context.request
        identities = [
            value
            for value in (
                context.resolve("customer_email", args.email, turn.context.email),
                context.resolve("customer_phone", args.phone, turn.context.phone),
                turn.customer_id,
            )
            if value
        ]
        contact = await self._contacts.find(identities)
        if contact is None:
            return ToolResult.missing("No contact record matches this customer.")

        return ToolResult.ok(
            "Contact record found.", contact=contact.model_dump(mode="json")
        ).remembering(customer_email=contact.email, customer_phone=contact.phone)


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
    remembers = ("ticket_id",)

    def __init__(self, tickets: TicketWriter) -> None:
        self._tickets = tickets

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: AddTicketNoteArgs, context: ToolContext) -> ToolResult:
        ticket_id = context.resolve("ticket_id", args.ticket_id, context.request.context.ticket_id)
        if not ticket_id:
            return ToolResult.needs_input(
                "ticket_id", "Ask the customer which ticket the note belongs to."
            )

        if not await self._tickets.add_note(ticket_id, args.note):
            return ToolResult.missing(f"No ticket matches {ticket_id}, so nothing was written.")

        return ToolResult.ok(f"The note was added to ticket {ticket_id}.").remembering(
            ticket_id=ticket_id
        )


class SearchKnowledgeArgs(BaseModel):
    question: str = Field(description="What you need to know, phrased as the customer asked it.")


class SearchKnowledge:
    """Looks things up in the policy library.

    A tool the model has to ask for, rather than passages that appear in every
    prompt. Asking is cheaper, since most turns need no policy at all, and the
    answer then arrives as a tool result, which is the one place the agent has
    been told text is material to read and not an instruction to follow.
    """

    name = "search_knowledge"
    description = (
        "Search company policy and help articles. "
        "Use this before stating any policy on returns, shipping, refunds or warranties, "
        "and quote what it gives you rather than what you remember."
    )
    safety = Safety.READ
    args_model = SearchKnowledgeArgs
    remembers = ()
    """A passage is not an identifier. Policy changes, so it is looked up each time."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self._knowledge = knowledge

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: SearchKnowledgeArgs, context: ToolContext) -> ToolResult:
        passages = await self._knowledge.find(args.question)
        if not passages:
            return ToolResult.missing(
                "Nothing in the policy library covers that. "
                "Say so rather than answering from memory."
            )
        return ToolResult.ok(
            f"Found {len(passages)} passage(s).", passages=[_quote(p) for p in passages]
        )


class FinishProcedureArgs(ConfirmableArgs):
    summary: str = Field(
        description="One line on what was settled, for the record rather than the customer."
    )


class FinishProcedure:
    """Closes off the procedure the conversation was following.

    The agent says when it is done rather than the engine inferring it from a
    reply that sounded conclusive. It is a write, so the registry makes the
    model check first, which is the "anything else I can help with?" a person
    would ask anyway.

    Closes the procedure, never the conversation. The customer can keep
    talking; the next thing they say is simply routed from scratch.
    """

    name = "finish_procedure"
    description = (
        "Mark the current procedure as finished once the customer's original request "
        "is fully settled and they have nothing further on it. "
        "Confirm with the customer before calling this."
    )
    safety = Safety.WRITE
    args_model = FinishProcedureArgs
    remembers = ()

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: FinishProcedureArgs, context: ToolContext) -> ToolResult:
        return ToolResult.done(
            "Noted as finished. Answer anything else from scratch rather than "
            "assuming it belongs to this one."
        )


def _quote(passage: Passage) -> dict[str, str]:
    """A passage as the model reads it.

    The match score is left out. It cannot be calibrated by a reader and it
    only invites an answer hedged with a number the customer cannot use.
    """
    return {
        "title": passage.chunk.title,
        "source": passage.chunk.source,
        "text": passage.chunk.text,
    }
