"""Starting data, so the first request through Swagger actually routes."""

from __future__ import annotations

from .domain.crm import Contact, Order, Ticket
from .domain.knowledge import Document
from .domain.models import Personality, SopDefinition

SUPPORT_SOP = SopDefinition(
    sop_id="order_support",
    agent_id="support_agent",
    description="questions about an existing order",
    instructions=(
        "1. Greet the customer once, then get to the point.\n"
        "2. Ask for the order reference if you do not have it.\n"
        "3. Explain the current status in one or two sentences.\n"
        "4. Offer a next step, and say when a human will take over if you cannot help."
    ),
    examples=[
        "Where is my order?",
        "My delivery is late",
        "Can I change the address on my order?",
    ],
    personality=Personality(
        name="Ava",
        identity="a support teammate for an online store",
        tone="warm, direct, never padded with filler",
    ),
    example_responses=[
        "Thanks for waiting. Order #1001 left our warehouse today and should reach you by Friday.",
    ],
)

DEFAULT_SOPS = [SUPPORT_SOP]

SAMPLE_CONTACTS = [
    Contact(
        contact_id="cust-1",
        name="Priya Sharma",
        email="priya@example.com",
        phone="+919800000001",
        tags=["vip"],
    ),
]

SAMPLE_ORDERS = [
    Order(
        order_id="1001",
        status="shipped",
        amount=1499.0,
        currency="INR",
        payment_status="paid",
        customer_name="Priya Sharma",
    ),
]

SAMPLE_TICKETS = [
    Ticket(ticket_id="T-501", status="open", subject="Late delivery for order 1001"),
]

SAMPLE_DOCUMENTS = [
    Document(
        document_id="doc-returns",
        title="Returns and refunds",
        source="handbook/returns.md",
        text=(
            "Most items can be returned within 30 days of delivery. The item has to be "
            "unused and in its original packaging, and the order reference has to be on "
            "the return note.\n\n"
            "Refunds go back to the original payment method and take three to five "
            "working days to appear once the warehouse has confirmed the return. The "
            "original delivery charge is only refunded when the item arrived damaged or "
            "was not what the customer ordered.\n\n"
            "Personalised items and gift cards cannot be returned at all. Earrings and "
            "underwear can be returned only while the hygiene seal is intact."
        ),
    ),
    Document(
        document_id="doc-delivery",
        title="Delivery and shipping",
        source="handbook/delivery.md",
        text=(
            "Standard delivery takes three to five working days. Express delivery "
            "arrives the next working day for orders placed before 2pm.\n\n"
            "Standard delivery is free on orders over fifty pounds and costs three "
            "pounds ninety-five otherwise. Express delivery is six pounds ninety-five "
            "whatever the order is worth.\n\n"
            "An order counts as lost only once fifteen working days have passed since "
            "dispatch. Before that, ask the customer to check with their neighbours and "
            "their local depot, because most parcels reported missing turn up there."
        ),
    ),
]
