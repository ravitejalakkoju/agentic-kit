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
        "Has my order shipped yet?",
        "Can I change the address on my order?",
        "I want to cancel an order I placed",
    ],
    events=["order.shipped"],
    catch_all=True,
    personality=Personality(
        name="Ava",
        identity="a support teammate for an online store",
        tone="warm, direct, never padded with filler",
    ),
    example_responses=[
        "Thanks for waiting. Order #1001 left our warehouse today and should reach you by Friday.",
    ],
)

RETURNS_SOP = SopDefinition(
    sop_id="returns_and_refunds",
    agent_id="returns_agent",
    description="sending something back, refunds and exchanges",
    instructions=(
        "1. Find out which item the customer wants to send back, and why.\n"
        "2. Check the returns policy before quoting a window or a charge.\n"
        "3. Say what happens next and when the money lands.\n"
        "4. Offer an exchange where a return would leave them without the thing they wanted."
    ),
    examples=[
        "I want to return this jacket",
        "How do I get a refund?",
        "Can I exchange it for a different size?",
        "When will my money come back?",
    ],
    events=["refund.issued"],
    personality=Personality(
        name="Rohan",
        identity="a returns teammate for an online store",
        tone="unhurried and practical, never defensive about a refund",
    ),
    example_responses=[
        "You have 30 days from delivery to send it back, and the refund lands three "
        "to five working days after it reaches us.",
    ],
)

DELIVERY_SOP = SopDefinition(
    sop_id="delivery_issue",
    agent_id="delivery_agent",
    description="a parcel that is lost, damaged or delivered to the wrong place",
    instructions=(
        "1. Establish what the tracking says before offering a theory.\n"
        "2. Check the policy for how long a parcel must be missing before it counts as lost.\n"
        "3. Ask them to check with neighbours and the local depot if it is inside that window.\n"
        "4. Raise it with a human teammate once the window has passed."
    ),
    examples=[
        "My parcel never arrived",
        "The box turned up damaged",
        "It says delivered but I do not have it",
        "The courier left it with a neighbour",
    ],
    events=["delivery.failed"],
    personality=Personality(
        name="Mei",
        identity="a delivery teammate for an online store",
        tone="calm and specific, apologises once and then fixes things",
    ),
    example_responses=[
        "Tracking says it was handed over on Tuesday. Before I raise it as lost, "
        "could you check with your neighbours and your local depot?",
    ],
)

DEFAULT_SOPS = [SUPPORT_SOP, RETURNS_SOP, DELIVERY_SOP]

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
