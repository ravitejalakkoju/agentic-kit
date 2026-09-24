"""Records the engine reads from the customer's systems. Read-only snapshots."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Contact(BaseModel):
    contact_id: str
    name: str
    email: str | None = None
    phone: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def identities(self) -> set[str]:
        return {value for value in (self.contact_id, self.email, self.phone) if value}


class Order(BaseModel):
    order_id: str
    status: str
    amount: float
    currency: str
    payment_status: str
    customer_name: str


class Ticket(BaseModel):
    ticket_id: str
    status: str
    subject: str
