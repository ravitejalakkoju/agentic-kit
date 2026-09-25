"""The door for a turn nobody typed."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..domain.models import FlowKind, TurnContext, TurnRequest, TurnResult
from .deps import Wired

router = APIRouter(prefix="/v1", tags=["events"])


class EventBody(BaseModel):
    """Something that happened, and who it happened to."""

    event: str
    """Which event this is. A procedure has to claim the name or nothing runs."""
    conversation_id: str
    customer_id: str
    text: str
    """What happened, in a sentence, written by the system that knows.

    The agent reads it and writes its own message to the customer from it, so
    this is a note rather than a draft. It is screened on the way in like
    anything else the engine did not write itself.
    """
    context: TurnContext = Field(default_factory=TurnContext)
    """Ids the event already knows. The same collectors that serve a question
    look the records up from here, so a payload needs no handling of its own."""


@router.post("/events", summary="Tell the agent something happened")
async def create_event(body: EventBody, wired: Wired) -> TurnResult:
    """Runs the turn and answers with it.

    Nothing is queued. Surviving a restart, retrying, and not delivering
    twice are a queue's problems, and none of them are this endpoint's until
    something is actually waiting on one.
    """
    return await wired.engine.handle(
        TurnRequest(
            conversation_id=body.conversation_id,
            customer_id=body.customer_id,
            text=body.text,
            kind=FlowKind.EVENT,
            event=body.event,
            context=body.context,
        )
    )
