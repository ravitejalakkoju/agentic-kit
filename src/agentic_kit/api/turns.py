from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from ..composition import Components
from ..domain.models import ConversationState, FlowKind, SopDefinition, TurnRequest, TurnResult

router = APIRouter(prefix="/v1", tags=["conversation"])


def components(request: Request) -> Components:
    return request.app.state.components


Wired = Annotated[Components, Depends(components)]


@router.post("/turns", summary="Run one conversation turn")
async def create_turn(body: TurnRequest, wired: Wired) -> TurnResult:
    return await wired.engine.handle(FlowKind.CONVERSATION, body)


@router.get("/conversations/{conversation_id}", summary="Inspect persisted state")
async def get_conversation(conversation_id: str, wired: Wired) -> ConversationState:
    conversation = await wired.conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.get("/sops", summary="List the seeded procedures")
async def list_sops(wired: Wired) -> list[SopDefinition]:
    return await wired.catalog.list_all()
