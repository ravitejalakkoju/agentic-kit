from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..domain.models import ConversationState, FlowKind, SopDefinition, TurnRequest, TurnResult
from .deps import Wired

router = APIRouter(prefix="/v1", tags=["conversation"])


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
