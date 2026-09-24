"""Shows the prompt a turn would be sent with, without calling the model."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..domain.models import TurnRequest
from .deps import Wired

router = APIRouter(prefix="/v1", tags=["prompts"])


class SectionView(BaseModel):
    key: str
    title: str
    priority: int
    content: str


class PromptPreview(BaseModel):
    sop_id: str
    sections: list[SectionView]
    system: str


@router.post("/prompts/preview", summary="Preview the system prompt for a turn")
async def preview_prompt(body: TurnRequest, wired: Wired) -> PromptPreview:
    sop = await wired.catalog.match(body.text)
    if sop is None:
        raise HTTPException(status_code=404, detail="no procedure matched")

    # The preview is only worth having if it is the prompt the model would get,
    # so it reads the same remembered facts a real turn would.
    conversation = await wired.conversations.get(body.conversation_id)
    prompt = await wired.prompts.build(
        sop=sop,
        request=body,
        tools=wired.tools.definitions(body),
        memory=conversation.memory if conversation else None,
    )
    return PromptPreview(
        sop_id=sop.sop_id,
        sections=[
            SectionView(key=s.key, title=s.title, priority=s.priority, content=s.content)
            for s in prompt.sections
        ],
        system=prompt.system,
    )
