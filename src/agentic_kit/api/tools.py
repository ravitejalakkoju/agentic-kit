"""The tool catalog, so what the agent can do is readable without the source."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..domain.tools import Safety
from .deps import Wired

router = APIRouter(prefix="/v1", tags=["tools"])


class ToolView(BaseModel):
    name: str
    description: str
    safety: Safety
    parameters: dict[str, Any]
    remembers: list[str]
    """What this tool may write into memory, and so into every later prompt."""


@router.get("/tools", summary="List the tools the agent can use")
async def list_tools(wired: Wired) -> list[ToolView]:
    return [
        ToolView(
            name=tool.name,
            description=tool.description,
            safety=tool.safety,
            parameters=tool.parameters,
            remembers=list(tool.remembers),
        )
        for tool in wired.tools.catalog()
    ]
