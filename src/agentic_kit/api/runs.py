"""Read the run records written by the engine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..domain.models import RunRecord
from .deps import Wired

router = APIRouter(prefix="/v1", tags=["runs"])


@router.get("/conversations/{conversation_id}/runs", summary="List a conversation's runs")
async def list_runs(conversation_id: str, wired: Wired) -> list[RunRecord]:
    return await wired.runs.list_for(conversation_id)


@router.get("/runs/{run_id}", summary="Inspect one run")
async def get_run(run_id: str, wired: Wired) -> RunRecord:
    run = await wired.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run
