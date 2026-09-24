from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import graph, turns
from .composition import build
from .errors import EngineError, ProviderError
from .settings import load_settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.settings = settings
    app.state.components = build(settings)
    yield


app = FastAPI(
    title="agentic-kit",
    version="0.1.0",
    summary="A conversation engine you can read: graph in, reply out.",
    lifespan=lifespan,
)
app.include_router(turns.router)
app.include_router(graph.router)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.exception_handler(ProviderError)
async def handle_provider_error(_: Request, error: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(error)})


@app.exception_handler(EngineError)
async def handle_engine_error(_: Request, error: EngineError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(error)})
