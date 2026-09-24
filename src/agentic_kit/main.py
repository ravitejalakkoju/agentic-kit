from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import graph, prompts, turns
from .composition import build
from .errors import EngineError, ProviderError
from .ports.llm import LlmPort
from .settings import Settings, load_settings

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None, llm: LlmPort | None = None) -> FastAPI:
    """Build the HTTP app. Tests pass their own settings and model; the server passes neither."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or load_settings()
        app.state.settings = resolved
        app.state.components = build(resolved, llm)
        yield

    app = FastAPI(
        title="agentic-kit",
        version="0.1.0",
        summary="A conversation engine you can read: graph in, reply out.",
        lifespan=lifespan,
    )
    app.include_router(turns.router)
    app.include_router(graph.router)
    app.include_router(prompts.router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.exception_handler(ProviderError)
    async def handle_provider_error(_: Request, error: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(EngineError)
    async def handle_engine_error(_: Request, error: EngineError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(error)})

    return app


app = create_app()
