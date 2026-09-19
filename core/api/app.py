"""
Project Almond V3 — Standalone FastAPI Application Factory
Mounts /v3 endpoints with typed contracts, durable ingestion worker lifecycle,
and strict error handling.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from core.api.routes import (
    memories,
    query,
    timeline,
    entities,
    evaluations,
    health,
)
from core.knowledge.entity_store import EntityStore
from core.knowledge.fact_store import FactStore
from core.knowledge.timeline_store import TimelineStore
from core.memory_store import MemoryStore
from core.retrieval.retrieval_engine import RetrievalEngine
from core.workers.ingestion_service import IngestionService

logger = logging.getLogger("almond.api")


def create_app(
    store: Optional[MemoryStore] = None,
    db_path: str = "almond_v3.db",
    chroma_path: str = "./almond_v3_chroma",
    start_worker: bool = True,
) -> FastAPI:
    """
    Factory creating the configured FastAPI instance for Almond V3.
    """

    # Initialise core components if not provided
    memory_store = store or MemoryStore(db_path=db_path, chroma_path=chroma_path)
    entity_store = EntityStore(memory_store._conn)
    timeline_store = TimelineStore(memory_store._conn)
    fact_store = FactStore(memory_store._conn)

    retrieval_engine = RetrievalEngine(
        memory_store=memory_store,
        entity_store=entity_store,
        timeline_store=timeline_store,
    )

    ingestion_service = IngestionService(
        memory_store=memory_store,
        entity_store=entity_store,
        timeline_store=timeline_store,
        fact_store=fact_store,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            ingestion_service.start(recover_stale=True)
        yield
        if start_worker:
            ingestion_service.stop()

    app = FastAPI(
        title="Almond V3 Memory Service",
        version="3.0.0",
        description="Standalone, typed, durable memory service for multi-tenant cognitive architectures.",
        lifespan=lifespan,
    )

    # Attach shared state
    app.state.store = memory_store
    app.state.entity_store = entity_store
    app.state.timeline_store = timeline_store
    app.state.fact_store = fact_store
    app.state.retrieval_engine = retrieval_engine
    app.state.service = ingestion_service

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount /v3 router
    app.include_router(memories.router, prefix="/v3")
    app.include_router(query.router, prefix="/v3")
    app.include_router(timeline.router, prefix="/v3")
    app.include_router(entities.router, prefix="/v3")
    app.include_router(evaluations.router, prefix="/v3")
    app.include_router(health.router, prefix="/v3")

    # Clean error handlers — no leaked stack traces or DB details to external clients
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "validation_error", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled API error on %s %s: %s", request.method, request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred processing the request."},
        )

    return app


# Default app instance for uvicorn
app = create_app()
