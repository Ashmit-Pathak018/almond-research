"""
Project Almond V3 — Health Route Handler
GET /v3/health
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, Request

from core.api.schemas import HealthResponse, ComponentHealth
from core.clock import get_clock

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def get_store(request: Request):
    return request.app.state.store


def get_service(request: Request):
    return request.app.state.service


@router.get("/health", response_model=HealthResponse)
def get_health(
    store = Depends(get_store),
    service = Depends(get_service),
):
    components = {}
    overall_status = "healthy"

    # 1. Check SQLite
    try:
        store._conn.execute("SELECT 1").fetchone()
        components["sqlite"] = ComponentHealth(status="healthy", details="SQLite source of truth accessible")
    except Exception as e:
        components["sqlite"] = ComponentHealth(status="unhealthy", details=str(e))
        overall_status = "unhealthy"

    # 2. Check Ingestion Service
    try:
        worker_alive = service._worker_thread is not None and service._worker_thread.is_alive()
        if worker_alive:
            components["ingestion_worker"] = ComponentHealth(status="healthy", details="Worker thread running")
        else:
            components["ingestion_worker"] = ComponentHealth(status="degraded", details="Worker thread idle or stopped")
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        components["ingestion_worker"] = ComponentHealth(status="unhealthy", details=str(e))
        if overall_status == "healthy":
            overall_status = "degraded"

    # 3. Check Chroma Derived Index
    try:
        count = store._collection.count()
        components["chroma"] = ComponentHealth(status="healthy", details=f"Chroma index online ({count} vectors)")
    except Exception as e:
        # Phase 3 invariant: Graceful degradation if Chroma is unavailable
        components["chroma"] = ComponentHealth(
            status="degraded",
            details=f"Chroma offline; falling back to SQLite FTS5/relational: {e}"
        )
        if overall_status == "healthy":
            overall_status = "degraded"

    # 4. Clock Details
    clock = get_clock()
    clock_info = {
        "provider": type(clock).__name__,
        "now": clock.now(),
        "reference_time": clock.reference_time(),
    }

    return HealthResponse(
        status=overall_status,
        version="v3",
        components=components,
        clock=clock_info,
    )
