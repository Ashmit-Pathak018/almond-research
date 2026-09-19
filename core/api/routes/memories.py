"""
Project Almond V3 — Memories Route Handlers
POST /v3/memories
GET  /v3/memories/{id}
DELETE /v3/memories/{id}
"""

from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request

from core.api.schemas import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    MemoryResponse,
    DeleteMemoryResponse,
)
from core.memory_block import MemoryBlock, MemoryTag

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memories"])


def get_service(request: Request):
    return request.app.state.service


def get_store(request: Request):
    return request.app.state.store


@router.post("/memories", response_model=CreateMemoryResponse, status_code=201)
def create_memory(
    req: CreateMemoryRequest,
    service = Depends(get_service),
):
    try:
        tag_enum = MemoryTag(req.tag)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tag '{req.tag}'. Valid tags are: {[t.value for t in MemoryTag]}"
        )

    if req.sync:
        # Synchronous evaluation path
        block = service.ingest_sync(
            content=req.content,
            namespace_id=req.namespace,
            tag=tag_enum,
            importance_score=req.importance_score,
            keywords=req.keywords,
            event_time=req.event_time,
            summary=req.summary,
        )
        return CreateMemoryResponse(
            memory_id=block.id,
            job_id=None,
            status="SUCCESS",
            namespace=req.namespace,
        )
    else:
        # Asynchronous durable service path
        job = service.submit_job(
            content=req.content,
            namespace_id=req.namespace,
            tag=tag_enum,
            importance_score=req.importance_score,
            keywords=req.keywords,
            event_time=req.event_time,
            summary=req.summary,
        )
        return CreateMemoryResponse(
            memory_id=job.memory_id,
            job_id=job.job_id,
            status=job.status,
            namespace=req.namespace,
        )


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
def get_memory(
    memory_id: str,
    store = Depends(get_store),
):
    block = store.get_by_id(memory_id)
    if not block:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")

    return MemoryResponse(
        id=block.id,
        namespace_id=getattr(block, "namespace_id", "default"),
        content=block.content,
        summary=block.summary,
        tag=block.tag.value,
        tier=block.tier.value,
        state=getattr(block, "state", "ACTIVE"),
        importance_score=block.importance_score,
        p_eff=round(block.p_eff, 4),
        keywords=block.keywords or [],
        event_time=block.event_time,
        created_at=block.created_at,
        updated_at=getattr(block, "updated_at", block.created_at),
        last_accessed_at=block.last_accessed_at,
        access_count=block.access_count,
    )


@router.delete("/memories/{memory_id}", response_model=DeleteMemoryResponse)
def delete_memory(
    memory_id: str,
    store = Depends(get_store),
):
    block = store.get_by_id(memory_id)
    if not block:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")

    store.delete(memory_id)
    return DeleteMemoryResponse(deleted=True, memory_id=memory_id)
