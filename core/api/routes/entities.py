"""
Project Almond V3 — Entities Route Handler
GET /v3/entities/{id}
"""

from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from core.api.schemas import EntityResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["entities"])


def get_entity_store(request: Request):
    return request.app.state.entity_store


@router.get("/entities/{entity_id}", response_model=EntityResponse)
def get_entity(
    entity_id: str,
    namespace: Optional[str] = Query(None, description="Optional namespace isolation scope"),
    entity_store = Depends(get_entity_store),
):
    entity = entity_store.get_entity(entity_id, namespace_id=namespace)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    aliases = [a.alias_name for a in entity_store.get_aliases(entity_id)]
    mem_links = entity_store.get_memories_for_entity(entity_id)
    linked_memory_ids = [m["memory_id"] for m in mem_links]

    return EntityResponse(
        entity_id=entity.entity_id,
        namespace_id=entity.namespace_id,
        canonical_name=entity.canonical_name,
        entity_type=entity.entity_type,
        summary=entity.summary,
        aliases=aliases,
        linked_memory_ids=linked_memory_ids,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )
