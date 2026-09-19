"""
Project Almond V3 — Timeline Route Handler
GET /v3/timeline
"""

from __future__ import annotations
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, Request

from core.api.schemas import TimelineResponse, TimelineEventResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["timeline"])


def get_timeline_store(request: Request):
    return request.app.state.timeline_store


@router.get("/timeline", response_model=TimelineResponse)
def get_timeline(
    namespace: str = Query("default", description="Namespace ID"),
    limit: int = Query(50, ge=1, le=200, description="Max events to return"),
    start_time: Optional[float] = Query(None, description="Filter events after or on this epoch time"),
    end_time: Optional[float] = Query(None, description="Filter events before or on this epoch time"),
    timeline_store = Depends(get_timeline_store),
):
    if start_time is not None and end_time is not None:
        events = timeline_store.get_between(namespace, start_time, end_time)[:limit]
    elif start_time is not None:
        events = timeline_store.get_after(namespace, start_time)[:limit]
    elif end_time is not None:
        events = timeline_store.get_before(namespace, end_time)[:limit]
    else:
        events = timeline_store.get_chronological(namespace, limit=limit)

    events_resp = []
    for ev in events:
        linked_mems = timeline_store.get_memories_for_event(ev.event_id)
        events_resp.append(
            TimelineEventResponse(
                event_id=ev.event_id,
                namespace_id=ev.namespace_id,
                title=ev.title,
                event_type=ev.event_type,
                start_time=ev.start_time,
                end_time=ev.end_time,
                confidence=ev.confidence,
                linked_memory_ids=linked_mems,
            )
        )

    return TimelineResponse(
        namespace=namespace,
        events=events_resp,
        total_count=len(events_resp),
    )
