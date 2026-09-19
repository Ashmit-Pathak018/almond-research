"""
Project Almond V3 — Query & Trace Route Handlers
POST /v3/query
POST /v3/query/trace
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, Request

from core.api.schemas import (
    QueryRequest,
    QueryResponse,
    CandidateResponse,
    QueryTraceResponse,
)
from core.retrieval.contracts import RetrievalQuery

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def get_engine(request: Request):
    return request.app.state.retrieval_engine


@router.post("/query", response_model=QueryResponse)
def query_memories(
    req: QueryRequest,
    engine = Depends(get_engine),
):
    retrieval_query = RetrievalQuery(
        query_text=req.query,
        namespace_id=req.namespace,
        reference_time=req.reference_time,
        top_k=req.top_k,
    )

    result = engine.query(retrieval_query)

    candidates_resp = [
        CandidateResponse(
            memory_id=c.memory_id,
            content=c.content,
            tag=c.tag,
            tier=c.tier,
            final_score=round(c.final_score, 4),
            rank=c.rank,
            source_channels=[ch.value for ch in c.source_channels],
            event_time=c.event_time,
        )
        for c in result.candidates
    ]

    return QueryResponse(
        query=req.query,
        namespace=req.namespace,
        context_text=result.context_text,
        is_abstention=result.is_abstention,
        confidence=round(result.confidence, 4),
        memory_ids=result.memory_ids,
        candidates=candidates_resp,
        trace_id=result.trace.trace_id if result.trace else None,
    )


@router.post("/query/trace", response_model=QueryTraceResponse)
def query_trace(
    req: QueryRequest,
    engine = Depends(get_engine),
):
    retrieval_query = RetrievalQuery(
        query_text=req.query,
        namespace_id=req.namespace,
        reference_time=req.reference_time,
        top_k=req.top_k,
    )

    result = engine.query(retrieval_query)
    trace = result.trace

    return QueryTraceResponse(
        trace_id=trace.trace_id,
        query=trace.query,
        namespace_id=trace.namespace_id,
        reference_time=trace.reference_time,
        detected_intent=trace.detected_intent,
        intent_confidence=round(trace.intent_confidence, 4),
        channels_run=trace.channels_run,
        candidates_per_channel=trace.candidates_per_channel,
        fusion_merged_count=trace.fusion_merged_count,
        temporal_constraints_detected=trace.temporal_constraints_detected,
        rejected_count=len(trace.rejected_candidates),
        rejected_candidates=trace.rejected_candidates,
        ranking_weights_used=trace.ranking_weights_used,
        ranked_candidates=trace.ranked_candidates,
        final_context_ids=trace.final_context_ids,
        duration_ms=round(trace.duration_ms, 2),
        created_at=trace.created_at,
    )
