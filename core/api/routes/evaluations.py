"""
Project Almond V3 — Evaluations Route Handler
POST /v3/evaluations
"""

from __future__ import annotations
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request

from core.api.schemas import EvaluationRequest, EvaluationResponse
from tests.golden.golden_cases import GOLDEN_CASES
from tests.golden.run_golden_cases_v3 import setup_knowledge_for_case
from core.retrieval.contracts import RetrievalQuery
from core.memory_block import MemoryBlock, MemoryTag

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluations"])


def get_engine(request: Request):
    return request.app.state.retrieval_engine


def get_service(request: Request):
    return request.app.state.service


@router.post("/evaluations", response_model=EvaluationResponse)
def run_evaluation(
    req: EvaluationRequest,
    engine = Depends(get_engine),
    service = Depends(get_service),
):
    eval_id = str(uuid.uuid4())

    if req.evaluation_type.lower() != "golden":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported evaluation_type '{req.evaluation_type}'. Currently supported: 'golden'"
        )

    cases_to_run = GOLDEN_CASES
    if req.case_id:
        cases_to_run = [c for c in GOLDEN_CASES if c.case_id.upper() == req.case_id.upper()]
        if not cases_to_run:
            raise HTTPException(status_code=404, detail=f"Golden case '{req.case_id}' not found")

    # Run evaluation in an isolated evaluation namespace
    eval_ns = f"eval_{eval_id[:8]}"
    passed = 0
    failed = 0
    details = {}

    for case in cases_to_run:
        try:
            # Seed case preconditions using synchronous service ingestion in eval namespace
            for p in case.preconditions:
                try:
                    tag = MemoryTag(p.tag)
                except ValueError:
                    tag = MemoryTag.USER_PROFILE

                service.ingest_sync(
                    content=p.content,
                    namespace_id=eval_ns,
                    tag=tag,
                    event_time=p.event_time,
                    memory_id=p.id,
                )

            # Query using engine
            rq = RetrievalQuery(
                query_text=case.query,
                namespace_id=eval_ns,
                reference_time=case.reference_time,
            )
            res = engine.query(rq)

            # Check case criteria
            case_passed = True
            if case.expected_abstention:
                if not res.is_abstention:
                    case_passed = False
            elif case.expected_retrieved_ids:
                if not res.memory_ids or res.memory_ids[0] != case.expected_retrieved_ids[0]:
                    case_passed = False

            if case_passed:
                passed += 1
                details[case.case_id] = "PASS"
            else:
                failed += 1
                details[case.case_id] = "FAIL"

        except Exception as e:
            failed += 1
            details[case.case_id] = f"ERROR: {e}"

    return EvaluationResponse(
        evaluation_id=eval_id,
        status="completed" if failed == 0 else "failures_detected",
        total_cases=len(cases_to_run),
        passed_cases=passed,
        failed_cases=failed,
        details=details,
    )
