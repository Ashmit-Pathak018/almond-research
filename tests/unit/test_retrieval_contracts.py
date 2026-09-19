"""
Unit tests for core/retrieval/contracts.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retrieval.contracts import (
    RetrievalChannel, RetrievalQuery, RetrievalCandidate, RetrievalTrace, RetrievalResult
)

def test_retrieval_query():
    q = RetrievalQuery(query_text="What did Bob work on?", namespace_id="user_1", reference_time=1700000000.0)
    assert q.query_text == "What did Bob work on?"
    assert q.namespace_id == "user_1"
    assert q.effective_reference_time() == 1700000000.0
    print("PASS: test_retrieval_query")

def test_retrieval_candidate_merging():
    cand = RetrievalCandidate(memory_id="mem_1")
    cand.add_channel_result(RetrievalChannel.SEMANTIC, raw_score=0.25, normalized_score=0.75)
    cand.add_channel_result(RetrievalChannel.ENTITY, raw_score=1.0, normalized_score=0.95, metadata={"matched_alias": "Bob", "confidence": 0.95})
    
    assert len(cand.source_channels) == 2
    assert RetrievalChannel.SEMANTIC in cand.source_channels
    assert RetrievalChannel.ENTITY in cand.source_channels
    assert cand.raw_scores["SEMANTIC"] == 0.25
    assert cand.normalized_scores["SEMANTIC"] == 0.75
    assert cand.normalized_scores["ENTITY"] == 0.95
    assert "Bob" in cand.entity_matches
    assert cand.entity_confidence == 0.95
    print("PASS: test_retrieval_candidate_merging")

def test_retrieval_trace_serialization():
    trace = RetrievalTrace(query="test query", namespace_id="default", reference_time=1704067200.0)
    trace.detected_intent = "TEMPORAL"
    trace.channels_run = ["SEMANTIC", "TEMPORAL"]
    trace.candidates_per_channel = {"SEMANTIC": 5, "TEMPORAL": 3}
    trace.fusion_merged_count = 7
    trace.final_context_ids = ["mem_1", "mem_2"]
    
    d = trace.to_dict()
    assert d["query"] == "test query"
    assert d["detected_intent"] == "TEMPORAL"
    assert d["fusion_merged_count"] == 7
    assert d["final_context_ids"] == ["mem_1", "mem_2"]
    print("PASS: test_retrieval_trace_serialization")

if __name__ == "__main__":
    test_retrieval_query()
    test_retrieval_candidate_merging()
    test_retrieval_trace_serialization()
    print("All retrieval contracts tests passed successfully.")
