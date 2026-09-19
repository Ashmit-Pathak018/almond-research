"""
Unit tests for core/retrieval/ranker_v3.py
Tests:
- Locked V3 scoring formula execution
- Normalization of P_eff prior signal
- Weight configuration per intent
- RetrievalTrace ranked breakdown population
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retrieval.contracts import RetrievalCandidate, RetrievalTrace, RetrievalChannel
from core.retrieval.intent_router import IntentAnalysis, IntentType
from core.retrieval.ranker_v3 import RankerV3, RankingWeights


def test_ranker_v3_scoring_formula():
    weights = RankingWeights(
        w_relevance=0.40,
        w_priority=0.20,
        w_temporal=0.20,
        w_entity=0.20,
    )
    ranker = RankerV3(default_weights=weights)

    intent = IntentAnalysis(
        intent_type=IntentType.HYBRID,
        confidence=1.0,
        channel_weights={}
    )
    trace = RetrievalTrace()

    # Candidate 1: High relevance, moderate priority
    c1 = RetrievalCandidate(
        memory_id="m1",
        fused_score=0.90,          # s_rel = 0.90
        lifecycle_priority=8.0,    # p_eff_norm = 8.0 / 10.0 = 0.80
        temporal_fit=0.50,         # s_temporal = 0.50
        entity_confidence=0.70,    # s_entity = 0.70
    )
    c1.source_channels = [RetrievalChannel.SEMANTIC]

    # Candidate 2: Low relevance, high priority
    c2 = RetrievalCandidate(
        memory_id="m2",
        fused_score=0.20,          # s_rel = 0.20
        lifecycle_priority=10.0,   # p_eff_norm = 10.0 / 10.0 = 1.00
        temporal_fit=0.10,         # s_temporal = 0.10
        entity_confidence=0.0,     # s_entity = 0.0
    )
    c2.source_channels = [RetrievalChannel.SEMANTIC]

    ranked = ranker.rank([c1, c2], intent, trace=trace, top_k=5)

    # Score calculation for c1:
    # 0.40 * 0.90 + 0.20 * 0.80 + 0.20 * 0.50 + 0.20 * 0.70
    # = 0.36 + 0.16 + 0.10 + 0.14 = 0.76
    expected_c1 = 0.40 * 0.90 + 0.20 * 0.80 + 0.20 * 0.50 + 0.20 * 0.70
    assert abs(c1.final_score - expected_c1) < 1e-4

    # Score calculation for c2:
    # 0.40 * 0.20 + 0.20 * 1.00 + 0.20 * 0.10 + 0.20 * 0.0
    # = 0.08 + 0.20 + 0.02 + 0.0 = 0.30
    expected_c2 = 0.40 * 0.20 + 0.20 * 1.00 + 0.20 * 0.10 + 0.20 * 0.0
    assert abs(c2.final_score - expected_c2) < 1e-4

    assert ranked[0].memory_id == "m1"
    assert ranked[0].rank == 1
    assert ranked[1].memory_id == "m2"
    assert ranked[1].rank == 2

    # Verify trace population
    assert len(trace.ranked_candidates) == 2
    assert trace.ranked_candidates[0]["memory_id"] == "m1"
    assert trace.ranked_candidates[0]["final_score"] == round(expected_c1, 4)

    print("PASS: test_ranker_v3_scoring_formula")


if __name__ == "__main__":
    test_ranker_v3_scoring_formula()
    print("All RankerV3 unit tests passed successfully.")
