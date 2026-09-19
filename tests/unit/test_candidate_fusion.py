"""
Unit tests for core/retrieval/candidate_fusion.py
Tests:
- Deduplication by memory_id across multi-channel candidate lists
- Merging channel evidence and normalized scores
- Weighted score calculation across channels
- Preservation of source channels and provenance
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel
from core.retrieval.candidate_fusion import CandidateFusion


def test_fusion_deduplication_and_merging():
    fusion = CandidateFusion()

    # Candidate M42 from Semantic Channel
    c_sem = RetrievalCandidate(memory_id="M42", content="Content of M42")
    c_sem.add_channel_result(RetrievalChannel.SEMANTIC, raw_score=0.19, normalized_score=0.81)

    # Candidate M42 from Lexical Channel
    c_lex = RetrievalCandidate(memory_id="M42")
    c_lex.add_channel_result(RetrievalChannel.LEXICAL, raw_score=3.0, normalized_score=0.73)

    # Candidate M42 from Entity Channel
    c_ent = RetrievalCandidate(memory_id="M42")
    c_ent.add_channel_result(
        RetrievalChannel.ENTITY,
        raw_score=0.95,
        normalized_score=0.95,
        metadata={"matched_alias": "Bob", "confidence": 0.95}
    )

    # Candidate M99 only from Semantic
    c_other = RetrievalCandidate(memory_id="M99", content="Content of M99")
    c_other.add_channel_result(RetrievalChannel.SEMANTIC, raw_score=0.40, normalized_score=0.60)

    channel_results = {
        RetrievalChannel.SEMANTIC: [c_sem, c_other],
        RetrievalChannel.LEXICAL: [c_lex],
        RetrievalChannel.ENTITY: [c_ent],
        RetrievalChannel.TEMPORAL: [],
    }

    channel_weights = {
        RetrievalChannel.SEMANTIC: 0.35,
        RetrievalChannel.LEXICAL: 0.25,
        RetrievalChannel.ENTITY: 0.20,
        RetrievalChannel.TEMPORAL: 0.20,
    }

    fused = fusion.fuse(channel_results, channel_weights)

    # Must deduplicate M42 into a single candidate
    assert len(fused) == 2
    fused_m42 = next(c for c in fused if c.memory_id == "M42")

    # Verify channels merged
    assert RetrievalChannel.SEMANTIC in fused_m42.source_channels
    assert RetrievalChannel.LEXICAL in fused_m42.source_channels
    assert RetrievalChannel.ENTITY in fused_m42.source_channels

    # Verify scores preserved
    assert fused_m42.raw_scores["SEMANTIC"] == 0.19
    assert fused_m42.raw_scores["LEXICAL"] == 3.0
    assert fused_m42.normalized_scores["SEMANTIC"] == 0.81
    assert fused_m42.normalized_scores["LEXICAL"] == 0.73
    assert fused_m42.normalized_scores["ENTITY"] == 0.95

    # Verify entity evidence
    assert "Bob" in fused_m42.entity_matches
    assert fused_m42.entity_confidence == 0.95

    # Verify fused relevance score calculation (Semantic + Lexical)
    expected_fused = (0.35 * 0.81 + 0.25 * 0.73) / (0.35 + 0.25)
    assert abs(fused_m42.fused_score - expected_fused) < 1e-4

    # M42 should rank above M99
    assert fused[0].memory_id == "M42"
    assert fused[1].memory_id == "M99"

    print("PASS: test_fusion_deduplication_and_merging")


if __name__ == "__main__":
    test_fusion_deduplication_and_merging()
    print("All CandidateFusion unit tests passed successfully.")
