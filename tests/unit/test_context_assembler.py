"""
Unit tests for core/retrieval/context_assembler.py
Tests:
- Formatting structured evidence into clean context string
- Preserving memory IDs, dates, and entities
- Deterministic abstention with NO_RELEVANT_MEMORIES
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retrieval.contracts import RetrievalCandidate, RetrievalTrace
from core.retrieval.context_assembler import ContextAssembler, ABSTENTION_TOKEN


def test_context_assembler_formatting():
    assembler = ContextAssembler()
    trace = RetrievalTrace()

    c1 = RetrievalCandidate(
        memory_id="mem_1",
        content="The client infrastructure security audit was completed.",
        event_time=1709900000.0,
        entity_matches=["Robert Vance"],
        final_score=0.85,
    )
    c2 = RetrievalCandidate(
        memory_id="mem_2",
        content="We migrated to PostgreSQL with zero downtime.",
        event_time=1709294400.0,
        final_score=0.75,
    )

    result = assembler.assemble([c1, c2], trace)

    assert result.is_abstention is False
    assert result.confidence == 0.85
    assert len(result.candidates) == 2
    assert "Memory ID: mem_1" in result.context_text
    assert "Entities: Robert Vance" in result.context_text
    assert "The client infrastructure security audit was completed." in result.context_text
    assert "Memory ID: mem_2" in result.context_text
    assert trace.final_context_ids == ["mem_1", "mem_2"]

    print("PASS: test_context_assembler_formatting")


def test_context_assembler_abstention():
    assembler = ContextAssembler(confidence_threshold=0.10)
    trace = RetrievalTrace()

    # Empty candidate list
    res_empty = assembler.assemble([], trace)
    assert res_empty.is_abstention is True
    assert res_empty.context_text == ABSTENTION_TOKEN
    assert res_empty.candidates == []
    assert trace.final_context_ids == []

    # Below-threshold candidate
    c_weak = RetrievalCandidate(
        memory_id="mem_weak",
        content="Vague unrelated rumor",
        final_score=0.04,
    )
    res_weak = assembler.assemble([c_weak], trace)
    assert res_weak.is_abstention is True
    assert res_weak.context_text == ABSTENTION_TOKEN
    assert res_weak.candidates == []

    print("PASS: test_context_assembler_abstention")


if __name__ == "__main__":
    test_context_assembler_formatting()
    test_context_assembler_abstention()
    print("All ContextAssembler unit tests passed successfully.")
