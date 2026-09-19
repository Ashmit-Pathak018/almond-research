"""
Unit tests for core/retrieval/semantic_retriever.py
Tests:
- Vector query execution
- Handling empty Chroma index
- Handling missing/unavailable Chroma collection
- Namespace filtering
- Distance-to-similarity normalization
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_store import MemoryStore
from core.retrieval.contracts import RetrievalQuery, RetrievalChannel
from core.retrieval.semantic_retriever import SemanticRetriever


def test_semantic_retriever_query():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "semantic_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_test")
        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        retriever = SemanticRetriever(store)

        mb = MemoryBlock(
            id="mem_vector_1",
            content="Kubernetes pod autoscaling using Prometheus metrics.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=8.0,
        )
        store.save(mb)

        query = RetrievalQuery(query_text="Kubernetes Prometheus autoscaling", namespace_id="default")
        results = retriever.retrieve(query)

        assert len(results) >= 1
        assert results[0].memory_id == "mem_vector_1"
        assert RetrievalChannel.SEMANTIC in results[0].source_channels
        score = results[0].normalized_scores[RetrievalChannel.SEMANTIC.value]
        assert 0.0 < score <= 1.0

        store.close()
    print("PASS: test_semantic_retriever_query")


def test_semantic_retriever_empty_collection():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "semantic_empty.db")
        chroma_path = os.path.join(tmpdir, "chroma_empty")
        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        retriever = SemanticRetriever(store)

        query = RetrievalQuery(query_text="anything at all", namespace_id="default")
        results = retriever.retrieve(query)
        assert results == []

        store.close()
    print("PASS: test_semantic_retriever_empty_collection")


def test_semantic_retriever_unavailable_collection():
    class DummyStore:
        _collection = None

    retriever = SemanticRetriever(DummyStore())
    query = RetrievalQuery(query_text="query", namespace_id="default")
    results = retriever.retrieve(query)
    assert results == []
    print("PASS: test_semantic_retriever_unavailable_collection")


def test_semantic_retriever_namespace_isolation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "semantic_ns.db")
        chroma_path = os.path.join(tmpdir, "chroma_ns")
        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        retriever = SemanticRetriever(store)

        mb_a = MemoryBlock(
            id="mem_a",
            content="Top secret project blueprint for namespace Alpha.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=5.0,
            namespace_id="alpha",
        )
        store.save(mb_a)

        mb_b = MemoryBlock(
            id="mem_b",
            content="Public company news for namespace Beta.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=5.0,
            namespace_id="beta",
        )
        store.save(mb_b)

        query_a = RetrievalQuery(query_text="Top secret project blueprint", namespace_id="alpha")
        results_a = retriever.retrieve(query_a)
        ids_a = [r.memory_id for r in results_a]
        assert "mem_a" in ids_a
        assert "mem_b" not in ids_a

        store.close()
    print("PASS: test_semantic_retriever_namespace_isolation")


if __name__ == "__main__":
    test_semantic_retriever_query()
    test_semantic_retriever_empty_collection()
    test_semantic_retriever_unavailable_collection()
    test_semantic_retriever_namespace_isolation()
    print("All SemanticRetriever unit tests passed successfully.")
