"""
Project Almond V3 — Semantic Retriever
Chroma-backed dense vector retrieval.
Derived accelerator only — gracefully degrades when Chroma is empty or unavailable.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Any

from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel, RetrievalQuery

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Retrieves candidates by dense semantic similarity using Chroma vector index.
    """

    def __init__(self, memory_store: Any):
        self.store = memory_store

    def retrieve(self, query: RetrievalQuery, top_k: Optional[int] = None) -> List[RetrievalCandidate]:
        k = top_k or query.top_k
        if not hasattr(self.store, "_collection") or self.store._collection is None:
            logger.debug("SemanticRetriever: Chroma collection unavailable.")
            return []

        try:
            count = self.store._collection.count()
            if count == 0:
                logger.debug("SemanticRetriever: Chroma collection is empty.")
                return []
        except Exception as e:
            logger.warning("SemanticRetriever count check failed: %s", e)
            return []

        # Construct namespace filter
        where_filter: Optional[dict] = None
        if query.namespace_id:
            where_filter = {"namespace_id": query.namespace_id}

        try:
            results = self.store._collection.query(
                query_texts=[query.query_text],
                n_results=min(k, count),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning("SemanticRetriever query failed: %s", e)
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        candidates: List[RetrievalCandidate] = []
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [1.0] * len(ids)
        documents = results.get("documents", [[]])[0] if "documents" in results else [""] * len(ids)
        metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results else [{}] * len(ids)

        for mid, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # Chroma cosine distance is 1.0 - cos(u, v). Convert to similarity in [0.0, 1.0].
            raw_distance = float(dist) if dist is not None else 1.0
            norm_sim = max(0.0, min(1.0, 1.0 - raw_distance))

            if norm_sim < 0.10:
                # Below semantic relevance noise floor
                continue

            cand = RetrievalCandidate(
                memory_id=mid,
                content=doc or None,
                tag=meta.get("tag") if meta else None,
                tier=meta.get("tier") if meta else None,
                last_accessed_at=meta.get("last_accessed_at") if meta else None,
                lifecycle_priority=meta.get("p_eff", 0.0) if meta else 0.0,
            )
            cand.add_channel_result(
                channel=RetrievalChannel.SEMANTIC,
                raw_score=raw_distance,
                normalized_score=norm_sim
            )
            candidates.append(cand)

        return candidates
