import math
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

print("===== NEW RETRIEVAL PIPELINE LOADED =====")


@dataclass
class RetrievalCandidate:
    """
    Wrapper to avoid mutating raw memory block dictionaries.
    Holds intermediate scores and retrieval telemetry safely.
    """

    block: Dict[str, Any]

    distance: float

    similarity: float = 0.0
    normalized_peff: float = 0.0
    recency_score: float = 0.0
    keyword_score: float = 0.0

    hybrid_score: float = 0.0

    trace: Dict[str, Any] = field(default_factory=dict)


class RetrievalOptimizer:

    def __init__(
        self,
        min_similarity: float = 0.35,
        score_dropoff_ratio: float = 0.65,
        max_results: int = 5,
        recency_tau: float = 14.0
    ):

        # --------------------------------------------------------------
        # Retrieval Controls
        # --------------------------------------------------------------

        self.min_similarity = min_similarity

        self.score_dropoff_ratio = score_dropoff_ratio

        self.max_results = max_results

        self.recency_tau = recency_tau

        # --------------------------------------------------------------
        # Debug State
        # --------------------------------------------------------------

        self.latest_rejections: List[Dict[str, Any]] = []

        self.latest_accepts: List[Dict[str, Any]] = []

        # --------------------------------------------------------------
        # Ablation Flags
        # --------------------------------------------------------------

        self.disable_intent = False
        self.disable_keyword = False
        self.disable_recency = False
        self.disable_peff = False

    # ======================================================================
    # KEYWORD OVERLAP
    # ======================================================================

    def _calculate_keyword_score(
        self,
        query: str,
        block_keywords: List[str]
    ) -> float:

        if not query or not block_keywords:
            return 0.0

        clean_query = re.sub(
            r"[^\w\s]",
            "",
            query.lower()
        )

        query_terms = set(clean_query.split())

        block_terms = set(
            k.lower()
            for k in block_keywords
        )

        if not query_terms or not block_terms:
            return 0.0

        overlap = query_terms.intersection(block_terms)

        hits = len(overlap)

        return min(1.0, hits / 2.0)

    # ======================================================================
    # MAIN RETRIEVAL PIPELINE
    # ======================================================================

    def rerank_and_filter(
        self,
        query: str,
        raw_chroma_results: List[Dict],
        current_time: float,
        *args,
        **kwargs
    ) -> List[Dict]:

        """
        Hybrid retrieval reranking pipeline.

        Handles:
        - semantic filtering
        - p_eff weighting
        - recency weighting
        - keyword precision
        - adaptive retrieval depth
        - rejection tracing
        """

        print("[DEBUG] NEW RETRIEVAL PIPELINE ACTIVE")

        self.latest_rejections = []

        self.latest_accepts = []

        if not raw_chroma_results:
            return []

        candidates: List[RetrievalCandidate] = []

        # ------------------------------------------------------------------
        # Dynamic normalization
        # ------------------------------------------------------------------

        max_peff = max(
            (
                b.get("p_eff", 1.0)
                for b in raw_chroma_results
            ),
            default=1.0
        )

        if max_peff <= 0.0:
            max_peff = 1.0

        # ------------------------------------------------------------------
        # Candidate Evaluation
        # ------------------------------------------------------------------

        for block in raw_chroma_results:

            # --------------------------------------------------------------
            # Similarity
            # --------------------------------------------------------------

            raw_distance = block.get(
                "distance",
                1.0
            )

            similarity = 1.0 - raw_distance

            # --------------------------------------------------------------
            # Similarity Threshold
            # --------------------------------------------------------------

            if similarity < self.min_similarity:

                self.latest_rejections.append({
                    "id": block.get("id"),
                    "reason": f"low_similarity:{round(similarity,4)}",
                    "similarity": round(similarity, 4)
                })

                continue

            # --------------------------------------------------------------
            # P_eff
            # --------------------------------------------------------------

            normalized_p_eff = min(
                block.get("p_eff", 0.0) / max_peff,
                1.0
            )

            if self.disable_peff:
                normalized_p_eff = 0.0

            # --------------------------------------------------------------
            # Recency
            # --------------------------------------------------------------

            last_accessed = block.get(
                "last_accessed_at",
                current_time
            )

            delta_days = max(
                0.0,
                (current_time - last_accessed) / 86400.0
            )

            recency_score = math.exp(
                -delta_days / self.recency_tau
            )

            if self.disable_recency:
                recency_score = 0.0

            # --------------------------------------------------------------
            # Keyword Precision
            # --------------------------------------------------------------

            keyword_score = self._calculate_keyword_score(
                query,
                block.get("keywords", [])
            )

            if self.disable_keyword:
                keyword_score = 0.0

            # --------------------------------------------------------------
            # Hybrid Score
            # --------------------------------------------------------------

            hybrid_score = (
                (similarity * 0.6)
                + (normalized_p_eff * 0.2)
                + (recency_score * 0.1)
                + (keyword_score * 0.1)
            )

            # --------------------------------------------------------------
            # Candidate Wrap
            # --------------------------------------------------------------

            candidate = RetrievalCandidate(
                block=block,
                distance=raw_distance,
                similarity=similarity,
                normalized_peff=normalized_p_eff,
                recency_score=recency_score,
                keyword_score=keyword_score,
                hybrid_score=hybrid_score,
                trace={
                    "raw_distance": round(raw_distance, 4),
                    "similarity": round(similarity, 4),
                    "p_eff_contribution": round(normalized_p_eff, 4),
                    "recency_contribution": round(recency_score, 4),
                    "keyword_contribution": round(keyword_score, 4),
                    "final_hybrid_score": round(hybrid_score, 4),
                    "status": "accepted"
                }
            )

            candidates.append(candidate)

        # ------------------------------------------------------------------
        # Sort Candidates
        # ------------------------------------------------------------------

        candidates.sort(
            key=lambda x: x.hybrid_score,
            reverse=True
        )

        # ------------------------------------------------------------------
        # Adaptive Retrieval Depth
        # ------------------------------------------------------------------

        final_context = []

        if candidates:

            top_score = candidates[0].hybrid_score

            for candidate in candidates:

                # ----------------------------------------------------------
                # Score Dropoff
                # ----------------------------------------------------------

                if candidate.hybrid_score < (
                    top_score * self.score_dropoff_ratio
                ):

                    candidate.trace["status"] = "rejected"

                    self.latest_rejections.append({
                        "id": candidate.block.get("id"),
                        "reason": "threshold_dropoff",
                        "hybrid_score": round(candidate.hybrid_score, 4),
                        "top_score": round(top_score, 4)
                    })

                    continue

                # ----------------------------------------------------------
                # Hard Cap
                # ----------------------------------------------------------

                if len(final_context) >= self.max_results:

                    candidate.trace["status"] = "rejected"

                    self.latest_rejections.append({
                        "id": candidate.block.get("id"),
                        "reason": "max_results_cap",
                        "hybrid_score": round(candidate.hybrid_score, 4)
                    })

                    continue

                # ----------------------------------------------------------
                # Export Clean Block
                # ----------------------------------------------------------

                result_block = dict(candidate.block)

                result_block["retrieval_trace"] = (
                    candidate.trace
                )

                final_context.append(result_block)

                self.latest_accepts.append(result_block)

        return final_context