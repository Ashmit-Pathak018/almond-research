"""
Project Almond V3 — Ranker V3
Combines multi-channel signals, normalized effective priority (P_eff), temporal fit, and entity fit.
Implements the locked V3 scoring formula.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.retrieval.contracts import RetrievalCandidate, RetrievalTrace
from core.retrieval.intent_router import IntentAnalysis, IntentType

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Configurable weights for Ranker V3."""
    w_relevance: float = 0.40
    w_priority: float = 0.20
    w_temporal: float = 0.20
    w_entity: float = 0.20

    @classmethod
    def for_intent(cls, intent_type: IntentType) -> RankingWeights:
        """Adaptive weights tuned to the dominant retrieval intent."""
        if intent_type == IntentType.TEMPORAL:
            return cls(w_relevance=0.30, w_priority=0.15, w_temporal=0.45, w_entity=0.10)
        elif intent_type == IntentType.ENTITY:
            return cls(w_relevance=0.30, w_priority=0.15, w_temporal=0.10, w_entity=0.45)
        elif intent_type == IntentType.COMPARISON:
            return cls(w_relevance=0.35, w_priority=0.15, w_temporal=0.35, w_entity=0.15)
        elif intent_type == IntentType.SEMANTIC:
            return cls(w_relevance=0.55, w_priority=0.25, w_temporal=0.10, w_entity=0.10)
        else:  # HYBRID
            return cls(w_relevance=0.30, w_priority=0.20, w_temporal=0.25, w_entity=0.25)


class RankerV3:
    """
    Ranks candidates using the locked V3 retrieval formula:
        Score = w_r * S_relevance + w_p * P_eff_norm + w_t * S_temporal + w_e * S_entity
    """

    def __init__(self, default_weights: Optional[RankingWeights] = None):
        self.default_weights = default_weights

    def rank(
        self,
        candidates: List[RetrievalCandidate],
        intent: IntentAnalysis,
        trace: Optional[RetrievalTrace] = None,
        top_k: int = 10,
    ) -> List[RetrievalCandidate]:
        """
        Calculates final score for each candidate and sorts descending.
        Populates candidate score breakdowns and records them into trace.
        """
        if self.default_weights:
            weights = self.default_weights
        else:
            # Check if intent specialized channel actually yielded matches
            has_entity_match = any(c.entity_confidence > 0.05 for c in candidates)
            has_temporal_match = any(c.temporal_fit > 0.05 for c in candidates)

            if intent.intent_type == IntentType.ENTITY and not has_entity_match:
                weights = RankingWeights.for_intent(IntentType.SEMANTIC)
            elif intent.intent_type == IntentType.TEMPORAL and not has_temporal_match:
                weights = RankingWeights.for_intent(IntentType.SEMANTIC)
            else:
                weights = RankingWeights.for_intent(intent.intent_type)

        if trace:
            trace.ranking_weights_used = {
                "w_relevance": weights.w_relevance,
                "w_priority": weights.w_priority,
                "w_temporal": weights.w_temporal,
                "w_entity": weights.w_entity,
            }

        scored_candidates: List[RetrievalCandidate] = []

        for cand in candidates:
            # 1. Relevance signal: fused score from semantic + lexical
            s_rel = cand.fused_score

            # 2. Effective Priority signal: normalized to [0.0, 1.0]
            # Max base importance is 10.0, freshness in [0.0, 1.0] -> P_eff in [0.0, 10.0]
            p_eff_norm = max(0.0, min(1.0, cand.lifecycle_priority / 10.0))

            # 3. Temporal fit signal in [0.0, 1.0]
            s_temporal = max(0.0, min(1.0, cand.temporal_fit))

            # 4. Entity fit signal in [0.0, 1.0]
            s_entity = max(0.0, min(1.0, cand.entity_confidence))

            # Query match signal: at least one channel (relevance, temporal, entity) must show relevance
            query_match_strength = max(s_rel, s_temporal, s_entity)
            if query_match_strength < 0.05:
                final_score = 0.0
            else:
                # Locked scoring formula
                final_score = (
                    weights.w_relevance * s_rel
                    + weights.w_priority * p_eff_norm
                    + weights.w_temporal * s_temporal
                    + weights.w_entity * s_entity
                )

            cand.final_score = final_score
            scored_candidates.append(cand)

        # Sort descending by final score
        scored_candidates.sort(key=lambda c: c.final_score, reverse=True)

        # Assign ranks and record in trace
        for idx, c in enumerate(scored_candidates):
            c.rank = idx + 1
            if trace and idx < top_k:
                trace.ranked_candidates.append({
                    "rank": c.rank,
                    "memory_id": c.memory_id,
                    "final_score": round(c.final_score, 4),
                    "s_rel": round(c.fused_score, 4),
                    "p_eff": round(c.lifecycle_priority, 4),
                    "s_temporal": round(c.temporal_fit, 4),
                    "s_entity": round(c.entity_confidence, 4),
                    "channels": [ch.value for ch in c.source_channels],
                })

        return scored_candidates[:top_k]
