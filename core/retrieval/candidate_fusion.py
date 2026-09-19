"""
Project Almond V3 — Candidate Fusion
Fuses multi-channel retrieval results into a deduplicated candidate pool.
Preserves raw scores, channel provenance, and normalized multi-signal evidence.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel

logger = logging.getLogger(__name__)


class CandidateFusion:
    """
    Combines candidate lists from Semantic, Lexical, Entity, and Temporal channels.
    Deduplicates by memory_id, merges evidence, and computes combined relevance.
    """

    def __init__(self):
        pass

    def fuse(
        self,
        channel_results: Dict[RetrievalChannel, List[RetrievalCandidate]],
        channel_weights: Dict[RetrievalChannel, float],
    ) -> List[RetrievalCandidate]:
        """
        Fuse multi-channel candidates into a single deduplicated list.
        """
        merged_map: Dict[str, RetrievalCandidate] = {}

        # 1. Deduplicate and merge candidates across channels
        for channel, candidates in channel_results.items():
            for cand in candidates:
                mid = cand.memory_id
                if mid not in merged_map:
                    merged_map[mid] = cand
                else:
                    existing = merged_map[mid]
                    # Merge channels
                    for ch in cand.source_channels:
                        if ch not in existing.source_channels:
                            existing.source_channels.append(ch)
                    # Merge scores
                    existing.raw_scores.update(cand.raw_scores)
                    existing.normalized_scores.update(cand.normalized_scores)
                    # Merge text/attributes if missing
                    if not existing.content and cand.content:
                        existing.content = cand.content
                    if not existing.tag and cand.tag:
                        existing.tag = cand.tag
                    if not existing.tier and cand.tier:
                        existing.tier = cand.tier
                    if existing.event_time is None and cand.event_time is not None:
                        existing.event_time = cand.event_time
                    if existing.last_accessed_at is None and cand.last_accessed_at is not None:
                        existing.last_accessed_at = cand.last_accessed_at
                    # Merge entity evidence
                    for em in cand.entity_matches:
                        if em not in existing.entity_matches:
                            existing.entity_matches.append(em)
                    existing.entity_confidence = max(existing.entity_confidence, cand.entity_confidence)
                    # Merge temporal evidence
                    existing.temporal_metadata.update(cand.temporal_metadata)
                    existing.temporal_fit = max(existing.temporal_fit, cand.temporal_fit)

        # 2. Compute fused relevance score (S_relevance) from Semantic and Lexical
        fused_candidates: List[RetrievalCandidate] = []
        w_sem = channel_weights.get(RetrievalChannel.SEMANTIC, 0.5)
        w_lex = channel_weights.get(RetrievalChannel.LEXICAL, 0.5)
        w_rel_total = w_sem + w_lex if (w_sem + w_lex) > 0 else 1.0

        for mid, cand in merged_map.items():
            sem_norm = cand.normalized_scores.get(RetrievalChannel.SEMANTIC.value, None)
            lex_norm = cand.normalized_scores.get(RetrievalChannel.LEXICAL.value, None)

            if sem_norm is not None and lex_norm is not None:
                cand.fused_score = (w_sem * sem_norm + w_lex * lex_norm) / w_rel_total
            elif sem_norm is not None:
                cand.fused_score = sem_norm
            elif lex_norm is not None:
                cand.fused_score = lex_norm
            else:
                cand.fused_score = 0.0

            fused_candidates.append(cand)

        # Sort by fused score descending
        fused_candidates.sort(key=lambda c: max(c.fused_score, c.entity_confidence, c.temporal_fit), reverse=True)
        return fused_candidates
