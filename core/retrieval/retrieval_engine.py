"""
Project Almond V3 — Retrieval Engine
Master coordinator for the multi-channel retrieval, fusion, constraint reasoning, and ranking pipeline.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Any

from core.retrieval.contracts import (
    RetrievalCandidate, RetrievalChannel, RetrievalQuery, RetrievalResult, RetrievalTrace
)
from core.retrieval.intent_router import IntentRouter, IntentType
from core.retrieval.semantic_retriever import SemanticRetriever
from core.retrieval.lexical_retriever import LexicalRetriever
from core.retrieval.entity_retriever import EntityRetriever
from core.retrieval.temporal_retriever import TemporalRetriever
from core.retrieval.candidate_fusion import CandidateFusion
from core.retrieval.temporal_reasoner import TemporalReasoner
from core.retrieval.ranker_v3 import RankerV3
from core.retrieval.context_assembler import ContextAssembler

from core.memory_store import MemoryStore
from core.knowledge.entity_store import EntityStore
from core.knowledge.timeline_store import TimelineStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    V3 Multi-Channel Retrieval Engine.
    Executes independent retrieval channels, fuses candidates, reasons over temporal constraints,
    applies Ranker V3, and assembles data-only context with a full RetrievalTrace.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        entity_store: Optional[EntityStore] = None,
        timeline_store: Optional[TimelineStore] = None,
    ):
        self.memory_store = memory_store
        self.conn = memory_store._conn
        self.entity_store = entity_store or EntityStore(self.conn)
        self.timeline_store = timeline_store or TimelineStore(self.conn)

        # Initialize pipeline stages
        self.router = IntentRouter()
        self.semantic_retriever = SemanticRetriever(self.memory_store)
        self.lexical_retriever = LexicalRetriever(self.conn)
        self.entity_retriever = EntityRetriever(self.entity_store)
        self.temporal_retriever = TemporalRetriever(self.timeline_store)
        self.fusion = CandidateFusion()
        self.reasoner = TemporalReasoner(self.timeline_store)
        self.ranker = RankerV3()
        self.assembler = ContextAssembler()

    def query(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Execute full multi-channel retrieval for a given query.
        """
        t0 = time.time()
        ref_time = query.effective_reference_time()

        trace = RetrievalTrace(
            query=query.query_text,
            namespace_id=query.namespace_id,
            reference_time=ref_time,
        )

        # ------------------------------------------------------------------
        # Stage 1: Intent Routing
        # ------------------------------------------------------------------
        intent = self.router.analyze(query.query_text)
        trace.detected_intent = intent.intent_type.value
        trace.intent_confidence = intent.confidence

        # ------------------------------------------------------------------
        # Stage 2: Concurrent / Multi-Channel Retrieval
        # ------------------------------------------------------------------
        channel_results: Dict[RetrievalChannel, List[RetrievalCandidate]] = {}

        # 2a. Semantic Channel (Chroma)
        sem_cands = self.semantic_retriever.retrieve(query)
        channel_results[RetrievalChannel.SEMANTIC] = sem_cands
        trace.candidates_per_channel["SEMANTIC"] = len(sem_cands)
        trace.channels_run.append("SEMANTIC")

        # 2b. Lexical Channel (SQLite)
        lex_cands = self.lexical_retriever.retrieve(query)
        channel_results[RetrievalChannel.LEXICAL] = lex_cands
        trace.candidates_per_channel["LEXICAL"] = len(lex_cands)
        trace.channels_run.append("LEXICAL")

        # 2c. Entity Channel (EntityStore)
        ent_cands = self.entity_retriever.retrieve(query, entity_hints=intent.entities_detected)
        channel_results[RetrievalChannel.ENTITY] = ent_cands
        trace.candidates_per_channel["ENTITY"] = len(ent_cands)
        trace.channels_run.append("ENTITY")

        # 2d. Temporal Channel (TimelineStore)
        if intent.temporal_direction == "before" and intent.temporal_anchor:
            # We will also resolve constraints in Reasoner, but retrieve initial pool
            temp_cands = self.temporal_retriever.retrieve_chronological(query)
        elif intent.temporal_direction == "chronological" or intent.temporal_markers:
            temp_cands = self.temporal_retriever.retrieve_chronological(query)
        else:
            temp_cands = self.temporal_retriever.retrieve_chronological(query, limit=10)

        channel_results[RetrievalChannel.TEMPORAL] = temp_cands
        trace.candidates_per_channel["TEMPORAL"] = len(temp_cands)
        trace.channels_run.append("TEMPORAL")

        # ------------------------------------------------------------------
        # Stage 3: Candidate Fusion
        # ------------------------------------------------------------------
        fused_candidates = self.fusion.fuse(channel_results, intent.channel_weights)
        trace.fusion_merged_count = len(fused_candidates)

        # ------------------------------------------------------------------
        # Stage 3.5: Hydrate candidate attributes and compute P_eff at ref_time
        # ------------------------------------------------------------------
        from core.lifecycle.decay import (
            get_policy_for_tag, resolve_anchor_timestamp, compute_delta_t,
            compute_stability_factor, compute_freshness, compute_effective_priority
        )

        for c in fused_candidates:
            block = self.memory_store.get_by_id(c.memory_id)
            if block:
                if not c.content:
                    c.content = block.content
                c.tag = block.tag.value
                c.tier = block.tier.value
                c.importance_score = block.importance_score
                if c.event_time is None:
                    c.event_time = block.event_time
                if c.last_accessed_at is None:
                    c.last_accessed_at = block.last_accessed_at

                policy = get_policy_for_tag(block.tag)
                anchor = resolve_anchor_timestamp(
                    policy.decay_anchor,
                    block.event_time,
                    block.last_accessed_at,
                    block.updated_at,
                    block.created_at
                )
                dt = compute_delta_t(ref_time, anchor)
                s = compute_stability_factor(block.access_count, policy.stability_divisor)
                c.freshness = compute_freshness(dt, policy.decay_rate_lambda, s)
                c.lifecycle_priority = compute_effective_priority(block.importance_score, c.freshness)

        # ------------------------------------------------------------------
        # Stage 4: Temporal / Constraint Reasoning
        # ------------------------------------------------------------------
        reasoning_res = self.reasoner.resolve_constraints(fused_candidates, query, intent)
        trace.temporal_constraints_detected = reasoning_res.constraints_applied
        trace.rejected_candidates = [
            {"memory_id": r.memory_id, "reason": r.rejection_reason}
            for r in reasoning_res.rejected_candidates
        ]

        # ------------------------------------------------------------------
        # Stage 5: Ranking V3
        # ------------------------------------------------------------------
        ranked_candidates = self.ranker.rank(
            reasoning_res.accepted_candidates,
            intent,
            trace=trace,
            top_k=query.top_k,
        )

        # ------------------------------------------------------------------
        # Stage 6: Context Assembly
        # ------------------------------------------------------------------
        trace.duration_ms = (time.time() - t0) * 1000.0
        result = self.assembler.assemble(ranked_candidates, trace, intent=intent)

        # Optional: Save trace to database for observability
        self._save_trace_to_db(trace)

        return result

    def _save_trace_to_db(self, trace: RetrievalTrace) -> None:
        """Persist trace into retrieval_traces table for observability."""
        try:
            import json
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO retrieval_traces (
                        trace_id, namespace_id, query, intent_detected, reference_time,
                        channels_used, candidates_retrieved, candidates_ranked,
                        selected_memory_ids_json, trace_payload_json, duration_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id,
                    trace.namespace_id,
                    trace.query,
                    trace.detected_intent,
                    trace.reference_time,
                    json.dumps(trace.channels_run),
                    sum(trace.candidates_per_channel.values()),
                    len(trace.ranked_candidates),
                    json.dumps(trace.final_context_ids),
                    json.dumps(trace.to_dict()),
                    trace.duration_ms,
                    trace.created_at,
                ))
        except Exception as e:
            logger.debug("Failed to log retrieval trace to db: %s", e)
