"""
Project Almond — Memory Controller v2
Store-integrated orchestrator. All evictions now persist to SQLite.

Changes from original:
    - FIX 1: Context assembly sorts L2 with reverse=True.
    - FIX 3: Centralized token estimation helper.
    - FIX 4: Duplicate injection guard during page-in.
    - FIX 5: Retrieval pipeline latency tracking.
    - FIX 6: Stateful retrieval session trace saved per cycle.
    - PHASE 2: Fact + entity extraction pipeline on every ingested memory.
    - PHASE 3: Intent-aware retrieval routing via QueryAnalyzer + routers.
              retrieval_pipeline_v2 / RetrievalOptimizer retired.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from memory_block import MemoryBlock, MemoryTag, MemoryTier
from memory_store import MemoryStore

# ── Phase 2: extraction pipeline ──────────────────────────────────────────
from memory_pipeline_v2.memory_classifier import (
    MemoryClassifier, RawMemory as CRaw, MemorySource as CSource,
)
from memory_pipeline_v2.fact_extractor import FactExtractor
from memory_pipeline_v2.entity_extractor import EntityExtractor, EntityRegistry

# ── Phase 3: intent-aware retrieval ───────────────────────────────────────
from memory_pipeline_v2.timeline_index import TimelineIndex
from memory_pipeline_v2.query_analyzer import QueryAnalyzer, IntentType
from memory_pipeline_v2.temporal_retriever import TemporalRetriever
from memory_pipeline_v2.comparison_retriever import ComparisonRetriever
from memory_pipeline_v2.ranking_engine import (
    RankingEngine, RetrievalAuditLog, MemoryMetaStore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Centralised token estimation helper.
    Future: replace with tiktoken or a model-specific tokenizer.
    """
    return len(text) // 4


# ---------------------------------------------------------------------------
# Eviction policy config
# ---------------------------------------------------------------------------

@dataclass
class EvictionPolicy:
    """
    Tunable thresholds for tier transitions.
    All values are P_eff scores. Adjust for ablation studies.
    """
    l2_eviction:   float = 2.0    # Below this → page out to L3
    l3_eviction:   float = 0.5    # Below this → summarise and demote to L4
    l4_deletion:   float = 0.05   # Below this → hard delete
    l2_max_blocks: int   = 20     # Hard cap on active context blocks


# ---------------------------------------------------------------------------
# Controller state snapshot
# ---------------------------------------------------------------------------

@dataclass
class ControllerState:
    cycle_timestamp:        float
    l1_count:               int
    l2_count:               int
    l3_count:               int
    l4_count:               int
    evicted_to_l3:          list[str]
    evicted_to_l4:          list[str]
    deleted:                list[str]
    paged_in:               list[str]
    context_token_estimate: int


# ---------------------------------------------------------------------------
# Memory Controller
# ---------------------------------------------------------------------------

class MemoryController:
    """
    T-MMU orchestrator — backed by SQLite via MemoryStore.

    Also acts as the MemoryStore adapter for the new retrieval modules.
    TemporalRetriever and ComparisonRetriever receive `self` as their
    memory_store argument — no changes required to memory_store.py.
    """

    def __init__(
        self,
        policy:     Optional[EvictionPolicy] = None,
        store:      Optional[MemoryStore]    = None,
        llm_adapter=None,                              # any .complete(prompt, max_tokens)->str
        db_path:    str = "almond_timeline.db",
        audit_db:   str = "almond_audit.db",
    ):
        self.policy = policy or EvictionPolicy()
        self.store  = store

        # ── Phase 2: extraction singletons ────────────────────────────────
        self._classifier  = MemoryClassifier(llm=llm_adapter)
        self._fact_ext    = FactExtractor(llm=llm_adapter)
        self._entity_reg  = EntityRegistry()
        self._entity_ext  = EntityExtractor(registry=self._entity_reg, llm=llm_adapter)
        self._meta_store  = MemoryMetaStore()

        # ── Phase 3: timeline store + retrieval ───────────────────────────
        self._timeline     = TimelineIndex(db_path)
        self._analyzer     = QueryAnalyzer(llm=llm_adapter)
        self._temporal_ret = TemporalRetriever(self._timeline, self._entity_reg, self)
        self._comp_ret     = ComparisonRetriever(self._entity_reg, self)
        self._audit        = RetrievalAuditLog(audit_db)
        self._rank_engine  = RankingEngine(
            entity_registry=self._entity_reg,
            timeline_index=self._timeline,
            meta_store=self._meta_store,
            audit_log=self._audit,
        )
        # Full ranked-candidate ID list from the most recent _smart_page_in
        # call, regardless of hydration status. Drives prompt prioritisation
        # in chat() — see _smart_page_in for why this is separate from
        # paged_in. Starts empty so _assemble_context's fallback (pure
        # recency) is used correctly on the very first turn, before any
        # retrieval has run.
        self._last_ranked_ids: list[str] = []

        # Session trace — kept for observability / eval
        self.last_retrieval_trace: dict[str, Any] = {}

        # Per-stage cumulative timing (ms), for runtime profiling. Reset
        # explicitly via reset_stage_timings() between eval iterations if
        # per-question (rather than cumulative) timing is wanted. Keys are
        # populated lazily by _time_stage() — absent until first hit.
        self.stage_timings_ms: dict[str, float] = {}
        self.stage_call_counts: dict[str, int] = {}

        # Only L1 + L2 live in RAM; L3/L4 are disk-only
        self._l1: dict[str, MemoryBlock] = {}
        self._l2: dict[str, MemoryBlock] = {}
        self._cycle_count: int = 0

        # Rehydrate in-memory state from the store if one was provided.
        # This is what makes the cache restore work: after copying the SQLite
        # DB + Chroma dir and creating a fresh Almond instance, this call
        # loads all previously-ingested blocks and entity registry back into
        # RAM so retrieval works without re-running ingestion LLM calls.
        if self.store:
            self._rehydrate()

    # -----------------------------------------------------------------------
    # TIMING INSTRUMENTATION
    # -----------------------------------------------------------------------
    def _record_stage_time(self, stage: str, elapsed_ms: float) -> None:
        """Accumulate elapsed_ms under `stage` and bump its call count."""
        self.stage_timings_ms[stage] = self.stage_timings_ms.get(stage, 0.0) + elapsed_ms
        self.stage_call_counts[stage] = self.stage_call_counts.get(stage, 0) + 1

    def reset_stage_timings(self) -> None:
        """Zero out accumulated timing data. Call between eval iterations
        if per-question (rather than cumulative-across-the-whole-run) timing
        breakdowns are wanted."""
        self.stage_timings_ms = {}
        self.stage_call_counts = {}

    def get_stage_timing_report(self) -> dict[str, Any]:
        """Return a summary: total/avg ms per stage, sorted by total descending.
        Use this to identify the dominant bottleneck stage at a glance."""
        report = {}
        for stage, total_ms in sorted(
            self.stage_timings_ms.items(), key=lambda kv: kv[1], reverse=True
        ):
            count = self.stage_call_counts.get(stage, 1)
            report[stage] = {
                "total_ms": round(total_ms, 1),
                "calls": count,
                "avg_ms": round(total_ms / count, 1) if count else 0.0,
            }
        return report

    # -----------------------------------------------------------------------
    # MemoryStore adapter interface
    # (used by TemporalRetriever and ComparisonRetriever as their store)
    # -----------------------------------------------------------------------

    def get_by_id(self, memory_id: str) -> Optional[str]:
        """
        Return the text content of a block by ID, or None.

        Resolution order:
          1. L1 RAM dict (fast, O(1))
          2. L2 RAM dict (fast, O(1))
          3. SQLite via get_blocks_by_ids() (disk, only when not in RAM)

        Q7: SQLite fallback is always active — if a memory was injected
        directly to L3 (cold storage, not in RAM dicts), it will be found
        here via SQLite and its content returned correctly.
        """
        block = self.get_block_by_id(memory_id)
        return block.content if block else None

    def get_block_by_id(self, memory_id: str) -> Optional["MemoryBlock"]:
        """
        Return the full MemoryBlock by ID, or None — same resolution order
        as get_by_id() (L1 -> L2 -> SQLite), but preserves metadata
        (created_at, tier, importance_score, etc.) instead of discarding
        it down to a bare string.

        Added so callers like ComparisonRetriever can order/score candidate
        memories by recency or other metadata, not just plain-text content.
        get_by_id() now delegates to this and extracts .content, so existing
        callers (TemporalRetriever, etc.) are unaffected.
        """
        for pool in (self._l1, self._l2):
            if memory_id in pool:
                return pool[memory_id]
        if self.store:
            blocks = self.store.get_blocks_by_ids([memory_id])
            if blocks:
                logger.debug("[GET_BLOCK_BY_ID] %s resolved from SQLite (not in RAM)", memory_id[:8])
                return blocks[0]
        return None

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, str, float]]:
        """
        Thin adapter over MemoryStore.semantic_search_metadata.
        Returns (memory_id, text, score) tuples as the new retrievers expect.

        Bug fix: semantic_search_metadata intentionally omits document content
        (metadata-first pipeline). We resolve content from SQLite after scoring
        so we never drag full content through Chroma unnecessarily.

        Bug fix: search BOTH L2 and L3 tiers. During LongMemEval replay,
        memories live in L2 (never evicted) so a L3-only filter returns nothing.
        """
        if not self.store:
            return []

        results = []
        seen: set[str] = set()

        # Search L2 first (active RAM — present during benchmark replay),
        # then L3 (cold storage — present during normal sessions).
        for tier in (MemoryTier.L2_ACTIVE_RAM, MemoryTier.L3_VIRTUAL_SWAP):
            raw = self.store.semantic_search_metadata(
                query=query,
                tier=tier,
                n_results=top_k,
            )
            for d in raw:
                bid = d["id"]
                if bid in seen:
                    continue
                seen.add(bid)
                # Resolve content from SQLite — content is never in Chroma metadata
                text = self.get_by_id(bid) or ""
                score = 1.0 - d.get("distance", 1.0)
                results.append((bid, text, score))

        # Sort by score descending, cap at top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------------
    # Startup: session resume
    # -----------------------------------------------------------------------

    def _rehydrate(self) -> None:
        assert self.store is not None
        l1_blocks = self.store.get_all(MemoryTier.L1_HOT_CACHE)
        l2_blocks = self.store.get_all(MemoryTier.L2_ACTIVE_RAM)
        for b in l1_blocks:
            self._l1[b.id] = b
        for b in l2_blocks:
            self._l2[b.id] = b
        logger.info(
            "[REHYDRATE] Loaded %d L1 + %d L2 blocks from disk.",
            len(l1_blocks), len(l2_blocks),
        )
        # Phase 2: restore entity registry from disk so cross-session
        # entity resolution works without reprocessing old memories.
        if hasattr(self.store, "load_entity_registry"):
            self.store.load_entity_registry(self._entity_reg)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add(self, block: MemoryBlock) -> None:
        """Add a MemoryBlock to the in-memory pool and persist to disk.

        The tier attribute is normalised to match the dict the block is
        placed into. This prevents the silent mismatch where a block lives
        in self._l2 but still carries tier=L3_VIRTUAL_SWAP — which made
        _build_messages() filter it out of the memory preamble.

        Rule: whatever dict a block enters, its .tier reflects that dict.
        """
        if block.tier == MemoryTier.L1_HOT_CACHE:
            self._l1[block.id] = block
            # L1 tier is intentional — leave as-is
        else:
            # Everything non-L1 lives in the L2 active RAM pool.
            # Normalise the tier attribute so _build_messages() and
            # _assemble_context() both agree on where this block lives.
            block.tier = MemoryTier.L2_ACTIVE_RAM
            self._l2[block.id] = block
        if self.store:
            _t0 = time.time()
            self.store.save(block)
            self._record_stage_time("ingest.store_save", (time.time() - _t0) * 1000)
        logger.debug("[ADD] %s → %s tag=%s", block.id[:8], block.tier.value, block.tag.value)

    def prepare_context(
        self,
        user_message: str,
        query_intent: Optional[MemoryTag] = None,
    ) -> list[MemoryBlock]:
        """
        Main entry point called before every LLM call.

        Order:
          1. Eviction sweep (P_eff — lifecycle, unchanged)
          2. Intent-aware page-in (Phase 3 — replaces RetrievalOptimizer)
          3. L2 cap enforcement
          4. State snapshot + logging
          5. Context assembly
        """
        self._cycle_count += 1
        logger.info("[CYCLE %d] Starting eviction sweep.", self._cycle_count)

        _t0 = time.time()
        evicted_l3, evicted_l4, deleted = self._run_eviction_sweep()
        self._record_stage_time("retrieval.eviction_sweep", (time.time() - _t0) * 1000)

        # Cap BEFORE page-in so retrieved blocks are never immediately evicted.
        # Old order (page-in → cap) allowed _enforce_l2_cap to evict blocks
        # that _smart_page_in just promoted, silently removing retrieved memories
        # before _assemble_context could return them.
        _t0 = time.time()
        self._enforce_l2_cap()
        self._record_stage_time("retrieval.l2_cap_enforce", (time.time() - _t0) * 1000)

        _t0 = time.time()
        paged_in = self._smart_page_in(user_message)
        self._record_stage_time("retrieval.smart_page_in", (time.time() - _t0) * 1000)
        # smart_page_in is the umbrella stage covering: intent classification,
        # entity resolution, retriever dispatch (temporal/comparison/vector),
        # and ranking engine scoring. See query_analyzer/temporal_retriever/
        # comparison_retriever/ranking_engine for finer-grained breakdowns if
        # this stage dominates the report.

        _t0 = time.time()
        l3_count = self.store.tier_counts().get(MemoryTier.L3_VIRTUAL_SWAP.value, 0) if self.store else 0
        l4_count = self.store.tier_counts().get(MemoryTier.L4_ARCHIVE.value, 0)      if self.store else 0
        self._record_stage_time("retrieval.tier_counts", (time.time() - _t0) * 1000)

        state = ControllerState(
            cycle_timestamp=time.time(),
            l1_count=len(self._l1),
            l2_count=len(self._l2),
            l3_count=l3_count,
            l4_count=l4_count,
            evicted_to_l3=evicted_l3,
            evicted_to_l4=evicted_l4,
            deleted=deleted,
            paged_in=paged_in,
            context_token_estimate=self._estimate_tokens(),
        )
        self._log_state(state)
        # Pass the FULL ranked-candidate list (not just paged_in) so they
        # appear first in context regardless of recency or whether they
        # needed fresh hydration. Using paged_in here was the bug: any
        # ranked memory already resident in L2 (the common case for short
        # benchmark sessions) was excluded from paged_in by the duplicate
        # guard in _smart_page_in, so the prompt silently fell back to pure
        # recency and discarded the ranking engine's output whenever the
        # right memories happened to already be active. self._last_ranked_ids
        # captures every ranked candidate regardless of hydration status.
        _t0 = time.time()
        result = self._assemble_context(priority_ids=self._last_ranked_ids)
        self._record_stage_time("retrieval.assemble_context", (time.time() - _t0) * 1000)
        return result

    def ingest_response(
        self,
        content: str,
        tag: MemoryTag,
        importance_score: float,
        keywords: Optional[list[str]] = None,
        session_id: Optional[str]     = None,
    ) -> MemoryBlock:
        """
        Store the LLM reply as a MemoryBlock, then run Phase 2 extraction
        so every stored memory gets structured facts and entity links.
        """
        target_tier = (
            MemoryTier.L1_HOT_CACHE
            if tag == MemoryTag.CORE_RULE
            else MemoryTier.L2_ACTIVE_RAM
        )
        block = MemoryBlock(
            content=content,
            tag=tag,
            importance_score=importance_score,
            keywords=keywords or [],
            source="assistant",
            session_id=session_id,
            tier=target_tier,
        )
        self.add(block)

        # Phase 2: extract facts and entities from every ingested block
        self._run_extraction_pipeline(block)

        return block

    # -----------------------------------------------------------------------
    # Phase 2: extraction pipeline
    # -----------------------------------------------------------------------

    def _run_extraction_pipeline(self, block: MemoryBlock) -> None:
        """
        Run classification → fact extraction → entity extraction on a block.
        Results are stored in the new parallel tables (facts + entity registry).
        This never mutates the MemoryBlock itself.
        """
        ts = datetime.now()

        # Classify (gives us memory_type for downstream use)
        raw = CRaw(
            id=block.id,
            source=CSource.ASSISTANT if block.source == "assistant" else CSource.USER,
            text=block.content,
            timestamp=ts,
            session_id=block.session_id or "default",
            conversation_turn=0,
        )
        _t0 = time.time()
        classified = self._classifier.classify(raw)
        self._record_stage_time("ingest.classify", (time.time() - _t0) * 1000)

        # Extract structured facts
        _t0 = time.time()
        facts = self._fact_ext.extract(
            block.id, block.content, classified.memory_type.value, ts
        )
        self._record_stage_time("ingest.fact_extract", (time.time() - _t0) * 1000)

        # Persist facts to store (if store has save_fact — Phase 2 migration)
        _t0 = time.time()
        if self.store and hasattr(self.store, "save_fact"):
            for fact in facts:
                self.store.save_fact(fact)
        self._record_stage_time("ingest.save_facts", (time.time() - _t0) * 1000)

        # Index temporal events in the timeline
        _t0 = time.time()
        linked = self._entity_ext.extract_and_link(block.id, block.content, ts)
        self._record_stage_time("ingest.entity_extract_link", (time.time() - _t0) * 1000)
        entity_ids = [le.entity.id for le in linked]

        _t0 = time.time()
        for fact in facts:
            self._timeline.store_fact(fact, entity_ids=entity_ids)
        self._record_stage_time("ingest.timeline_store", (time.time() - _t0) * 1000)

        # Persist entities (if store has save_entity — Phase 2 migration)
        _t0 = time.time()
        if self.store and hasattr(self.store, "save_entity"):
            for le in linked:
                self.store.save_entity(le.entity)
        self._record_stage_time("ingest.save_entities", (time.time() - _t0) * 1000)

        # Update meta store for ranking signals
        _t0 = time.time()
        self._meta_store.upsert(
            block.id,
            memory_type=classified.memory_type.value,
            retrieval_count=block.access_count,
            age_days=int(block.delta_t),
            fact_confidences=[f.confidence for f in facts],
        )
        self._record_stage_time("ingest.meta_upsert", (time.time() - _t0) * 1000)

        logger.debug(
            "[EXTRACT] %s → type=%s facts=%d entities=%d",
            block.id[:8], classified.memory_type.value, len(facts), len(linked),
        )

    # -----------------------------------------------------------------------
    # Phase 3: intent-aware page-in (replaces _semantic_page_in)
    # -----------------------------------------------------------------------

    def _smart_page_in(self, user_message: str) -> list[str]:
        """
        Routes the query to the correct retriever based on intent, then uses
        ranking_engine to score and order candidates before page-in.

        Replaces: RetrievalOptimizer.rerank_and_filter()
        Preserves: duplicate guard, block.touch(), tier promotion, store.save()
        """
        if not self.store:
            return []

        paged_in: list[str] = []
        t0 = time.time()

        # ── Step 1: Analyse intent ─────────────────────────────────────────
        intent = self._analyzer.analyze(user_message)
        intent_time = time.time() - t0
        self._record_stage_time("retrieval.intent_analysis", intent_time * 1000)
        logger.info(
            "[RETRIEVAL] intent=%s conf=%.2f entities=%s",
            intent.intent_type.value, intent.confidence, intent.entities_mentioned,
        )

        # ── Step 2: Route to correct retriever ────────────────────────────
        t_route = time.time()
        from memory_pipeline_v2.temporal_retriever import RetrievedMemory

        if intent.intent_type == IntentType.TEMPORAL:
            raw_result = self._temporal_ret.retrieve(intent)
            candidates = raw_result.memories
            used_fallback = raw_result.used_fallback

        elif intent.intent_type == IntentType.COMPARISON:
            raw_result = self._comp_ret.retrieve(intent)
            candidates = raw_result.flat_memories
            used_fallback = raw_result.used_fallback

        elif intent.intent_type == IntentType.EVENT:
            raw_result = self._temporal_ret.retrieve(intent)
            candidates = raw_result.memories
            used_fallback = raw_result.used_fallback

        else:
            # FACTUAL, RELATIONSHIP, AMBIGUOUS → semantic search via adapter
            sem_results = self.semantic_search(user_message, top_k=20)
            candidates = [
                RetrievedMemory(memory_id=mid, text=text, score=score, source="vector")
                for mid, text, score in sem_results
            ]
            used_fallback = False

        route_time = time.time() - t_route
        self._record_stage_time(f"retrieval.route_{intent.intent_type.value.lower()}", route_time * 1000)

        # ── Step 3: Rank with intent-weighted signals ──────────────────────
        t_rank = time.time()
        ranked = self._rank_engine.rank(candidates, intent, used_fallback=used_fallback)
        rank_time = time.time() - t_rank
        self._record_stage_time("retrieval.rank_engine", rank_time * 1000)

        # ── Step 4: Hydrate and promote to L2 ─────────────────────────────
        t_hydrate = time.time()
        ids_to_hydrate = [
            r.memory_id for r in ranked
            if r.memory_id not in self._l2      # FIX 4: duplicate guard
        ]
        hydrated = self.store.get_blocks_by_ids(ids_to_hydrate) if ids_to_hydrate else []

        # Orphan detection: IDs Chroma knows about but SQLite doesn't.
        # This happens when Chroma persists across runs but SQLite is reset.
        # Orphaned IDs produce Chroma candidates that can never be hydrated,
        # making retrieved_count misleadingly high while paged_in stays empty.
        #
        # Q5: This fires and logs a WARNING whenever orphans are found.
        # Q6: If ALL candidates are orphans, hydrated=[] → paged_in stays empty.
        #     Retrieval gracefully returns zero results (no crash, no corrupt state).
        #     The orphans are cleaned from Chroma so subsequent queries improve.
        hydrated_ids = {b.id for b in hydrated}
        orphaned     = [mid for mid in ids_to_hydrate if mid not in hydrated_ids]
        if orphaned:
            logger.warning(
                "[ORPHAN] %d/%d IDs are Chroma ghosts (no SQLite record). "                "Chroma/SQLite desync detected — cleaning %d stale entries.",
                len(orphaned), len(ids_to_hydrate), len(orphaned),
            )
            cleaned = 0
            for oid in orphaned:
                try:
                    self.store._collection.delete(ids=[oid])
                    cleaned += 1
                except Exception as e:
                    logger.debug("[ORPHAN] Could not delete %s from Chroma: %s", oid[:8], e)
            logger.warning("[ORPHAN] Cleaned %d ghost entries from Chroma.", cleaned)

            # Q6: 100% orphan case — retrieval returns nothing but doesn't crash.
            # Log explicitly so it's visible in eval output.
            if len(orphaned) == len(ids_to_hydrate) and ids_to_hydrate:
                logger.warning(
                    "[ORPHAN] ALL %d candidates were ghosts. "                    "Run eval after deleting almond_chroma_db/ to resync.",
                    len(ids_to_hydrate),
                )

        for block in hydrated:
            if block.id in self._l2:
                continue
            block.touch()
            block.tier = MemoryTier.L2_ACTIVE_RAM
            self._l2[block.id] = block
            self.store.save(block)
            paged_in.append(block.id)
            logger.info("[PAGE-IN] %s promoted. intent=%s", block.id[:8], intent.intent_type.value)

        # ranked_ids captures EVERY memory the ranking engine selected as
        # relevant, regardless of whether it needed fresh hydration from
        # cold storage. paged_in only tracks the subset that needed
        # promotion (used for diagnostics/state logging below) - it
        # previously also drove prompt prioritisation in _assemble_context,
        # which meant any ranked memory that was already resident in L2
        # (the common case for short benchmark sessions, where almost
        # everything ends up L2-resident) got silently excluded from
        # priority_ids and the prompt fell back to pure recency, discarding
        # the ranking engine's output entirely. See chat() below: this list
        # is now what actually drives priority_ids, not paged_in.
        self._last_ranked_ids = [r.memory_id for r in ranked]

        hydrate_time = time.time() - t_hydrate
        self._record_stage_time("retrieval.hydrate_promote", hydrate_time * 1000)
        total_time   = time.time() - t0

        # ── Step 5: Save retrieval trace (FIX 6) ──────────────────────────
        self.last_retrieval_trace = {
            "query":             user_message,
            "intent_type":       intent.intent_type.value,
            "intent_confidence": round(intent.confidence, 3),
            "used_fallback":     used_fallback,
            "total_candidates":  len(candidates),
            "ranked_count":      len(ranked),
            "paged_in":          paged_in,
            "top5_scores":       [round(r.final_score, 4) for r in ranked[:5]],
            "top5_reasoning":    [r.reasoning for r in ranked[:5]],
            "route_ms":          round(route_time * 1000, 2),
            "rank_ms":           round(rank_time * 1000, 2),
            "hydrate_ms":        round(hydrate_time * 1000, 2),
            "total_ms":          round(total_time * 1000, 2),
        }

        logger.info(
            "[RETRIEVAL TIMING] route=%.1fms rank=%.1fms hydrate=%.1fms | "
            "candidates=%d ranked=%d paged_in=%d",
            route_time * 1000, rank_time * 1000, hydrate_time * 1000,
            len(candidates), len(ranked), len(paged_in),
        )

        return paged_in

    # -----------------------------------------------------------------------
    # Eviction logic (P_eff — unchanged)
    # -----------------------------------------------------------------------

    def _run_eviction_sweep(self) -> tuple[list[str], list[str], list[str]]:
        evicted_to_l3: list[str] = []
        evicted_to_l4: list[str] = []
        deleted:       list[str] = []

        # L2 → L3
        for bid, block in list(self._l2.items()):
            if block.p_eff < self.policy.l2_eviction:
                block.tier = MemoryTier.L3_VIRTUAL_SWAP
                del self._l2[bid]
                evicted_to_l3.append(bid)
                if self.store:
                    self.store.save(block)
                logger.debug("[EVICT L2→L3] %s peff=%.4f", bid[:8], block.p_eff)

        # L3 → L4, L4 → delete
        if self.store:
            for block in self.store.get_all(MemoryTier.L3_VIRTUAL_SWAP):
                if block.p_eff < self.policy.l3_eviction:
                    block.summary = self._summarize(block)
                    block.tier    = MemoryTier.L4_ARCHIVE
                    self.store.save(block)
                    evicted_to_l4.append(block.id)
                    logger.debug("[EVICT L3→L4] %s peff=%.4f", block.id[:8], block.p_eff)

            for block in self.store.get_all(MemoryTier.L4_ARCHIVE):
                if block.p_eff < self.policy.l4_deletion:
                    self.store.delete(block.id)
                    deleted.append(block.id)
                    logger.debug("[DELETE] %s peff=%.4f", block.id[:8], block.p_eff)

        return evicted_to_l3, evicted_to_l4, deleted

    def _enforce_l2_cap(self) -> None:
        if len(self._l2) <= self.policy.l2_max_blocks:
            return
        # Sort by (p_eff ASC, last_accessed_at ASC) so:
        # - lowest p_eff goes first (original behaviour)
        # - ties broken by recency — most recently accessed blocks survive
        # This prevents arbitrary eviction when all blocks share the same
        # importance_score (as in benchmark replay where all = 6.0).
        sorted_blocks = sorted(
            self._l2.values(),
            key=lambda b: (b.p_eff, b.last_accessed_at)
        )
        overflow = len(self._l2) - self.policy.l2_max_blocks
        for block in sorted_blocks[:overflow]:
            block.tier = MemoryTier.L3_VIRTUAL_SWAP
            del self._l2[block.id]
            if self.store:
                self.store.save(block)
            logger.debug("[CAP EVICT] %s peff=%.4f accessed=%.0f",
                         block.id[:8], block.p_eff, block.last_accessed_at)

    # -----------------------------------------------------------------------
    # Context assembly (unchanged — FIX 1 preserved)
    # -----------------------------------------------------------------------

    def _assemble_context(
        self,
        priority_ids: list[str] | None = None,
    ) -> list[MemoryBlock]:
        """
        Build context list for the LLM prompt.

        priority_ids: memory IDs from _smart_page_in that must appear first.
        Retrieved memories are placed at the front of the L2 list regardless
        of last_accessed_at, so they are never displaced by older but more
        recently touched blocks.
        """
        l1 = list(self._l1.values())

        priority_set  = set(priority_ids or [])
        priority_blocks = [
            self._l2[bid] for bid in (priority_ids or [])
            if bid in self._l2
        ]
        remaining = sorted(
            [b for b in self._l2.values() if b.id not in priority_set],
            key=lambda b: b.last_accessed_at,
            reverse=True,
        )
        return l1 + priority_blocks + remaining

    # -----------------------------------------------------------------------
    # Helpers (unchanged)
    # -----------------------------------------------------------------------

    def _summarize(self, block: MemoryBlock) -> str:
        words     = block.content.split()
        truncated = " ".join(words[:20])
        return f"[ARCHIVED] {truncated}{'...' if len(words) > 20 else ''}"

    def _estimate_tokens(self) -> int:
        """FIX 3: centralised estimation."""
        return sum(
            estimate_tokens(b.to_context_snippet())
            for pool in (self._l1, self._l2)
            for b in pool.values()
        )

    def _log_state(self, state: ControllerState) -> None:
        logger.info(
            "[STATE] L1=%d L2=%d L3=%d L4=%d | "
            "evicted→L3=%d evicted→L4=%d deleted=%d paged_in=%d | ~%d tokens",
            state.l1_count, state.l2_count, state.l3_count, state.l4_count,
            len(state.evicted_to_l3), len(state.evicted_to_l4),
            len(state.deleted), len(state.paged_in),
            state.context_token_estimate,
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @property
    def l2_count(self) -> int:
        """Current L2 block count. Read AFTER prepare_context() for l2_peak."""
        return len(self._l2)

    @property
    def l1_count(self) -> int:
        """Current L1 block count."""
        return len(self._l1)

    def get_retrieval_trace(self) -> dict:
        """
        Return last retrieval trace enriched with post-page-in L2 state.
        Eval should call this instead of reading last_retrieval_trace directly,
        so l2_peak is always measured after hydration completes.
        """
        trace = dict(self.last_retrieval_trace)
        trace["l2_count_after_pagein"] = len(self._l2)
        trace["l1_count"]              = len(self._l1)
        return trace

    def dump_pool(self, tier: Optional[MemoryTier] = None) -> list[dict]:
        if self.store:
            blocks = self.store.get_all(tier)
            ram    = {**self._l1, **self._l2}
            blocks = [ram.get(b.id, b) for b in blocks]
        else:
            pools   = {MemoryTier.L1_HOT_CACHE: self._l1, MemoryTier.L2_ACTIVE_RAM: self._l2}
            targets = {tier: pools[tier]} if tier and tier in pools else pools
            blocks  = [b for pool in targets.values() for b in pool.values()]

        return sorted([
            {
                "id":               b.id,
                "tier":             b.tier.value,
                "tag":              b.tag.value,
                "p_eff":            round(b.p_eff, 6),
                "delta_t_days":     round(b.delta_t, 4),
                "access_count":     b.access_count,
                "importance_score": b.importance_score,
                "content_preview":  b.content[:60],
            }
            for b in blocks
        ], key=lambda x: x["p_eff"], reverse=True)