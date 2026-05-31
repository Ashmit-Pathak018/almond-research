"""
Project Almond — Memory Controller v2
Store-integrated orchestrator. All evictions now persist to SQLite.

Changes from v1/v2-Base:
    - FIX 1: Context assembly now sorts L2 with reverse=True.
    - FIX 3: Centralized token estimation helper introduced.
    - FIX 4: Duplicate injection guard added during page-in.
    - FIX 5: Retrieval pipeline latency tracking added.
    - FIX 6: Stateful retrieval session trace saved per cycle.
    - SOTA: Semantic Search Metadata-first pipeline integration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from memory_block import MemoryBlock, MemoryTag, MemoryTier
from memory_store import MemoryStore
from retrieval_pipeline_v2 import RetrievalOptimizer


# ---------------------------------------------------------------------------
# Global Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Centralized token estimation helper.
    Future: Replace with tiktoken or model-specific tokenizer (e.g., Llama 3).
    """
    return len(text) // 4


# ---------------------------------------------------------------------------
# Eviction Threshold Config
# ---------------------------------------------------------------------------

@dataclass
class EvictionPolicy:
    """
    Tunable thresholds for tier transitions.
    All values are Peff scores. Adjust for ablation studies.
    """
    l2_eviction:   float = 2.0   # Below this → page out to L3
    l3_eviction:   float = 0.5   # Below this → summarize and demote to L4
    l4_deletion:   float = 0.05  # Below this → hard delete
    l2_max_blocks: int   = 20    # Hard cap on active context blocks

logger = logging.getLogger(__name__)


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


class MemoryController:
    """
    T-MMU orchestrator — backed by SQLite via MemoryStore.
    """

    def __init__(
        self,
        policy: Optional[EvictionPolicy] = None,
        store:  Optional[MemoryStore]    = None,
    ):
        self.policy = policy or EvictionPolicy()
        self.store  = store
        
        # Stateful optimizer allows tracking rejection analytics over a session
        self.optimizer = RetrievalOptimizer(min_similarity=0.45)
        
        # Session Analytics Trace
        self.last_retrieval_trace: Dict[str, Any] = {}

        # Only L1 + L2 live in RAM — L3/L4 are disk-only
        self._l1: dict[str, MemoryBlock] = {}
        self._l2: dict[str, MemoryBlock] = {}

        self._cycle_count: int = 0

        if self.store:
            self._rehydrate()

    # -----------------------------------------------------------------------
    # Startup: Session Resume
    # -----------------------------------------------------------------------

    def _rehydrate(self) -> None:
        """
        On startup, load L1 and L2 blocks from the DB back into RAM.
        """
        assert self.store is not None

        l1_blocks = self.store.get_all(MemoryTier.L1_HOT_CACHE)
        l2_blocks = self.store.get_all(MemoryTier.L2_ACTIVE_RAM)

        for b in l1_blocks:
            self._l1[b.id] = b
        for b in l2_blocks:
            self._l2[b.id] = b

        logger.info(
            f"[REHYDRATE] Loaded {len(l1_blocks)} L1 + "
            f"{len(l2_blocks)} L2 blocks from disk."
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add(self, block: MemoryBlock) -> None:
        """
        Add a MemoryBlock to the in-memory pool and persist to disk.
        """
        if block.tier == MemoryTier.L1_HOT_CACHE:
            self._l1[block.id] = block
        else:
            self._l2[block.id] = block

        if self.store:
            self.store.save(block)

        logger.debug(f"[ADD] {block.id[:8]} → {block.tier.value} tag={block.tag.value}")

    def prepare_context(self, user_message: str, query_intent: Optional[MemoryTag] = None) -> list[MemoryBlock]:
        """
        Main entry point before every LLM call.
        """
        self._cycle_count += 1
        logger.info(f"[CYCLE {self._cycle_count}] Starting eviction sweep.")

        evicted_l3, evicted_l4, deleted = self._run_eviction_sweep()
        
        # SOTA Semantic Search triggered here
        paged_in = self._semantic_page_in(user_message, query_intent)
        
        self._enforce_l2_cap()

        # Tier counts for state: L3/L4 come from DB
        l3_count = self.store.tier_counts().get(MemoryTier.L3_VIRTUAL_SWAP.value, 0) if self.store else 0
        l4_count = self.store.tier_counts().get(MemoryTier.L4_ARCHIVE.value, 0)      if self.store else 0

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
        return self._assemble_context()

    def ingest_response(
        self,
        content: str,
        tag: MemoryTag,
        importance_score: float,
        keywords: Optional[list[str]] = None,
        session_id: Optional[str]     = None,
    ) -> MemoryBlock:
        target_tier = MemoryTier.L1_HOT_CACHE if tag == MemoryTag.CORE_RULE else MemoryTier.L2_ACTIVE_RAM

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
        return block

    # -----------------------------------------------------------------------
    # Eviction Logic
    # -----------------------------------------------------------------------

    def _run_eviction_sweep(self) -> tuple[list[str], list[str], list[str]]:
        evicted_to_l3: list[str] = []
        evicted_to_l4: list[str] = []
        deleted:       list[str] = []

        # --- L2 → L3 ---
        for bid, block in list(self._l2.items()):
            if block.p_eff < self.policy.l2_eviction:
                block.tier = MemoryTier.L3_VIRTUAL_SWAP
                del self._l2[bid]
                evicted_to_l3.append(bid)

                if self.store:
                    self.store.save(block)
                logger.debug(f"[EVICT L2→L3] {bid[:8]} peff={block.p_eff:.4f}")

        # --- L3 → L4 (disk-only, no RAM pool to update) ---
        if self.store:
            l3_blocks = self.store.get_all(MemoryTier.L3_VIRTUAL_SWAP)
            for block in l3_blocks:
                if block.p_eff < self.policy.l3_eviction:
                    block.summary = self._summarize(block)
                    block.tier    = MemoryTier.L4_ARCHIVE
                    self.store.save(block)
                    evicted_to_l4.append(block.id)
                    logger.debug(f"[EVICT L3→L4] {block.id[:8]} peff={block.p_eff:.4f}")

            # --- L4 → Delete ---
            l4_blocks = self.store.get_all(MemoryTier.L4_ARCHIVE)
            for block in l4_blocks:
                if block.p_eff < self.policy.l4_deletion:
                    self.store.delete(block.id)
                    deleted.append(block.id)
                    logger.debug(f"[DELETE] {block.id[:8]} peff={block.p_eff:.4f}")

        return evicted_to_l3, evicted_to_l4, deleted

    def _enforce_l2_cap(self) -> None:
        if len(self._l2) <= self.policy.l2_max_blocks:
            return

        sorted_blocks = sorted(self._l2.values(), key=lambda b: b.p_eff)
        overflow = len(self._l2) - self.policy.l2_max_blocks

        for block in sorted_blocks[:overflow]:
            block.tier = MemoryTier.L3_VIRTUAL_SWAP
            del self._l2[block.id]

            if self.store:
                self.store.save(block)
            logger.debug(f"[CAP EVICT] {block.id[:8]} peff={block.p_eff:.4f}")

    # -----------------------------------------------------------------------
    # Page-In (L3 DB → L2 RAM)
    # -----------------------------------------------------------------------

    def _semantic_page_in(self, user_message: str, query_intent: Optional[MemoryTag] = None) -> list[str]:
        if not self.store:
            return []

        paged_in: list[str] = []
        
        # FIX 5: TIMING - Semantic Net
        t0 = time.time()
        raw_results = self.store.semantic_search_metadata(
            query=user_message,
            tier=MemoryTier.L3_VIRTUAL_SWAP,
            n_results=20
        )
        semantic_time = time.time() - t0

        # FIX 5: TIMING - Reranker
        t1 = time.time()
        intent_str = query_intent.value if query_intent else ""
        survivors = self.optimizer.rerank_and_filter(user_message,raw_results,time.time(),intent=intent_str)
        rerank_time = time.time() - t1

        # FIX 5: TIMING - Targeted Hydration
        t2 = time.time()
        
        # FIX 4: Duplicate injection guard. Only hydrate blocks not already in L2.
        survivor_ids = [s["id"] for s in survivors]
        needed_ids = [bid for bid in survivor_ids if bid not in self._l2]
        
        hydrated_blocks = self.store.get_blocks_by_ids(needed_ids)
        print("\n===== RETRIEVED MEMORIES =====")
        for block in hydrated_blocks:
            print(f"[{block.tag}] {block.content[:250]}")
        hydration_time = time.time() - t2

        block_traces = {s["id"]: s["retrieval_trace"] for s in survivors}

        for block in hydrated_blocks:
            # Final sanity check against duplicates
            if block.id in self._l2:
                continue

            trace = block_traces.get(block.id, {})
            
            block.touch()
            block.tier = MemoryTier.L2_ACTIVE_RAM
            self._l2[block.id] = block

            self.store.save(block)
            paged_in.append(block.id)
            hybrid_score = trace.get('final_hybrid_score', 0.0)
            logger.info(f"[PAGE-IN] {block.id[:8]} promoted. Hybrid Score: {hybrid_score:.4f}")

        # FIX 6: Save retrieval session trace for observability
        self.last_retrieval_trace = {
            "query": user_message,
            "intent": intent_str,
            "semantic_time_ms": round(semantic_time * 1000, 2),
            "rerank_time_ms": round(rerank_time * 1000, 2),
            "hydration_time_ms": round(hydration_time * 1000, 2),
            "total_candidates": len(raw_results),
            "accepted": paged_in,
            "rejections": self.optimizer.latest_rejections,
        }
        
        logger.info(
            f"[RETRIEVAL TIMING] Semantic: {semantic_time*1000:.1f}ms | "
            f"Rerank: {rerank_time*1000:.1f}ms | "
            f"Hydrate: {hydration_time*1000:.1f}ms"
        )

        return paged_in

    # -----------------------------------------------------------------------
    # Context Assembly
    # -----------------------------------------------------------------------

    def _assemble_context(self) -> list[MemoryBlock]:
        """
        FIX 1: L2 is now sorted with reverse=True.
        Ensures temporal locality maps accurately against Transformer attention spans.
        """
        l1 = list(self._l1.values())
        l2 = sorted(self._l2.values(), key=lambda b: b.last_accessed_at, reverse=True)
        return l1 + l2

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _summarize(self, block: MemoryBlock) -> str:
        words = block.content.split()
        truncated = " ".join(words[:20])
        return f"[ARCHIVED] {truncated}{'...' if len(words) > 20 else ''}"

    def _estimate_tokens(self) -> int:
        """FIX 3: Uses centralized estimation helper"""
        return sum(
            estimate_tokens(b.to_context_snippet())
            for pool in (self._l1, self._l2)
            for b in pool.values()
        )

    def _log_state(self, state: ControllerState) -> None:
        logger.info(
            f"[STATE] L1={state.l1_count} L2={state.l2_count} "
            f"L3={state.l3_count} L4={state.l4_count} | "
            f"evicted→L3={len(state.evicted_to_l3)} "
            f"evicted→L4={len(state.evicted_to_l4)} "
            f"deleted={len(state.deleted)} "
            f"paged_in={len(state.paged_in)} | "
            f"~{state.context_token_estimate} tokens"
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def dump_pool(self, tier: Optional[MemoryTier] = None) -> list[dict]:
        if self.store:
            blocks = self.store.get_all(tier)
            ram = {**self._l1, **self._l2}
            blocks = [ram.get(b.id, b) for b in blocks]
        else:
            pools = {
                MemoryTier.L1_HOT_CACHE:  self._l1,
                MemoryTier.L2_ACTIVE_RAM: self._l2,
            }
            targets = {tier: pools[tier]} if tier and tier in pools else pools
            blocks  = [b for pool in targets.values() for b in pool.values()]

        return sorted([
            {
                "id":              b.id,
                "tier":            b.tier.value,
                "tag":             b.tag.value,
                "p_eff":           round(b.p_eff, 6),
                "delta_t_days":    round(b.delta_t, 4),
                "access_count":    b.access_count,
                "importance_score":b.importance_score,
                "content_preview": b.content[:60],
            }
            for b in blocks
        ], key=lambda x: x["p_eff"], reverse=True)