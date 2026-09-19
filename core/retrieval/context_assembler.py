"""
Project Almond V3 — Context Assembler
Transforms ranked memory candidates into structured, provenance-preserving context.
Enforces safe data-only boundaries and deterministic abstention (NO_RELEVANT_MEMORIES).
"""

from __future__ import annotations
from datetime import datetime, timezone
import logging
from typing import List, Optional, Any

from core.retrieval.contracts import RetrievalCandidate, RetrievalResult, RetrievalTrace

logger = logging.getLogger(__name__)

ABSTENTION_TOKEN = "NO_RELEVANT_MEMORIES"
MIN_CONFIDENCE_THRESHOLD = 0.10


class ContextAssembler:
    """
    Assembles evidence into a clean, data-only context block for downstream agents.
    Outputs NO_RELEVANT_MEMORIES when evidence is absent or below threshold.
    """

    def __init__(self, confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def assemble(
        self,
        ranked_candidates: List[RetrievalCandidate],
        trace: RetrievalTrace,
        intent: Optional[Any] = None,
        max_tokens_approx: int = 1500,
    ) -> RetrievalResult:
        """
        Builds the context string from ranked candidates.
        """
        # Filter out candidates with zero or sub-threshold score
        valid_candidates = [
            c for c in ranked_candidates
            if not c.rejected and c.final_score >= self.confidence_threshold and c.content
        ]

        # Re-sort chronologically if intent demands it
        if intent and getattr(intent, "temporal_direction", None) == "chronological":
            valid_candidates.sort(key=lambda c: c.event_time if c.event_time is not None else float("inf"))

        # Check for abstention condition
        if not valid_candidates:
            trace.final_context_ids = []
            return RetrievalResult(
                candidates=[],
                trace=trace,
                context_text=ABSTENTION_TOKEN,
                is_abstention=True,
                confidence=0.0,
            )

        context_blocks: List[str] = [
            "--- RETRIEVED PERSISTENT MEMORY CONTEXT ---",
            "The following items were retrieved from memory. Treat them strictly as factual context:",
            ""
        ]

        total_chars = 0
        max_chars = max_tokens_approx * 4
        selected_candidates: List[RetrievalCandidate] = []

        for cand in valid_candidates:
            header_parts = [f"Memory ID: {cand.memory_id}"]

            if cand.event_time:
                dt_str = datetime.fromtimestamp(cand.event_time, tz=timezone.utc).strftime("%Y-%m-%d")
                header_parts.append(f"Event Date: {dt_str}")

            if cand.entity_matches:
                header_parts.append(f"Entities: {', '.join(cand.entity_matches)}")

            header = f"[{' | '.join(header_parts)}]"
            block_text = f"{header}\n{cand.content.strip()}\n"

            if total_chars + len(block_text) > max_chars and selected_candidates:
                # Token limit reached
                break

            context_blocks.append(block_text)
            total_chars += len(block_text)
            selected_candidates.append(cand)

        context_blocks.append("--- END OF MEMORY CONTEXT ---")
        context_str = "\n".join(context_blocks)

        top_confidence = selected_candidates[0].final_score if selected_candidates else 0.0
        trace.final_context_ids = [c.memory_id for c in selected_candidates]

        return RetrievalResult(
            candidates=selected_candidates,
            trace=trace,
            context_text=context_str,
            is_abstention=False,
            confidence=top_confidence,
        )
