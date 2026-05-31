"""
Project Almond — Memory Block Schema
Core data contract for the Temporal Memory Management Unit (T-MMU).
All memory interactions are chunked into this structure before being
scored, tiered, or paged to storage.
"""

from __future__ import annotations

import math
import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    """OS-style storage tier. Maps to L1–L4 in the T-MMU architecture."""
    L1_HOT_CACHE   = "L1_HOT_CACHE"    # System prompts, immutable rules — never evicted
    L2_ACTIVE_RAM  = "L2_ACTIVE_RAM"   # Current context window sent to the LLM
    L3_VIRTUAL_SWAP = "L3_VIRTUAL_SWAP" # SQLite cold storage — retrievable on keyword match
    L4_ARCHIVE     = "L4_ARCHIVE"      # Summarized or pending deletion


class MemoryTag(str, Enum):
    """Semantic category governing decay behavior."""
    CORE_RULE    = "CORE_RULE"      # High stability — decays very slowly
    PROJECT_FACT = "PROJECT_FACT"   # Medium stability — decays on inactivity
    TASK         = "TASK"           # Medium decay — tied to session relevance
    SMALL_TALK   = "SMALL_TALK"     # High decay — ephemeral by design
    USER_PROFILE = "USER_PROFILE"   # Lowest decay — near-permanent identity facts
    EPISODIC     = "EPISODIC"       # Timestamped events; decay tied to recency


# ---------------------------------------------------------------------------
# Decay config — controls lambda (λ) per tag type
# ---------------------------------------------------------------------------

DECAY_CONSTANTS: dict[MemoryTag, float] = {
    MemoryTag.CORE_RULE:    0.001,   # Almost no decay
    MemoryTag.USER_PROFILE: 0.002,
    MemoryTag.PROJECT_FACT: 0.01,
    MemoryTag.TASK:         0.05,
    MemoryTag.EPISODIC:     0.08,
    MemoryTag.SMALL_TALK:   0.20,   # Rapid decay
}

# Stability factor (S) divisor — higher access_count → slower decay
STABILITY_SCALE: float = 10.0


# ---------------------------------------------------------------------------
# Core Schema
# ---------------------------------------------------------------------------

class MemoryBlock(BaseModel):
    """
    A single unit of memory in Project Almond's T-MMU.

    Lifecycle:
        Created → scored → tiered (L1-L4) → paged out to SQLite (L3)
        or summarized/deleted (L4) based on Peff threshold.
    """

    # --- Identity ---
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique memory identifier (UUIDv4)."
    )

    # --- Temporal ---
    created_at: float = Field(
        default_factory=time.time,
        description="Unix epoch timestamp of memory creation."
    )
    last_accessed_at: float = Field(
        default_factory=time.time,
        description="Unix epoch timestamp of most recent retrieval or reinforcement."
    )

    # --- Content ---
    content: str = Field(
        ...,
        min_length=1,
        description="The semantic text payload of this memory."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Dense single-sentence distillation used when paging to L4."
    )

    # --- Classification ---
    tag: MemoryTag = Field(
        ...,
        description="Semantic category — governs decay constant (λ)."
    )
    tier: MemoryTier = Field(
        default=MemoryTier.L2_ACTIVE_RAM,
        description="Current storage tier assignment."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Trigger words for retrieval from L3 virtual swap."
    )

    # --- Scoring Inputs ---
    importance_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Base importance rating (I_base). Range: 1.0 – 10.0."
    )
    access_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this block has been accessed. Acts as stability multiplier (S)."
    )

    # --- Provenance ---
    source: Optional[str] = Field(
        default=None,
        description="Origin of memory: 'user', 'assistant', 'system', or a tool name."
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session this block was created in. Enables session-scoped retrieval."
    )

    # ---------------------------------------------------------------------------
    # Computed fields
    # ---------------------------------------------------------------------------

    @computed_field
    @property
    def delta_t(self) -> float:
        """Days elapsed since last access. Core input to Peff formula."""
        seconds_elapsed = time.time() - self.last_accessed_at
        return seconds_elapsed / 86400.0  # Decay constants (λ) are day-denominated

    @computed_field
    @property
    def lambda_(self) -> float:
        """Decay constant (λ) resolved from tag type."""
        return DECAY_CONSTANTS[self.tag]

    @computed_field
    @property
    def stability_factor(self) -> float:
        """
        Stability factor (S) derived from access_count.
        Higher access → slower effective decay.
        S = access_count / STABILITY_SCALE
        """
        return self.access_count / STABILITY_SCALE

    @computed_field
    @property
    def p_eff(self) -> float:
        """
        Effective Priority Score.

        Formula:
            P_eff = I_base · exp( -(λ / S) · Δt )

        Where:
            I_base  = importance_score
            λ       = decay constant for this tag
            S       = stability factor (access_count / STABILITY_SCALE)
            Δt      = seconds since last access
        """
        exponent = -(self.lambda_ / self.stability_factor) * self.delta_t
        return self.importance_score * math.exp(exponent)

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def l1_requires_no_summary(self) -> MemoryBlock:
        """L1 blocks are immutable — they should never be summarized."""
        if self.tier == MemoryTier.L1_HOT_CACHE and self.summary is not None:
            raise ValueError("L1 (HOT_CACHE) blocks cannot have a summary — they are never evicted.")
        return self

    @model_validator(mode="after")
    def l4_requires_summary(self) -> MemoryBlock:
        """L4 blocks must carry a summary before archival."""
        if self.tier == MemoryTier.L4_ARCHIVE and self.summary is None:
            raise ValueError("L4 (ARCHIVE) blocks must include a summary before archival.")
        return self

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def touch(self) -> None:
        """Reinforce this memory: increment access_count and refresh last_accessed_at."""
        self.access_count += 1
        self.last_accessed_at = time.time()

    def to_context_snippet(self) -> str:
        """
        Returns the string injected into the LLM context window.
        Uses summary if available (L3/L4), otherwise full content.
        """
        return self.summary if self.summary else self.content

    class Config:
        use_enum_values = False  # Keep enum objects, not raw strings