"""
Project Almond V3 — Intent Router
Classifies incoming query intent into SEMANTIC, TEMPORAL, ENTITY, COMPARISON, or HYBRID.
Assigns channel activation weights without permanently excluding candidate channels.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from core.retrieval.contracts import RetrievalChannel


class IntentType(str, Enum):
    SEMANTIC = "SEMANTIC"
    TEMPORAL = "TEMPORAL"
    ENTITY = "ENTITY"
    COMPARISON = "COMPARISON"
    HYBRID = "HYBRID"


@dataclass
class IntentAnalysis:
    """Structured result of intent classification."""
    intent_type: IntentType
    confidence: float
    channel_weights: Dict[RetrievalChannel, float]
    temporal_markers: List[str] = field(default_factory=list)
    temporal_direction: Optional[str] = None  # "before", "after", "between", "chronological"
    temporal_anchor: Optional[str] = None
    entities_detected: List[str] = field(default_factory=list)
    comparison_targets: List[str] = field(default_factory=list)


# Deterministic pattern sets
TEMPORAL_BEFORE_PATTERNS = [
    r"\bbefore\s+(.+)",
    r"\bprior to\s+(.+)",
    r"\bpreceding\s+(.+)",
    r"\bleading up to\s+(.+)",
]

TEMPORAL_AFTER_PATTERNS = [
    r"\bafter\s+(.+)",
    r"\bfollowing\s+(.+)",
    r"\bsubsequent to\s+(.+)",
    r"\bsince\s+(.+)",
]

TEMPORAL_ORDER_PATTERNS = [
    r"\bchronological(?:ly)?\b",
    r"\btimeline\b",
    r"\bwhich (?:event|thing|one|vehicle) (?:did i|happened) first\b",
    r"\bwhat happened first\b",
    r"\bwhat was (?:the )?first\b",
    r"\border of events\b",
    r"\bin order\b",
]

COMPARISON_PATTERNS = [
    r"\bcompare\s+(.+?)\s+(?:with|to|and|vs|versus)\s+(.+)",
    r"\bwhat (?:is|was) the difference between\s+(.+?)\s+and\s+(.+)",
    r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+)",
    r"\bplanned vs\.?\s+(?:what was|what actually)?\s*(.+)",
]

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]


class IntentRouter:
    """
    Deterministic query intent router for V3 retrieval.
    Routes queries to proper channels and allocates weight budgets.
    """

    def __init__(self):
        pass

    def analyze(self, query: str) -> IntentAnalysis:
        q = query.strip()
        q_lower = q.lower()

        # 1. Check for comparison patterns
        comp_targets = []
        is_comparison = False
        for pat in COMPARISON_PATTERNS:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                is_comparison = True
                comp_targets = [g.strip() for g in m.groups() if g]
                break

        if is_comparison:
            # Comparison query uses both Temporal & Semantic/Lexical
            return IntentAnalysis(
                intent_type=IntentType.COMPARISON,
                confidence=0.90,
                channel_weights={
                    RetrievalChannel.SEMANTIC: 0.35,
                    RetrievalChannel.LEXICAL: 0.25,
                    RetrievalChannel.TEMPORAL: 0.30,
                    RetrievalChannel.ENTITY: 0.10,
                },
                comparison_targets=comp_targets,
                temporal_markers=["comparison"]
            )

        # 2. Check for explicit temporal direction & constraints
        temporal_markers = []
        temporal_direction = None
        temporal_anchor = None

        for pat in TEMPORAL_BEFORE_PATTERNS:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                temporal_direction = "before"
                temporal_anchor = m.group(1).strip().rstrip("?.")
                temporal_markers.append("before")
                break

        if not temporal_direction:
            for pat in TEMPORAL_AFTER_PATTERNS:
                m = re.search(pat, q, re.IGNORECASE)
                if m:
                    temporal_direction = "after"
                    temporal_anchor = m.group(1).strip().rstrip("?.")
                    temporal_markers.append("after")
                    break

        for pat in TEMPORAL_ORDER_PATTERNS:
            if re.search(pat, q_lower):
                temporal_markers.append("order")
                if not temporal_direction:
                    temporal_direction = "chronological"

        for month in MONTH_NAMES:
            if re.search(r"\b" + month + r"\b", q_lower):
                temporal_markers.append(month)

        if re.search(r"\b(?:yesterday|today|last week|last month|in 202\d|in 201\d)\b", q_lower):
            temporal_markers.append("relative_date")

        # 3. Detect candidate entities (e.g. capitalized tokens, names)
        entities_detected = []
        words = q.split()
        for idx, w in enumerate(words):
            clean_w = re.sub(r"[^\w]", "", w)
            # Capitalized word that is not at the very start of the sentence, or known proper names
            if clean_w and clean_w[0].isupper() and (idx > 0 or clean_w.lower() in ["bob", "alice", "postgresql", "mysql"]):
                if clean_w.lower() not in ["what", "how", "when", "why", "where", "who", "did", "was", "the", "i", "compare"]:
                    entities_detected.append(clean_w)

        # 4. Resolve dominant intent
        has_temporal = bool(temporal_direction or len(temporal_markers) >= 1)
        has_entities = bool(len(entities_detected) >= 1)

        if has_temporal and has_entities:
            return IntentAnalysis(
                intent_type=IntentType.HYBRID,
                confidence=0.85,
                channel_weights={
                    RetrievalChannel.SEMANTIC: 0.25,
                    RetrievalChannel.LEXICAL: 0.25,
                    RetrievalChannel.ENTITY: 0.25,
                    RetrievalChannel.TEMPORAL: 0.25,
                },
                temporal_markers=temporal_markers,
                temporal_direction=temporal_direction,
                temporal_anchor=temporal_anchor,
                entities_detected=entities_detected,
            )

        elif has_temporal:
            return IntentAnalysis(
                intent_type=IntentType.TEMPORAL,
                confidence=0.88,
                channel_weights={
                    RetrievalChannel.TEMPORAL: 0.50,
                    RetrievalChannel.SEMANTIC: 0.25,
                    RetrievalChannel.LEXICAL: 0.20,
                    RetrievalChannel.ENTITY: 0.05,
                },
                temporal_markers=temporal_markers,
                temporal_direction=temporal_direction,
                temporal_anchor=temporal_anchor,
                entities_detected=entities_detected,
            )

        elif has_entities:
            return IntentAnalysis(
                intent_type=IntentType.ENTITY,
                confidence=0.85,
                channel_weights={
                    RetrievalChannel.ENTITY: 0.45,
                    RetrievalChannel.SEMANTIC: 0.30,
                    RetrievalChannel.LEXICAL: 0.20,
                    RetrievalChannel.TEMPORAL: 0.05,
                },
                entities_detected=entities_detected,
            )

        else:
            # Default SEMANTIC / Lexical hybrid
            return IntentAnalysis(
                intent_type=IntentType.SEMANTIC,
                confidence=0.80,
                channel_weights={
                    RetrievalChannel.SEMANTIC: 0.55,
                    RetrievalChannel.LEXICAL: 0.35,
                    RetrievalChannel.ENTITY: 0.05,
                    RetrievalChannel.TEMPORAL: 0.05,
                },
            )
