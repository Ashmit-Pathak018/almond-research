"""
query_analyzer.py
-----------------
Phase 2 — Query intent analysis.

Every incoming question is analyzed before retrieval begins. The intent
determines which retrieval path runs, which ranking weights apply, and
whether a fallback retriever should be armed.

Intent types
------------
FACTUAL      "What laptop do I own?"  "Where do I work?"
             → semantic search over USER_FACT, PROJECT_FACT
TEMPORAL     "Which did I get first?"  "When did I buy the Dell?"
             → timeline index query
EVENT        "What happened after I serviced my car?"
             → timeline + event memories
COMPARISON   "Samsung or Dell — which do I prefer?"
             → multi-entity fetch for both targets
RELATIONSHIP "Who introduced me to machine learning?"
             → entity graph traversal

Design decisions
----------------
1. Heuristic pre-pass before LLM.
   Temporal and comparison markers are distinctive enough to catch without
   a model call in the majority of cases.

2. Secondary intent for router fallback.
   When primary confidence < 0.70, the router should run both the primary
   retriever and the secondary retriever and merge results.

3. Entity and temporal marker extraction in the same pass.
   The query_analyzer hands the router everything it needs in one object
   so the router doesn't need to re-parse the query.

4. Ambiguity is a first-class output, not an error.
   A query can return intent_type="AMBIGUOUS" with two candidate intents.
   The router handles this by running both retrievers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    FACTUAL      = "FACTUAL"
    TEMPORAL     = "TEMPORAL"
    EVENT        = "EVENT"
    COMPARISON   = "COMPARISON"
    RELATIONSHIP = "RELATIONSHIP"
    AMBIGUOUS    = "AMBIGUOUS"   # router runs top-2 retrievers


@dataclass
class QueryIntent:
    raw_query:           str
    intent_type:         IntentType
    confidence:          float                     # 0.0–1.0
    secondary_intent:    Optional[IntentType]      # used when confidence < 0.70

    entities_mentioned:  list[str] = field(default_factory=list)
    temporal_markers:    list[str] = field(default_factory=list)
    comparison_targets:  list[str] = field(default_factory=list)

    analysis_method:     str = "heuristic"   # "heuristic" | "llm" | "llm+heuristic"

    @property
    def is_ambiguous(self) -> bool:
        return self.confidence < 0.70 or self.intent_type == IntentType.AMBIGUOUS

    @property
    def needs_fallback(self) -> bool:
        """True when the router should arm a secondary retriever."""
        return self.confidence < 0.70 and self.secondary_intent is not None

    def to_dict(self) -> dict:
        return {
            "raw_query":          self.raw_query,
            "intent_type":        self.intent_type.value,
            "confidence":         self.confidence,
            "secondary_intent":   self.secondary_intent.value if self.secondary_intent else None,
            "entities_mentioned": self.entities_mentioned,
            "temporal_markers":   self.temporal_markers,
            "comparison_targets": self.comparison_targets,
            "analysis_method":    self.analysis_method,
            "is_ambiguous":       self.is_ambiguous,
            "needs_fallback":     self.needs_fallback,
        }


# ---------------------------------------------------------------------------
# Marker vocabulary
# ---------------------------------------------------------------------------

# Temporal markers: strong signal for TEMPORAL or EVENT intent
_TEMPORAL_STRONG = [
    "which came first", "which did i get first", "which was first",
    "how long ago", "how long before", "how long after",
    "what year did", "what month did", "when did i",
    "before or after", "before i got", "after i got",
    "how many days", "how many weeks", "how many months",
]
_TEMPORAL_WEAK = [
    "first", "before", "after", "earlier", "later", "originally",
    "when", "how long", "timeline", "sequence", "order",
    "oldest", "newest", "most recent", "latest",
]

# Comparison markers: signal for COMPARISON intent
_COMPARISON_STRONG = [
    " or ", " vs ", " versus ", "compared to", "compare",
    "which is better", "which do i prefer", "which should i",
    "difference between", "pros and cons",
]
_COMPARISON_WEAK = [
    "better", "worse", "prefer", "choose", "pick", "between",
]

# Event markers: signal for EVENT intent
_EVENT_MARKERS = [
    "what happened", "what went wrong", "what issue",
    "tell me about the time", "when i", "after i",
    "the incident", "the problem with", "what did i do",
    "what was the result", "how did it go",
]

# Relationship markers
_RELATIONSHIP_MARKERS = [
    "who introduced", "who told me about", "who recommended",
    "who works with", "how do i know", "who is",
    "connection between", "relationship between",
    "who gave me", "who sent me",
]

# Factual fallback markers (explicit facts)
_FACTUAL_MARKERS = [
    "what laptop", "what phone", "what device", "what computer",
    "where do i work", "where do i live", "what is my",
    "what are my", "do i have", "which version", "what model",
]


def _compile(terms: list[str]) -> list[re.Pattern]:
    return [re.compile(re.escape(t), re.IGNORECASE) for t in terms]


_RE_TEMPORAL_STRONG   = _compile(_TEMPORAL_STRONG)
_RE_TEMPORAL_WEAK     = _compile(_TEMPORAL_WEAK)
_RE_COMPARISON_STRONG = _compile(_COMPARISON_STRONG)
_RE_COMPARISON_WEAK   = _compile(_COMPARISON_WEAK)
_RE_EVENT             = _compile(_EVENT_MARKERS)
_RE_RELATIONSHIP      = _compile(_RELATIONSHIP_MARKERS)
_RE_FACTUAL           = _compile(_FACTUAL_MARKERS)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """You are a query intent classifier for a personal memory retrieval system.

Classify the query into exactly one intent type, then extract entities and temporal markers.

INTENT TYPES:
- FACTUAL:      asks for a stored fact ("what laptop do I own", "where do I work")
- TEMPORAL:     asks about time order or when something happened ("which came first", "when did I buy")
- EVENT:        asks about a past occurrence ("what happened when", "what issue did I have after")
- COMPARISON:   compares two or more things ("Samsung or Dell", "which laptop do I prefer")
- RELATIONSHIP: asks about connections between people or things ("who introduced me to X")

RULES:
- entities_mentioned: list of specific named things in the query (device names, people, places, projects)
- temporal_markers: list of time-related words/phrases found in the query
- comparison_targets: list of things being compared (only for COMPARISON intent)
- confidence: 0.0-1.0 (use < 0.70 for genuinely ambiguous queries)
- secondary_intent: if confidence < 0.70, provide the next most likely intent

QUERY: "{query}"

Respond with ONLY valid JSON (no markdown):
{{
  "intent_type": "FACTUAL",
  "confidence": 0.90,
  "secondary_intent": null,
  "entities_mentioned": ["Samsung Galaxy S22", "Dell XPS 13"],
  "temporal_markers": [],
  "comparison_targets": []
}}"""


# ---------------------------------------------------------------------------
# Entity extraction from query (lightweight, no LLM)
# ---------------------------------------------------------------------------

# Brand names and common device keywords for quick entity spotting in queries
_BRAND_PATTERN = re.compile(
    r'\b(Samsung\s+\w+(?:\s+\w+)?|Dell\s+\w+(?:\s+\w+)?|Apple\s+\w+(?:\s+\w+)?|'
    r'MacBook(?:\s+(?:Pro|Air|Mini))?|iPhone\s+\d+\w*|iPad\w*|'
    r'Pixel\s+\d+\w*|OnePlus\s+\w+|Lenovo\s+\w+(?:\s+\w+)?|'
    r'HP\s+\w+(?:\s+\w+)?|Asus\s+\w+(?:\s+\w+)?)',
    re.IGNORECASE
)

def _extract_entities_from_query(query: str) -> list[str]:
    """Lightweight entity extraction specifically for queries."""
    found = []
    for m in _BRAND_PATTERN.finditer(query):
        name = m.group(0).strip()
        if name and name not in found:
            found.append(name)
    return found

def _extract_comparison_targets(query: str) -> list[str]:
    """
    Extract the two sides of a comparison.
    "Samsung Galaxy S22 or Dell XPS 13" → ["Samsung Galaxy S22", "Dell XPS 13"]
    """
    # Try "X or Y" / "X vs Y" / "X versus Y"
    for sep in [r'\s+or\s+', r'\s+vs\.?\s+', r'\s+versus\s+']:
        match = re.split(sep, query, maxsplit=1, flags=re.IGNORECASE)
        if len(match) == 2:
            a = match[0].strip().split()[-3:]   # last 3 words of left side
            b = match[1].strip().split()[:3]    # first 3 words of right side
            left  = " ".join(a).strip("?,.")
            right = " ".join(b).strip("?,.")
            if left and right:
                return [left, right]

    # If we found entity mentions, return those as targets
    entities = _extract_entities_from_query(query)
    if len(entities) >= 2:
        return entities[:2]

    return []


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class QueryAnalyzer:
    """
    Determines the intent of an incoming retrieval query.

    The result drives which retriever runs, which ranking weights apply,
    and whether a fallback retriever should be armed.

    Usage
    -----
    analyzer = QueryAnalyzer(llm=my_llm)
    intent   = analyzer.analyze("Which did I get first — Samsung or Dell?")
    # intent.intent_type == IntentType.TEMPORAL  (or COMPARISON)
    # intent.entities_mentioned == ["Samsung", "Dell"]
    # intent.temporal_markers == ["first"]
    """

    def __init__(self, llm=None, heuristic_only: bool = False):
        self._llm            = llm
        self._heuristic_only = heuristic_only or (llm is None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, query: str) -> QueryIntent:
        """
        Analyze a query and return a QueryIntent.
        Always returns a result — never raises.
        """
        query = query.strip()
        if not query:
            return QueryIntent(
                raw_query=query,
                intent_type=IntentType.FACTUAL,
                confidence=0.30,
                secondary_intent=None,
                analysis_method="fallback",
            )

        # 1. Heuristic pre-pass
        heuristic = self._heuristic_analyze(query)

        # High-confidence heuristic → skip LLM
        if heuristic.confidence >= 0.80 or self._heuristic_only:
            return heuristic

        # 2. LLM for ambiguous cases
        llm_result = self._llm_analyze(query)
        if llm_result:
            # Blend: prefer LLM type but keep heuristic markers if LLM missed them
            if not llm_result.entities_mentioned and heuristic.entities_mentioned:
                llm_result.entities_mentioned = heuristic.entities_mentioned
            if not llm_result.temporal_markers and heuristic.temporal_markers:
                llm_result.temporal_markers = heuristic.temporal_markers
            if not llm_result.comparison_targets and heuristic.comparison_targets:
                llm_result.comparison_targets = heuristic.comparison_targets
            llm_result.analysis_method = "llm+heuristic"
            return llm_result

        return heuristic

    # ------------------------------------------------------------------
    # Heuristic analysis
    # ------------------------------------------------------------------

    def _heuristic_analyze(self, query: str) -> QueryIntent:
        lower = query.lower()

        # --- Extract markers ---
        temporal_found    = [p.pattern for p in _RE_TEMPORAL_STRONG if p.search(lower)]
        temporal_weak     = [p.pattern for p in _RE_TEMPORAL_WEAK   if p.search(lower)]
        comparison_found  = [p.pattern for p in _RE_COMPARISON_STRONG if p.search(lower)]
        comparison_weak   = [p.pattern for p in _RE_COMPARISON_WEAK   if p.search(lower)]
        event_found       = [p.pattern for p in _RE_EVENT             if p.search(lower)]
        relationship_found= [p.pattern for p in _RE_RELATIONSHIP      if p.search(lower)]
        factual_found     = [p.pattern for p in _RE_FACTUAL           if p.search(lower)]

        all_temporal = temporal_found + temporal_weak
        entities     = _extract_entities_from_query(query)
        comp_targets = _extract_comparison_targets(query) if (comparison_found or comparison_weak) else []

        # --- Score each intent ---
        scores: dict[IntentType, float] = {
            IntentType.FACTUAL:      0.40,   # baseline — always a candidate
            IntentType.TEMPORAL:     0.0,
            IntentType.EVENT:        0.0,
            IntentType.COMPARISON:   0.0,
            IntentType.RELATIONSHIP: 0.0,
        }

        if temporal_found:
            scores[IntentType.TEMPORAL] += 0.55
        if temporal_weak:
            scores[IntentType.TEMPORAL] += 0.30 * min(len(temporal_weak), 2)
        # Ordering words are a strong TEMPORAL signal
        _ordering_re = re.compile(
            r'\b(first|before|after|earlier|later|oldest|newest|most recent)\b', re.I)
        if _ordering_re.search(lower):
            scores[IntentType.TEMPORAL] += 0.30

        # "what happened" + ordering word → this is an EVENT question, not pure TEMPORAL
        _event_re = re.compile(r'\bwhat (happened|went wrong|occurred)\b', re.I)
        if _event_re.search(lower) and _ordering_re.search(lower):
            scores[IntentType.EVENT]    += 0.25
            scores[IntentType.TEMPORAL] -= 0.15

        if comparison_found:
            scores[IntentType.COMPARISON] += 0.55
        if comparison_weak and len(entities) >= 2:
            scores[IntentType.COMPARISON] += 0.25
        if comp_targets:
            scores[IntentType.COMPARISON] += 0.15

        if event_found:
            scores[IntentType.EVENT] += 0.55
        if temporal_weak and not temporal_found:
            # Weak temporal without strong → could be EVENT
            scores[IntentType.EVENT] += 0.10

        if relationship_found:
            scores[IntentType.RELATIONSHIP] += 0.60

        if factual_found:
            scores[IntentType.FACTUAL] += 0.35

        # --- Pick winner ---
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_intents[0]
        second_intent, second_score = sorted_intents[1]

        # Normalise score to [0, 1] — cap at 0.92 for heuristic (never fully certain)
        confidence = min(best_score, 0.92)

        # If winner is only marginally ahead → ambiguous
        if best_score - second_score < 0.10 and best_score < 0.70:
            intent_type      = IntentType.AMBIGUOUS
            secondary_intent = second_intent
            confidence       = max(best_score, 0.45)
        elif best_score < 0.45:
            # Nothing fired strongly — default to FACTUAL with low confidence
            intent_type      = IntentType.FACTUAL
            secondary_intent = second_intent if second_score > 0.30 else None
            confidence       = 0.45
        else:
            intent_type      = best_intent
            secondary_intent = second_intent if second_score > 0.30 else None

        return QueryIntent(
            raw_query=query,
            intent_type=intent_type,
            confidence=round(confidence, 3),
            secondary_intent=secondary_intent,
            entities_mentioned=entities,
            temporal_markers=all_temporal,
            comparison_targets=comp_targets,
            analysis_method="heuristic",
        )

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _llm_analyze(self, query: str) -> Optional[QueryIntent]:
        if not self._llm:
            return None

        prompt = _ANALYSIS_PROMPT.format(query=query[:500])
        try:
            raw = self._llm.complete(prompt, max_tokens=300)
        except Exception as e:
            logger.warning("QueryAnalyzer LLM call failed: %s", e)
            return None

        return self._parse_llm_response(query, raw)

    def _parse_llm_response(self, query: str, raw: str) -> Optional[QueryIntent]:
        if not raw:
            return None

        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",           "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict):
            return None

        raw_type = str(data.get("intent_type", "")).upper()
        try:
            intent_type = IntentType(raw_type)
        except ValueError:
            logger.warning("QueryAnalyzer: unknown intent_type %r", raw_type)
            return None

        raw_secondary = str(data.get("secondary_intent") or "").upper()
        secondary = None
        if raw_secondary:
            try:
                secondary = IntentType(raw_secondary)
            except ValueError:
                pass

        return QueryIntent(
            raw_query=query,
            intent_type=intent_type,
            confidence=float(data.get("confidence", 0.60)),
            secondary_intent=secondary,
            entities_mentioned=list(data.get("entities_mentioned", [])),
            temporal_markers=list(data.get("temporal_markers", [])),
            comparison_targets=list(data.get("comparison_targets", [])),
            analysis_method="llm",
        )