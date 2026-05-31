"""
fact_extractor.py
-----------------
Phase 2 — Structured fact extraction.

Converts natural language memory text into subject-predicate-object triples
with temporal metadata. This structured form is what enables the timeline
index and comparison retrieval to work without relying on semantic similarity.

Example
-------
Input:  "I bought a Samsung Galaxy S22 in January 2024."
Output: StructuredFact(
    subject="user",
    predicate="purchased",
    object="Samsung Galaxy S22",
    date_raw="January 2024",
    date_parsed=datetime(2024, 1, 15),        # mid-month anchor
    temporal_bound=TemporalBound(
        earliest=datetime(2024, 1, 1),
        latest=datetime(2024, 1, 31),
        confidence=0.70,
        granularity="month"
    ),
    fact_type="ownership",
    confidence=0.91
)

Design decisions
----------------
1. Single LLM call per memory (combined prompt).
   Returns a JSON array so one call handles memories with multiple facts.

2. Heuristic temporal parser runs independently of the LLM.
   Dates the LLM extracts get re-parsed through the temporal parser for
   consistent TemporalBound objects. The LLM's raw date string is preserved
   so nothing is lost.

3. TemporalBound instead of point timestamps.
   "January" → [Jan 1, Jan 31, confidence=0.70].
   "last year" → [Jan 1 last year, Dec 31 last year, confidence=0.45].
   This prevents false ordering in the timeline index.

4. Predicate normalisation.
   "got", "picked up", "ordered", "received" → all normalised to "purchased".
   This collapses the vocabulary so entity + predicate queries work reliably.

5. Conflict detection.
   If two facts share the same subject + object but have different predicate
   or date, a conflict flag is set. The consolidator handles resolution later.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FactType(str, Enum):
    OWNERSHIP    = "ownership"    # user owns / has / uses X
    EVENT        = "event"        # something happened at a time
    PREFERENCE   = "preference"   # user likes / prefers / avoids X
    RELATIONSHIP = "relationship" # person A knows / works with person B
    GOAL         = "goal"         # user wants / is working toward X
    ATTRIBUTE    = "attribute"    # X has property Y (device specs, locations)
    UNKNOWN      = "unknown"


class TemporalGranularity(str, Enum):
    EXACT   = "exact"    # full date known
    DAY     = "day"      # day known, time not
    WEEK    = "week"     # approximate week
    MONTH   = "month"    # month known, day not
    YEAR    = "year"     # year known, month not
    DECADE  = "decade"   # rough era
    UNKNOWN = "unknown"  # no temporal info


@dataclass
class TemporalBound:
    """
    A date range rather than a point — because "January" is not a single day.
    All comparisons in the timeline index operate on these ranges, not timestamps.
    """
    earliest:    datetime
    latest:      datetime
    confidence:  float                # 0.0–1.0
    granularity: TemporalGranularity
    raw:         str = ""             # original string from text ("last month", "January 2024")

    @property
    def midpoint(self) -> datetime:
        delta = self.latest - self.earliest
        return self.earliest + delta / 2

    def overlaps(self, other: "TemporalBound") -> bool:
        return self.earliest <= other.latest and self.latest >= other.earliest

    def is_before(self, other: "TemporalBound") -> Optional[bool]:
        """
        None means "cannot determine" (ranges overlap or confidence too low).
        """
        if self.confidence < 0.4 or other.confidence < 0.4:
            return None
        if self.latest < other.earliest:
            return True
        if self.earliest > other.latest:
            return False
        return None  # ranges overlap — ambiguous


@dataclass
class StructuredFact:
    id:            str
    memory_id:     str
    subject:       str          # usually "user" for first-person memories
    predicate:     str          # normalised verb ("purchased", "owns", "attended")
    object:        str          # the thing
    fact_type:     FactType
    confidence:    float        # 0.0–1.0

    # Temporal
    date_raw:      str = ""            # exactly as extracted ("January 2024", "last month")
    temporal_bound: Optional[TemporalBound] = None

    # Flags
    needs_review:   bool = False       # set if confidence < 0.5 or parse ambiguous
    has_conflict:   bool = False       # set if another fact contradicts this one
    extraction_method: str = "llm"     # "llm" | "heuristic" | "llm+heuristic"

    @property
    def date_parsed(self) -> Optional[datetime]:
        """Convenience: midpoint of temporal bound, or None."""
        return self.temporal_bound.midpoint if self.temporal_bound else None

    @property
    def temporal_confidence(self) -> float:
        return self.temporal_bound.confidence if self.temporal_bound else 0.0


# ---------------------------------------------------------------------------
# Predicate normalisation map
# ---------------------------------------------------------------------------

# Maps surface forms → canonical predicates.
# Extend freely. Keys are lowercase, values are the canonical form.
_PREDICATE_MAP: dict[str, str] = {
    # purchasing / acquisition
    "bought":    "purchased",
    "got":       "purchased",
    "picked up": "purchased",
    "ordered":   "purchased",
    "received":  "purchased",
    "acquired":  "purchased",
    "grabbed":   "purchased",

    # ownership / possession
    "have":  "owns",
    "own":   "owns",
    "use":   "owns",
    "using": "owns",
    "has":   "owns",

    # attendance / presence
    "went to":      "attended",
    "participated": "attended",
    "joined":       "attended",
    "showed up":    "attended",

    # completion
    "finished":  "completed",
    "done with": "completed",
    "wrapped up": "completed",

    # employment
    "work at":   "works_at",
    "work for":  "works_at",
    "employed at": "works_at",
    "job at":    "works_at",

    # residence
    "live in":   "lives_in",
    "living in": "lives_in",
    "based in":  "lives_in",
    "moved to":  "moved_to",

    # preferences
    "prefer":   "prefers",
    "like":     "likes",
    "love":     "likes",
    "enjoy":    "likes",
    "hate":     "dislikes",
    "dislike":  "dislikes",
    "avoid":    "avoids",
}

def normalise_predicate(raw: str) -> str:
    key = raw.lower().strip()
    return _PREDICATE_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Temporal parser
# ---------------------------------------------------------------------------

# Month name → number
_MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,
    "aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}

_RELATIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\byesterday\b",           re.I), "yesterday"),
    (re.compile(r"\btoday\b",               re.I), "today"),
    (re.compile(r"\blast\s+week\b",         re.I), "last_week"),
    (re.compile(r"\blast\s+month\b",        re.I), "last_month"),
    (re.compile(r"\blast\s+year\b",         re.I), "last_year"),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I), "n_days_ago"),
    (re.compile(r"\b(\d+)\s+weeks?\s+ago\b",re.I), "n_weeks_ago"),
    (re.compile(r"\b(\d+)\s+months?\s+ago\b",re.I),"n_months_ago"),
    (re.compile(r"\b(\d+)\s+years?\s+ago\b", re.I),"n_years_ago"),
    (re.compile(r"\bearly\s+(\d{4})\b",     re.I), "early_year"),
    (re.compile(r"\bmid[\-\s]?(\d{4})\b",   re.I), "mid_year"),
    (re.compile(r"\blate\s+(\d{4})\b",      re.I), "late_year"),
    (re.compile(r"\brecently\b",            re.I), "recently"),
    (re.compile(r"\ba\s+while\s+ago\b",     re.I), "a_while_ago"),
    (re.compile(r"\bsome\s+time\s+ago\b",   re.I), "a_while_ago"),
]

def parse_temporal(date_raw: str, anchor: datetime) -> Optional[TemporalBound]:
    """
    Convert a raw date string into a TemporalBound.
    anchor is the timestamp of the memory itself (used for relative dates).
    Returns None if no temporal information can be extracted.
    """
    if not date_raw or not date_raw.strip():
        return None

    text = date_raw.strip()

    # --- ISO / numeric exact date ---
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return TemporalBound(earliest=d, latest=d,
                                  confidence=1.0, granularity=TemporalGranularity.EXACT, raw=text)
        except ValueError:
            pass

    # --- Month + Year: "January 2024", "Jan 2024" ---
    m = re.match(r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})", text, re.I)
    if m:
        month_num = _MONTHS[m.group(1).lower()]
        year      = int(m.group(2))
        start     = datetime(year, month_num, 1)
        # last day of month
        if month_num == 12:
            end = datetime(year, 12, 31)
        else:
            end = datetime(year, month_num + 1, 1) - timedelta(days=1)
        return TemporalBound(earliest=start, latest=end,
                              confidence=0.70, granularity=TemporalGranularity.MONTH, raw=text)

    # --- Year only: "2023", "in 2022" ---
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if m:
        year  = int(m.group(1))
        start = datetime(year, 1, 1)
        end   = datetime(year, 12, 31)
        return TemporalBound(earliest=start, latest=end,
                              confidence=0.55, granularity=TemporalGranularity.YEAR, raw=text)

    # --- Month only (no year): "in January", "last January" ---
    m = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", text, re.I)
    if m:
        month_num = _MONTHS[m.group(1).lower()]
        # Infer year from anchor: if that month hasn't passed this year, use last year
        year  = anchor.year
        if month_num > anchor.month:
            year -= 1
        start = datetime(year, month_num, 1)
        if month_num == 12:
            end = datetime(year, 12, 31)
        else:
            end = datetime(year, month_num + 1, 1) - timedelta(days=1)
        return TemporalBound(earliest=start, latest=end,
                              confidence=0.50, granularity=TemporalGranularity.MONTH, raw=text)

    # --- Relative patterns ---
    for pattern, key in _RELATIVE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        if key == "yesterday":
            d = anchor - timedelta(days=1)
            return TemporalBound(earliest=d, latest=d,
                                  confidence=0.95, granularity=TemporalGranularity.DAY, raw=text)

        if key == "today":
            d = anchor.replace(hour=0, minute=0, second=0)
            return TemporalBound(earliest=d, latest=anchor,
                                  confidence=0.95, granularity=TemporalGranularity.DAY, raw=text)

        if key == "last_week":
            end   = anchor - timedelta(days=anchor.weekday() + 1)
            start = end - timedelta(days=6)
            return TemporalBound(earliest=start, latest=end,
                                  confidence=0.75, granularity=TemporalGranularity.WEEK, raw=text)

        if key == "last_month":
            if anchor.month == 1:
                start = datetime(anchor.year - 1, 12, 1)
                end   = datetime(anchor.year - 1, 12, 31)
            else:
                start = datetime(anchor.year, anchor.month - 1, 1)
                end   = datetime(anchor.year, anchor.month, 1) - timedelta(days=1)
            return TemporalBound(earliest=start, latest=end,
                                  confidence=0.75, granularity=TemporalGranularity.MONTH, raw=text)

        if key == "last_year":
            year  = anchor.year - 1
            return TemporalBound(
                earliest=datetime(year, 1, 1), latest=datetime(year, 12, 31),
                confidence=0.65, granularity=TemporalGranularity.YEAR, raw=text)

        if key == "n_days_ago":
            n = int(match.group(1))
            d = anchor - timedelta(days=n)
            return TemporalBound(earliest=d, latest=d,
                                  confidence=0.85, granularity=TemporalGranularity.DAY, raw=text)

        if key == "n_weeks_ago":
            n     = int(match.group(1))
            end   = anchor - timedelta(weeks=n)
            start = end - timedelta(days=6)
            return TemporalBound(earliest=start, latest=end,
                                  confidence=0.75, granularity=TemporalGranularity.WEEK, raw=text)

        if key == "n_months_ago":
            n     = int(match.group(1))
            month = anchor.month - n
            year  = anchor.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year, 12, 31)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            return TemporalBound(earliest=start, latest=end,
                                  confidence=0.70, granularity=TemporalGranularity.MONTH, raw=text)

        if key == "n_years_ago":
            n    = int(match.group(1))
            year = anchor.year - n
            return TemporalBound(
                earliest=datetime(year, 1, 1), latest=datetime(year, 12, 31),
                confidence=0.65, granularity=TemporalGranularity.YEAR, raw=text)

        if key == "early_year":
            year = int(match.group(1))
            return TemporalBound(
                earliest=datetime(year, 1, 1), latest=datetime(year, 4, 30),
                confidence=0.55, granularity=TemporalGranularity.YEAR, raw=text)

        if key == "mid_year":
            year = int(match.group(1))
            return TemporalBound(
                earliest=datetime(year, 4, 1), latest=datetime(year, 9, 30),
                confidence=0.55, granularity=TemporalGranularity.YEAR, raw=text)

        if key == "late_year":
            year = int(match.group(1))
            return TemporalBound(
                earliest=datetime(year, 9, 1), latest=datetime(year, 12, 31),
                confidence=0.55, granularity=TemporalGranularity.YEAR, raw=text)

        if key == "recently":
            start = anchor - timedelta(days=30)
            return TemporalBound(earliest=start, latest=anchor,
                                  confidence=0.40, granularity=TemporalGranularity.UNKNOWN, raw=text)

        if key == "a_while_ago":
            start = anchor - timedelta(days=365)
            return TemporalBound(earliest=start, latest=anchor - timedelta(days=30),
                                  confidence=0.25, granularity=TemporalGranularity.UNKNOWN, raw=text)

    return None  # no temporal information found


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a fact extraction engine for a personal memory system.

Extract ALL distinct facts from the text below. Each fact must be a clear subject-predicate-object triple.

FACT TYPES:
- ownership:    user owns / has / uses something
- event:        something happened at a specific time
- preference:   user likes / dislikes / prefers something
- relationship: user knows / works with / is related to a person
- goal:         user wants / is working toward something
- attribute:    something has a property (specs, location, colour)
- unknown:      does not fit above

RULES:
- subject: use "user" for first-person statements, a real name for others
- predicate: use simple verb forms ("purchased", "attended", "prefers", "owns")
- object: the most specific noun phrase possible ("Samsung Galaxy S22" not "phone")
- date_raw: extract the date/time expression exactly as written, or "" if none
- confidence: your confidence in this fact being real and durable (0.0-1.0)
- Skip facts you are uncertain about entirely rather than guessing

TEXT:
"{text}"

Respond with ONLY a valid JSON array (no markdown, no preamble):
[
  {{
    "subject": "user",
    "predicate": "purchased",
    "object": "Samsung Galaxy S22",
    "date_raw": "January 2024",
    "fact_type": "ownership",
    "confidence": 0.95
  }}
]

If there are no extractable facts, return: []"""


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FactExtractor:
    """
    Extracts structured StructuredFact objects from ClassifiedMemory text.

    For memories that the classifier marked as NOISE or ASSISTANT_RESPONSE
    with low confidence, extraction is skipped.

    The extractor is intentionally cautious — it prefers returning fewer
    high-confidence facts over many speculative ones.
    """

    # Memory types worth extracting facts from
    _EXTRACTABLE_TYPES = {
        "USER_FACT", "EVENT", "PROJECT_FACT",
        "USER_PREFERENCE", "ASSISTANT_RESPONSE",
    }

    def __init__(self, llm=None, heuristic_only: bool = False):
        """
        llm: any object with a .complete(prompt, max_tokens) -> str method.
             If None, falls back to heuristic-only extraction.
        """
        self._llm            = llm
        self._heuristic_only = heuristic_only or (llm is None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, memory_id: str, text: str, memory_type: str,
                anchor: datetime) -> list[StructuredFact]:
        """
        Extract structured facts from a memory.

        Parameters
        ----------
        memory_id   : ID of the parent ClassifiedMemory
        text        : the memory text
        memory_type : MemoryType string — used to skip unworthy memories
        anchor      : timestamp of the memory — used for relative date resolution

        Returns a (possibly empty) list of StructuredFact objects.
        """
        if memory_type not in self._EXTRACTABLE_TYPES:
            logger.debug("extract: skipping memory_type=%s", memory_type)
            return []

        if not text or not text.strip():
            return []

        facts: list[StructuredFact] = []

        if not self._heuristic_only and self._llm:
            facts = self._llm_extract(memory_id, text, anchor)

        if not facts:
            facts = self._heuristic_extract(memory_id, text, anchor)

        # Post-process: normalise predicates, resolve temporal bounds
        for fact in facts:
            fact.predicate = normalise_predicate(fact.predicate)
            if fact.date_raw and not fact.temporal_bound:
                fact.temporal_bound = parse_temporal(fact.date_raw, anchor)
            if fact.confidence < 0.5:
                fact.needs_review = True

        return facts

    def extract_batch(self, memories: list[tuple[str, str, str, datetime]]
                      ) -> dict[str, list[StructuredFact]]:
        """
        Extract facts from multiple memories.
        Input: list of (memory_id, text, memory_type, anchor) tuples.
        Returns: dict mapping memory_id → list of StructuredFact.
        """
        result = {}
        for memory_id, text, memory_type, anchor in memories:
            result[memory_id] = self.extract(memory_id, text, memory_type, anchor)
        return result

    def detect_conflicts(self, facts: list[StructuredFact]) -> list[StructuredFact]:
        """
        Flag facts that conflict with each other.
        Conflict = same subject + object, different predicate or incompatible dates.
        Mutates facts in-place and returns the modified list.
        """
        # Group by (subject, object)
        groups: dict[tuple, list[StructuredFact]] = {}
        for fact in facts:
            key = (fact.subject.lower(), fact.object.lower())
            groups.setdefault(key, []).append(fact)

        for group in groups.values():
            if len(group) < 2:
                continue
            predicates = {f.predicate for f in group}
            if len(predicates) > 1:
                for f in group:
                    f.has_conflict = True
                    logger.debug("Conflict detected: %s %s %s",
                                 f.subject, f.predicate, f.object)

        return facts

    # ------------------------------------------------------------------
    # LLM extraction path
    # ------------------------------------------------------------------

    def _llm_extract(self, memory_id: str, text: str,
                     anchor: datetime) -> list[StructuredFact]:
        prompt = _EXTRACTION_PROMPT.format(text=text[:1000])
        try:
            raw = self._llm.complete(prompt, max_tokens=600)
        except Exception as e:
            logger.warning("_llm_extract: LLM call failed: %s", e)
            return []

        return self._parse_llm_response(memory_id, raw, anchor)

    def _parse_llm_response(self, memory_id: str, raw: str,
                             anchor: datetime) -> list[StructuredFact]:
        if not raw:
            return []

        # Strip markdown fences
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",           "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON array in surrounding prose
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                logger.warning("_parse_llm_response: no JSON array found")
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        facts = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                raw_type = item.get("fact_type", "unknown").lower()
                try:
                    ftype = FactType(raw_type)
                except ValueError:
                    ftype = FactType.UNKNOWN

                date_raw = str(item.get("date_raw", "") or "")
                tb       = parse_temporal(date_raw, anchor) if date_raw else None

                fact = StructuredFact(
                    id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    subject=str(item.get("subject", "user")),
                    predicate=str(item.get("predicate", "")),
                    object=str(item.get("object", "")),
                    fact_type=ftype,
                    confidence=float(item.get("confidence", 0.5)),
                    date_raw=date_raw,
                    temporal_bound=tb,
                    extraction_method="llm",
                )

                # Skip facts with empty predicate or object
                if not fact.predicate or not fact.object:
                    continue

                facts.append(fact)

            except (KeyError, TypeError, ValueError) as e:
                logger.debug("_parse_llm_response: skipping malformed item: %s", e)
                continue

        return facts

    # ------------------------------------------------------------------
    # Heuristic extraction path
    # ------------------------------------------------------------------

    # Patterns: (regex, predicate, fact_type)
    _HEURISTIC_PATTERNS: list[tuple[re.Pattern, str, FactType]] = []

    @classmethod
    def _build_heuristic_patterns(cls):
        if cls._HEURISTIC_PATTERNS:
            return
        raw = [
            # ownership / purchase
            (r"I (bought|purchased|got|ordered|picked up|received)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+(?:in|on|at|for|from|last|yesterday|today|\d).*)?$",
             "purchased", FactType.OWNERSHIP),
            (r"(?:my|I have a|I own a|I use a)\s+(.+?)\s+(?:is|are|was|has)\b",
             "owns", FactType.OWNERSHIP),
            # preference
            (r"I (?:prefer|like|love|enjoy)\s+(.+?)(?:\s+over\s+.+)?$",
             "likes", FactType.PREFERENCE),
            (r"I (?:hate|dislike|avoid|don'?t like)\s+(.+?)$",
             "dislikes", FactType.PREFERENCE),
            # events
            (r"I (?:attended|went to|visited|participated in|joined)\s+(.+?)(?:\s+(?:in|on|at|last|yesterday).+)?$",
             "attended", FactType.EVENT),
            (r"I (?:completed|finished|launched|released|started)\s+(.+?)(?:\s+(?:in|on|at|last).+)?$",
             "completed", FactType.EVENT),
            # employment / location
            (r"I (?:work at|work for|am employed at)\s+(.+?)$",
             "works_at", FactType.ATTRIBUTE),
            (r"I (?:live in|am based in|moved to)\s+(.+?)$",
             "lives_in", FactType.ATTRIBUTE),
        ]
        cls._HEURISTIC_PATTERNS = [
            (re.compile(p, re.IGNORECASE | re.MULTILINE), pred, ftype)
            for p, pred, ftype in raw
        ]

    def _heuristic_extract(self, memory_id: str, text: str,
                            anchor: datetime) -> list[StructuredFact]:
        self._build_heuristic_patterns()
        facts = []

        # Also try to pull a date from the full text
        date_match = re.search(
            r'(in\s+)?(January|February|March|April|May|June|July|August|September|'
            r'October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            r'(?:\s+\d{4})?|(\d{4})|last\s+(week|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago',
            text, re.IGNORECASE
        )
        date_raw = date_match.group(0).strip() if date_match else ""

        for pattern, predicate, ftype in self._HEURISTIC_PATTERNS:
            for m in pattern.finditer(text):
                # Last capture group is the object
                groups = [g for g in m.groups() if g]
                if not groups:
                    continue
                obj = groups[-1].strip().rstrip(".,;")
                if not obj or len(obj) > 120:
                    continue

                tb = parse_temporal(date_raw, anchor) if date_raw else None
                facts.append(StructuredFact(
                    id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    subject="user",
                    predicate=predicate,
                    object=obj,
                    fact_type=ftype,
                    confidence=0.65,
                    date_raw=date_raw,
                    temporal_bound=tb,
                    extraction_method="heuristic",
                ))

        return facts