"""
memory_hygiene.py
-----------------
Phase 1 — Pre-storage gate and post-response filter.

Two jobs:
  1. should_store(memory)      — called before anything hits a store.
                                 Hard veto on garbage before it propagates.
  2. clean_response(text)      — called on LLM response text before
                                 the response itself is considered for storage.

This is intentionally conservative: when in doubt, store with a low-confidence
flag rather than silently discard. The only hard discards are patterns we are
certain produce zero retrievable value.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / lightweight types (kept here so Phase 1 has zero external deps)
# ---------------------------------------------------------------------------

class MemorySource(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class MemoryType(str, Enum):
    USER_FACT          = "USER_FACT"
    EVENT              = "EVENT"
    PROJECT_FACT       = "PROJECT_FACT"
    USER_PREFERENCE    = "USER_PREFERENCE"
    TEMPORARY_CONTEXT  = "TEMPORARY_CONTEXT"
    ASSISTANT_RESPONSE = "ASSISTANT_RESPONSE"
    NOISE              = "NOISE"


class HygieneVerdict(str, Enum):
    STORE        = "STORE"         # store as-is
    STORE_COLD   = "STORE_COLD"    # store but flag as low-value
    DISCARD      = "DISCARD"       # definitely do not store
    NEEDS_REVIEW = "NEEDS_REVIEW"  # borderline; log and store with flag


@dataclass
class RawMemory:
    id: str
    source: MemorySource
    text: str
    timestamp: datetime
    session_id: str
    conversation_turn: int
    memory_type: Optional[MemoryType] = None
    type_confidence: float = 0.0


@dataclass
class HygieneResult:
    verdict: HygieneVerdict
    reason: str
    cleaned_text: Optional[str] = None   # text after filler stripping
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# Hard discard: these phrases in assistant output carry zero factual value
# and create retrieval loops when re-stored.
# Hard discard: genuine ignorance/uncertainty/refusal signals.
# Two or more of these in a single response → discard the whole response.
_ASSISTANT_HARD_DISCARD: list[tuple[str, str]] = [
    (r"I don'?t (know|have|recall|remember|recogni[sz]e)",  "assistant_ignorance"),
    (r"I('m| am) not (sure|certain|aware|familiar)",        "assistant_uncertainty"),
    (r"I (cannot|can'?t) (help|assist|provide|answer)",     "assistant_refusal"),
    (r"As an AI( assistant| language model)?[,.]",          "ai_self_reference"),
    (r"I apologize",                                         "assistant_apology"),
    (r"I don'?t have access",                               "assistant_access_limit"),
    (r"my (knowledge|training) (cutoff|data)",              "knowledge_cutoff"),
    (r"I('m| am) unable to",                                "assistant_inability"),
]

# Soft filler: these appear in assistant output but DO NOT indicate the response
# has no value. They are stripped during clean_response, not used for discard voting.
_ASSISTANT_FILLER_DISCARD: list[tuple[str, str]] = [
    (r"(That'?s|This is) (a great|an interesting) question","filler_opener"),
    (r"I'?d be happy to (help|assist)",                     "filler_opener"),
    (r"Certainly[!.]",                                      "filler_opener"),
    (r"Of course[!.]",                                      "filler_opener"),
    (r"Sure[!,]",                                           "filler_opener"),
]

# Soft discard: user messages that carry no durable value
_USER_SOFT_DISCARD: list[tuple[str, str]] = [
    (r"^(ok|okay|sure|thanks|thank you|got it|alright|cool|nice|yep|yes|no)[.!]?$",
                                                            "acknowledgement"),
    (r"^(hi|hello|hey|bye|goodbye|see ya)[.!]?$",          "greeting"),
    (r"^\?+$",                                              "empty_question"),
]

# Filler sentences to strip from otherwise-storable text (don't discard, just clean).
# These match whole sentences so they're safe to remove even when surrounded by
# real content. Order matters: opener sentences are stripped first, then closers.
_STRIP_PATTERNS: list[str] = [
    # Opener whole-sentences. Use (?:^|(?<=\.\s)) so the preceding period
    # is NOT consumed — it stays in the string for the next match or cleanup.
    r"(?:^|(?<=\.\s))Sure[,!]?\s+I'?d be happy to (help|assist)[.!]?\s*",
    r"(?:^|(?<=\.\s))I'?d be happy to (help|assist)[.!]?\s*",
    r"(?:^|(?<=\.\s))Certainly[!.!]\s*",
    r"(?:^|(?<=\.\s))Of course[!.]\s*",
    r"(?:^|(?<=\.\s))Absolutely[!.]\s*",
    r"(?:^|(?<=\.\s))Sure[!,]\s*",
    r"(?:^|(?<=\.\s))Great[!,]\s*",
    # Standalone short openers at the very start of the string
    r"^(Sure[!.] )|(Certainly[!.] )|(Of course[!.] )",
    # Closer sentences
    r"\s+Is there anything else I can help (you with|with today)\??\.?",
    r"\s+Let me know if you (need|have) (anything else|more questions)[.!]?",
    r"\s+Feel free to ask if you need more information[.!]?",
    r"\s+Hope (that|this) helps[.!]?",
    # Cleanup: remove any leading/trailing orphaned periods after stripping
    r"^[.\s]+",
]

# Minimum thresholds
_MIN_CHARS          = 8     # below this, not worth storing
_MIN_WORDS          = 2     # below this, not worth storing
_MAX_REPETITION_PCT = 0.6   # if >60% of words are the same word, it's noise


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MemoryHygiene:
    """
    Stateless filter. Every method is a pure function of its inputs —
    no database calls, no LLM calls, no side effects.

    Usage
    -----
    hygiene = MemoryHygiene()

    # Before storing any memory:
    result = hygiene.evaluate(raw_memory)
    if result.verdict == HygieneVerdict.DISCARD:
        return   # drop it

    # After an LLM response, before re-storing the response text:
    cleaned = hygiene.clean_response(response_text)
    if cleaned is None:
        return   # drop it
    """

    def __init__(
        self,
        extra_assistant_patterns: Optional[list[str]] = None,
        extra_user_patterns:      Optional[list[str]] = None,
        min_chars:                int = _MIN_CHARS,
        min_words:                int = _MIN_WORDS,
    ):
        self._min_chars  = min_chars
        self._min_words  = min_words

        # Compile assistant hard-discard patterns (ignorance/refusal signals)
        base_asst = [(re.compile(p, re.IGNORECASE), label)
                     for p, label in _ASSISTANT_HARD_DISCARD]
        extra_asst = [(re.compile(p, re.IGNORECASE), "custom")
                      for p in (extra_assistant_patterns or [])]
        self._asst_patterns = base_asst + extra_asst

        # Compile assistant filler patterns (stripped but not counted toward discard)
        self._asst_filler_patterns = [
            (re.compile(p, re.IGNORECASE), label)
            for p, label in _ASSISTANT_FILLER_DISCARD
        ]

        # Compile user soft-discard patterns
        base_user = [(re.compile(p, re.IGNORECASE), label)
                     for p, label in _USER_SOFT_DISCARD]
        extra_usr = [(re.compile(p, re.IGNORECASE), "custom")
                     for p in (extra_user_patterns or [])]
        self._user_patterns = base_user + extra_usr

        # Compile strip patterns
        self._strip_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL)
                                 for p in _STRIP_PATTERNS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, memory: RawMemory) -> HygieneResult:
        """
        Main entry point. Returns a HygieneResult with a verdict and reason.

        Call this before passing a memory to the classifier or any store.
        """
        text = memory.text.strip()

        # --- Universal: too short to carry meaning ---
        if len(text) < self._min_chars or self._word_count(text) < self._min_words:
            return HygieneResult(
                verdict=HygieneVerdict.DISCARD,
                reason="too_short",
                cleaned_text=None,
                flags=["length_filter"]
            )

        # --- Universal: high repetition (noise signal) ---
        if self._repetition_ratio(text) > _MAX_REPETITION_PCT:
            return HygieneResult(
                verdict=HygieneVerdict.DISCARD,
                reason="high_repetition",
                flags=["repetition_filter"]
            )

        # --- Route by source ---
        if memory.source == MemorySource.ASSISTANT:
            return self._evaluate_assistant(text)

        if memory.source == MemorySource.USER:
            return self._evaluate_user(memory, text)

        # SYSTEM messages: store cold (may carry useful context but rarely fact-dense)
        return HygieneResult(
            verdict=HygieneVerdict.STORE_COLD,
            reason="system_message",
            cleaned_text=text,
            flags=["system_source"]
        )

    def clean_response(self, response_text: str) -> Optional[str]:
        """
        Strip filler from an LLM response and return the cleaned text,
        or None if the response should not be stored at all.

        Use this on the *full* response text after the LLM call, before
        any storage decision is made.
        """
        text = response_text.strip()

        if not text:
            return None

        # Hard check: if the response contains genuine ignorance/refusal signals,
        # discard the whole thing. Filler openers ("Sure!", "Certainly!") do NOT
        # count here — they're stripped below, not treated as noise votes.
        noise_count = sum(
            1 for pattern, _ in self._asst_patterns   # only hard-discard patterns
            if pattern.search(text)
        )
        if noise_count >= 2:
            logger.debug("clean_response: discarding response with %d noise matches", noise_count)
            return None
        # Single noise hit but text is very short — nothing of value left after removal
        if noise_count == 1 and len(text.split()) < 12:
            return None

        # Strip filler openers and closers
        cleaned = text
        for pattern in self._strip_patterns:
            cleaned = pattern.sub("", cleaned).strip()

        # If stripping left us with almost nothing, discard
        if len(cleaned) < self._min_chars:
            return None

        return cleaned if cleaned != text else text

    def is_storable(self, memory: RawMemory) -> bool:
        """Convenience bool wrapper around evaluate()."""
        return self.evaluate(memory).verdict in (
            HygieneVerdict.STORE,
            HygieneVerdict.STORE_COLD,
            HygieneVerdict.NEEDS_REVIEW,
        )

    def batch_evaluate(self, memories: list[RawMemory]) -> list[tuple[RawMemory, HygieneResult]]:
        """Evaluate a list of memories. Returns (memory, result) pairs."""
        return [(m, self.evaluate(m)) for m in memories]

    def filter_storable(self, memories: list[RawMemory]) -> list[RawMemory]:
        """Return only the memories that pass the hygiene filter."""
        return [m for m in memories if self.is_storable(m)]

    # ------------------------------------------------------------------
    # Internal evaluation paths
    # ------------------------------------------------------------------

    def _evaluate_assistant(self, text: str) -> HygieneResult:
        """
        Assistant messages are guilty until proven innocent.
        Any hard-discard pattern → discard immediately.
        """
        for pattern, label in self._asst_patterns:
            if pattern.search(text):
                logger.debug("Discarding assistant memory: matched pattern '%s'", label)
                return HygieneResult(
                    verdict=HygieneVerdict.DISCARD,
                    reason=f"assistant_noise:{label}",
                    flags=[label]
                )

        # Assistant text that contains no noise patterns may still carry value
        # (e.g. a factual answer the user can reference later).
        # Store cold — it won't be promoted to L2 easily, but it's recoverable.
        cleaned = self._strip_filler(text)
        return HygieneResult(
            verdict=HygieneVerdict.STORE_COLD,
            reason="assistant_clean",
            cleaned_text=cleaned,
            flags=["assistant_source"]
        )

    def _evaluate_user(self, memory: RawMemory, text: str) -> HygieneResult:
        """
        User messages default to storable. Only soft patterns trigger discard.
        If there's a pre-assigned type with good confidence, use it to refine.
        """
        # Soft discard for pure acknowledgements / greetings
        for pattern, label in self._user_patterns:
            if pattern.match(text):
                return HygieneResult(
                    verdict=HygieneVerdict.DISCARD,
                    reason=f"user_noise:{label}",
                    flags=[label]
                )

        # If the classifier already called this NOISE with high confidence, respect it
        if (memory.memory_type == MemoryType.NOISE
                and memory.type_confidence >= 0.85):
            return HygieneResult(
                verdict=HygieneVerdict.DISCARD,
                reason="classifier_noise_high_confidence",
                flags=["classifier_verdict"]
            )

        # TEMPORARY_CONTEXT: store cold — useful now, probably not later
        if memory.memory_type == MemoryType.TEMPORARY_CONTEXT:
            return HygieneResult(
                verdict=HygieneVerdict.STORE_COLD,
                reason="temporary_context",
                cleaned_text=text,
                flags=["temporary"]
            )

        # Everything else: store
        cleaned = self._strip_filler(text)
        return HygieneResult(
            verdict=HygieneVerdict.STORE,
            reason="user_message_clean",
            cleaned_text=cleaned,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strip_filler(self, text: str) -> str:
        cleaned = text
        for pattern in self._strip_patterns:
            cleaned = pattern.sub("", cleaned).strip()
        return cleaned or text

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        most_common_count = max(words.count(w) for w in set(words))
        return most_common_count / len(words)