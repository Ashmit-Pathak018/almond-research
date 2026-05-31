"""
memory_classifier.py
--------------------
Phase 1 — Memory type classification.

Assigns every incoming memory one of:

    USER_FACT          Stable personal information about the user
    EVENT              Something that happened at a point in time
    PROJECT_FACT       Information about a specific project or work item
    USER_PREFERENCE    Likes, dislikes, habits, routines
    TEMPORARY_CONTEXT  Task-specific context unlikely to matter later
    ASSISTANT_RESPONSE Assistant text that survived hygiene (store cold)
    NOISE              No durable value; should be discarded

Design decisions
----------------
1. Heuristic pre-pass runs before any LLM call.
   Catches the most obvious cases (greetings, pure assistant noise, very
   short messages) without spending a model token.

2. Single combined LLM prompt.
   Classification + a short rationale in one call. No round-trips.

3. Conservative on NOISE.
   Only label NOISE with confidence >= 0.85. Everything else stores
   with a type tag and low confidence rather than being silently dropped.

4. Source-aware defaults.
   Assistant messages default to ASSISTANT_RESPONSE unless they contain
   clear factual assertions (which become USER_FACT candidates).

5. Fallback chain.
   LLM unavailable → heuristic-only classification → always produces a result.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types (re-exported so callers only import from this module)
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


@dataclass
class RawMemory:
    id: str
    source: MemorySource
    text: str
    timestamp: datetime
    session_id: str
    conversation_turn: int


@dataclass
class ClassifiedMemory(RawMemory):
    memory_type:     MemoryType = MemoryType.NOISE
    type_confidence: float      = 0.0
    should_store:    bool       = False
    classification_method: str  = "unset"   # "heuristic" | "llm" | "fallback"
    rationale:       str        = ""
    # allow re-classification later (consolidator may update this)
    type_locked:     bool       = False


# ---------------------------------------------------------------------------
# LLM adapter protocol — swap in any backend
# ---------------------------------------------------------------------------

class LLMAdapter(Protocol):
    """
    Minimal interface the classifier needs from the LLM layer.
    Implement this for LM Studio, Ollama, OpenAI, or any other backend.
    """
    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        ...


class NullLLMAdapter:
    """
    Fallback adapter when no LLM is configured.
    Returns a structured JSON string that triggers heuristic-only classification.
    """
    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        return json.dumps({
            "memory_type": "TEMPORARY_CONTEXT",
            "confidence": 0.4,
            "rationale": "LLM unavailable — heuristic fallback",
        })


class LMStudioAdapter:
    """
    Adapter for a local LM Studio / llama.cpp server.
    
    Usage:
        adapter = LMStudioAdapter(base_url="http://localhost:1234/v1")
        classifier = MemoryClassifier(llm=adapter)
    """
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            import urllib.request
            payload = json.dumps({
                "model":       self.model,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": 0.1,   # Low temp for classification
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("LMStudioAdapter.complete failed: %s", e)
            return ""


# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------

# Patterns that strongly indicate a specific type without needing an LLM.
# Format: (compiled_regex, MemoryType, confidence, source_filter)
# source_filter = None means "any source"
_HEURISTIC_RULES: list[tuple[re.Pattern, MemoryType, float, Optional[MemorySource]]] = []

def _build_heuristic_rules() -> list[tuple[re.Pattern, MemoryType, float, Optional[MemorySource]]]:
    rules_raw = [
        # USER_FACT — ownership, identity, location
        (r"\b(I (own|have|bought|purchased|got|use|am|work|live|went to school))\b",
         MemoryType.USER_FACT, 0.75, MemorySource.USER),
        (r"\bmy (name|age|job|role|title|company|employer|city|country|phone|laptop|car|device)\b",
         MemoryType.USER_FACT, 0.80, MemorySource.USER),

        # USER_PREFERENCE — likes, dislikes, habits
        (r"\b(I (prefer|like|love|hate|dislike|enjoy|avoid|always|never|usually))\b",
         MemoryType.USER_PREFERENCE, 0.75, MemorySource.USER),
        (r"\bmy (favourite|favorite|preferred|go-to)\b",
         MemoryType.USER_PREFERENCE, 0.80, MemorySource.USER),

        # EVENT — past tense action verbs + time markers
        (r"\b(yesterday|last (week|month|year)|ago|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
         MemoryType.EVENT, 0.70, MemorySource.USER),
        (r"\b(attended|visited|went to|participated in|completed|finished|started|launched|released)\b",
         MemoryType.EVENT, 0.72, MemorySource.USER),

        # PROJECT_FACT — project/work keywords
        (r"\b(project|sprint|ticket|issue|pr|pull request|codebase|repo|repository|deploy|pipeline|almond)\b",
         MemoryType.PROJECT_FACT, 0.68, MemorySource.USER),

        # TEMPORARY_CONTEXT — task/reminder language
        (r"\b(remind me|TODO|to-do|don't forget|note:|note to self|remember to)\b",
         MemoryType.TEMPORARY_CONTEXT, 0.78, MemorySource.USER),

        # NOISE — pure assistant uncertainty in assistant source
        (r"\b(I don'?t know|I'?m not sure|I cannot|I apologize|As an AI)\b",
         MemoryType.NOISE, 0.92, MemorySource.ASSISTANT),
    ]
    return [
        (re.compile(pattern, re.IGNORECASE), mtype, conf, source)
        for pattern, mtype, conf, source in rules_raw
    ]

_HEURISTIC_RULES = _build_heuristic_rules()


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """You are a memory classification engine. Classify the following text into exactly one memory type.

MEMORY TYPES (choose exactly one):
- USER_FACT: Stable personal information (owns X, lives in Y, works at Z, age, name, relationships)
- EVENT: Something that happened at a specific point in time (purchased, attended, experienced, completed)
- PROJECT_FACT: Information about a specific project, task, or work item
- USER_PREFERENCE: Likes, dislikes, habits, routines, preferences
- TEMPORARY_CONTEXT: Task-specific information useful now but unlikely to matter later (reminders, current session state)
- ASSISTANT_RESPONSE: Text generated by an AI assistant (may have some value, store cold)
- NOISE: No durable value whatsoever (greetings, acknowledgements, pure filler)

RULES:
- When in doubt between USER_FACT and EVENT: if the action is ongoing/permanent → USER_FACT; if it was a one-time occurrence → EVENT
- A purchase is usually both USER_FACT (now owns something) AND EVENT (the purchase happened). Prefer USER_FACT.
- NOISE confidence must be >= 0.85 to justify discarding. Err toward a real type.
- Source: {source}

TEXT TO CLASSIFY:
"{text}"

Respond with ONLY a JSON object, no markdown, no explanation outside the JSON:
{{
  "memory_type": "ONE_OF_THE_TYPES_ABOVE",
  "confidence": 0.0,
  "rationale": "one sentence explaining why"
}}"""


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class MemoryClassifier:
    """
    Classifies a RawMemory into a typed ClassifiedMemory.

    Heuristic pre-pass → LLM classification → fallback if LLM fails.

    The classifier never discards memories — it only labels them.
    The hygiene module decides what to discard based on the label.
    """

    def __init__(
        self,
        llm:                   Optional[LLMAdapter] = None,
        heuristic_only:        bool = False,
        heuristic_threshold:   float = 0.78,
        min_noise_confidence:  float = 0.85,
    ):
        """
        Parameters
        ----------
        llm
            LLM adapter to use. Defaults to NullLLMAdapter (heuristic-only).
        heuristic_only
            If True, skip LLM entirely. Useful for unit tests or offline mode.
        heuristic_threshold
            Minimum heuristic confidence to skip the LLM call.
        min_noise_confidence
            NOISE classification is only accepted at this confidence or above.
            Below this, downgrade NOISE to TEMPORARY_CONTEXT to be safe.
        """
        self._llm                  = llm or NullLLMAdapter()
        self._heuristic_only       = heuristic_only
        self._heuristic_threshold  = heuristic_threshold
        self._min_noise_confidence = min_noise_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, memory: RawMemory) -> ClassifiedMemory:
        """
        Classify a single memory. Always returns a ClassifiedMemory —
        never raises, never returns None.
        """
        # 1. Source-level shortcut: system messages → TEMPORARY_CONTEXT
        if memory.source == MemorySource.SYSTEM:
            return self._make_classified(
                memory,
                mtype=MemoryType.TEMPORARY_CONTEXT,
                confidence=0.80,
                method="heuristic",
                rationale="System messages are temporary context by definition.",
            )

        # 2. Heuristic pre-pass
        heuristic = self._heuristic_classify(memory)
        if heuristic and heuristic.type_confidence >= self._heuristic_threshold:
            logger.debug("classify: heuristic hit for %s (conf=%.2f)",
                         heuristic.memory_type, heuristic.type_confidence)
            return heuristic

        # 3. Skip LLM if configured
        if self._heuristic_only:
            if heuristic:
                return heuristic
            return self._fallback_classify(memory, reason="heuristic_only_mode")

        # 4. LLM classification
        llm_result = self._llm_classify(memory)
        if llm_result:
            return llm_result

        # 5. Fallback: use heuristic result if available, else safe default
        if heuristic:
            return heuristic
        return self._fallback_classify(memory, reason="llm_failed_no_heuristic")

    def classify_batch(self, memories: list[RawMemory]) -> list[ClassifiedMemory]:
        """Classify a list of memories, preserving order."""
        return [self.classify(m) for m in memories]

    def reclassify(self, memory: ClassifiedMemory, force: bool = False) -> ClassifiedMemory:
        """
        Re-run classification on an already-classified memory.
        Used by the consolidator when more context is available.
        Respects type_locked unless force=True.
        """
        if memory.type_locked and not force:
            logger.debug("reclassify: skipping locked memory %s", memory.id)
            return memory
        raw = RawMemory(
            id=memory.id,
            source=memory.source,
            text=memory.text,
            timestamp=memory.timestamp,
            session_id=memory.session_id,
            conversation_turn=memory.conversation_turn,
        )
        return self.classify(raw)

    # ------------------------------------------------------------------
    # Internal classification paths
    # ------------------------------------------------------------------

    def _heuristic_classify(self, memory: RawMemory) -> Optional[ClassifiedMemory]:
        """
        Fast pattern-matching pass. Returns the first matching rule above
        threshold, or None if no rule fires.
        """
        best_type  = None
        best_conf  = 0.0
        best_label = ""

        for pattern, mtype, conf, source_filter in _HEURISTIC_RULES:
            # Skip if rule is source-specific and doesn't match
            if source_filter and memory.source != source_filter:
                continue
            if pattern.search(memory.text):
                if conf > best_conf:
                    best_type  = mtype
                    best_conf  = conf
                    best_label = pattern.pattern[:60]

        if best_type is None:
            return None

        # Safety: don't heuristically call NOISE unless very confident
        if best_type == MemoryType.NOISE and best_conf < self._min_noise_confidence:
            best_type = MemoryType.TEMPORARY_CONTEXT
            best_conf = 0.45
            best_label = "noise_downgraded_to_temporary"

        return self._make_classified(
            memory,
            mtype=best_type,
            confidence=best_conf,
            method="heuristic",
            rationale=f"Matched heuristic rule: {best_label}",
        )

    def _llm_classify(self, memory: RawMemory) -> Optional[ClassifiedMemory]:
        """
        Call the LLM with a structured classification prompt.
        Returns None on any failure — caller decides the fallback.
        """
        prompt = _CLASSIFICATION_PROMPT.format(
            source=memory.source.value,
            text=memory.text[:800],   # Truncate to avoid context overflow
        )

        try:
            raw_response = self._llm.complete(prompt, max_tokens=200)
        except Exception as e:
            logger.warning("LLM call failed in _llm_classify: %s", e)
            return None

        return self._parse_llm_response(memory, raw_response)

    def _parse_llm_response(
        self,
        memory: RawMemory,
        raw_response: str,
    ) -> Optional[ClassifiedMemory]:
        """
        Parse the LLM's JSON response into a ClassifiedMemory.
        Returns None if the response can't be parsed or is invalid.
        """
        if not raw_response or not raw_response.strip():
            return None

        # Strip markdown code fences if present
        text = raw_response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",           "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if not match:
                logger.warning("_parse_llm_response: no JSON found in response: %r", text[:100])
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        # Validate required fields
        raw_type   = data.get("memory_type", "").strip().upper()
        confidence = float(data.get("confidence", 0.5))
        rationale  = str(data.get("rationale", ""))

        try:
            mtype = MemoryType(raw_type)
        except ValueError:
            logger.warning("_parse_llm_response: unknown memory_type %r", raw_type)
            return None

        # Safety: reject NOISE classification if confidence is too low
        if mtype == MemoryType.NOISE and confidence < self._min_noise_confidence:
            logger.debug(
                "_parse_llm_response: NOISE confidence %.2f below threshold %.2f, "
                "downgrading to TEMPORARY_CONTEXT",
                confidence, self._min_noise_confidence,
            )
            mtype      = MemoryType.TEMPORARY_CONTEXT
            confidence = min(confidence, 0.5)
            rationale  = f"(NOISE downgraded) {rationale}"

        return self._make_classified(
            memory,
            mtype=mtype,
            confidence=confidence,
            method="llm",
            rationale=rationale,
        )

    def _fallback_classify(self, memory: RawMemory, reason: str) -> ClassifiedMemory:
        """
        Last-resort classification when heuristics and LLM both failed.
        Assigns a safe type based only on source, never NOISE.
        """
        logger.debug("_fallback_classify: reason=%s memory_id=%s", reason, memory.id)

        if memory.source == MemorySource.ASSISTANT:
            mtype = MemoryType.ASSISTANT_RESPONSE
        elif memory.source == MemorySource.USER:
            mtype = MemoryType.TEMPORARY_CONTEXT
        else:
            mtype = MemoryType.TEMPORARY_CONTEXT

        return self._make_classified(
            memory,
            mtype=mtype,
            confidence=0.30,
            method="fallback",
            rationale=f"Fallback classification ({reason}). Manual review recommended.",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_classified(
        memory:     RawMemory,
        mtype:      MemoryType,
        confidence: float,
        method:     str,
        rationale:  str,
    ) -> ClassifiedMemory:
        storable_types = {
            MemoryType.USER_FACT,
            MemoryType.EVENT,
            MemoryType.PROJECT_FACT,
            MemoryType.USER_PREFERENCE,
            MemoryType.TEMPORARY_CONTEXT,
            MemoryType.ASSISTANT_RESPONSE,
        }
        return ClassifiedMemory(
            id=memory.id,
            source=memory.source,
            text=memory.text,
            timestamp=memory.timestamp,
            session_id=memory.session_id,
            conversation_turn=memory.conversation_turn,
            memory_type=mtype,
            type_confidence=round(confidence, 4),
            should_store=mtype in storable_types,
            classification_method=method,
            rationale=rationale,
        )