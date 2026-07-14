"""
Project Almond — LLM Middleware Wrapper
Refactored Benchmark-Safe Runtime
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import anthropic
import openai

from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_controller_v2 import MemoryController, EvictionPolicy
from core.memory_store import MemoryStore

# ── Phase 1: hygiene gate ──────────────────────────────────────────────────
# Only these two imports are needed for Phase 1.
# Classifier, extractors, and timeline come in Phase 2/3.
from core.memory_pipeline_v2.memory_hygiene import (
    MemoryHygiene,
    HygieneVerdict,
    RawMemory as HRaw,
    MemorySource as HSource,
)

logger = logging.getLogger(__name__)

# Module-level singleton — stateless, safe to share across turns
_hygiene = MemoryHygiene()


# ============================================================================
# PROVIDERS
# ============================================================================

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI    = "openai"
    OLLAMA    = "ollama"


# ============================================================================
# LLM ADAPTER
# Thin wrapper so Phase 2/3 modules can call the LLM without importing Almond.
# ============================================================================

class _AlmondLLMAdapter:
    """Routes extraction-pipeline LLM calls through Almond's existing provider."""

    def __init__(self, almond):
        self._almond = almond

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            return self._almond._call_llm([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.warning("[LLM_ADAPTER] complete() failed: %s", e)
            return ""


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class AlmondConfig:

    # Provider
    provider:         LLMProvider = LLMProvider.OLLAMA
    model:            str         = "llama3.1:8b"
    ollama_base_url:  str         = "http://localhost:1234"
    max_tokens:       int         = 1024
    temperature:      float       = 0.3
    # Fixed seed for reproducible generation. Without this, entity/fact
    # extraction calls (which run the same prompt many times per session,
    # once per memory turn) can return slightly different results between
    # otherwise-identical runs - different entity names, different mention
    # boundaries - which cascades into different entity-registry merge
    # decisions, different candidate counts in retrieval, and ultimately
    # different benchmark scores on every run. Most OpenAI-compatible
    # backends (including Ollama) support this; passing it costs nothing
    # when unsupported (the param is just ignored or raises, in which case
    # set seed=None to disable it for that provider).
    seed:             Optional[int] = 42

    # Memory
    db_path:          str               = "almond.db"
    chroma_path:      Optional[str]     = None  # None -> derived from db_path (see __post_init__)
    session_id:       Optional[str]     = None
    eviction_policy:  EvictionPolicy    = field(default_factory=EvictionPolicy)

    # Benchmark / Evaluation
    benchmark_mode:   bool = False

    # Default ingestion behaviour
    default_response_tag:        MemoryTag = MemoryTag.EPISODIC
    default_response_importance: float     = 5.0

    # System Prompt
    system_prompt: str = (
        "You are Almond, an intelligent assistant with persistent memory.\n"
        "You may receive retrieved memories from earlier conversations.\n"
        "Use them naturally and carefully.\n"
        "If memory context does not contain enough information to answer "
        "confidently, say you do not know instead of inventing details.\n"
        "Be concise, factual, and direct.\n"
        "\n"
        "When answering questions about the ORDER or TIMING of events:\n"
        "- Use only the specific dates or time references explicitly stated "
        "in the memories (e.g. 'mid-February', 'February 27th', 'a month ago', "
        "'3 weeks ago').\n"
        "- Treat relative expressions as calendar durations from the "
        "conversation date: 'a month ago' is older than '3 weeks ago'; "
        "'mid-February' (around Feb 14-15) is earlier than 'February 27th'.\n"
        "- For purchased/acquired items, use the date the item was RECEIVED "
        "or PURCHASED, not when it was ordered or pre-ordered, unless the "
        "question specifically asks about ordering.\n"
        "- Use the dates and time references given, even if approximate "
        "(e.g. 'a month ago', '3 weeks ago', 'mid-February') — these are "
        "usable evidence, not reasons to abstain. Only say you do not have "
        "enough information if NEITHER item being compared has ANY date or "
        "time reference anywhere in the retrieved memories. If at least one "
        "memory mentions a date or relative time for each item, attempt the "
        "comparison using your best reasoning rather than declining to answer."
    )


# ============================================================================
# TURN LOGGING
# ============================================================================

@dataclass
class Turn:
    turn_index:              int
    session_id:              Optional[str]
    timestamp:               float
    user_message:            str
    assistant_reply:         str
    context_block_count:     int
    context_token_estimate:  int
    paged_in_ids:            list[str]
    evicted_ids:             list[str]
    latency_ms:              float
    provider:                str
    model:                   str


# ============================================================================
# ALMOND
# ============================================================================

class Almond:

    def __init__(self, config: Optional[AlmondConfig] = None):
        self.config = config or AlmondConfig()

        chroma_path = self.config.chroma_path
        if chroma_path is None:
            # Derive a chroma directory name from db_path so that distinct
            # db_path values (e.g. per-iteration eval databases) automatically
            # get distinct, non-colliding Chroma directories rather than all
            # sharing the single hardcoded "./almond_chroma_db" path.
            chroma_path = str(Path(self.config.db_path).with_suffix("")) + "_chroma"
        self.store = MemoryStore(db_path=Path(self.config.db_path), chroma_path=chroma_path)

        # Wire the LLM adapter — lets Phase 2/3 extraction modules use
        # whichever provider Almond is already configured for (Ollama/OpenAI/Anthropic).
        self.controller = MemoryController(
            policy=self.config.eviction_policy,
            store=self.store,
            llm_adapter=_AlmondLLMAdapter(self),
        )

        self._turn_index:   int         = 0
        self._turn_log:     list[Turn]  = []
        self._chat_history: list[dict]  = []

        self._boot()

    # =========================================================================
    # BOOT
    # =========================================================================

    def _boot(self):
        existing_l1 = self.store.get_all(MemoryTier.L1_HOT_CACHE)

        if existing_l1:
            logger.info("[BOOT] Rehydrated %d L1 block(s)", len(existing_l1))
            return

        system_block = MemoryBlock(
            content=self.config.system_prompt,
            tag=MemoryTag.CORE_RULE,
            importance_score=10.0,
            tier=MemoryTier.L1_HOT_CACHE,
            source="system",
            session_id=self.config.session_id,
            keywords=[],
        )
        self.controller.add(system_block)
        logger.info("[BOOT] System prompt injected")

    # =========================================================================
    # MAIN CHAT
    # =========================================================================

    def chat(self, user_message: str) -> str:
        start = time.time()
        self._turn_index += 1

        # ------------------------------------------------------------------
        # Phase 1: user message hygiene gate
        # Greetings, one-word acks ("ok", "thanks") are not worth storing.
        # The gate never blocks the LLM call — only the storage decision.
        # ------------------------------------------------------------------
        _user_hraw = HRaw(
            id=f"u-{self._turn_index}",
            source=HSource.USER,
            text=user_message,
            timestamp=__import__("datetime").datetime.now(),
            session_id=self.config.session_id or "default",
            conversation_turn=self._turn_index,
        )
        _user_hygiene = _hygiene.evaluate(_user_hraw)
        _store_user_message = _user_hygiene.verdict != HygieneVerdict.DISCARD

        # ------------------------------------------------------------------
        # MEMORY PREP
        # ------------------------------------------------------------------
        _t0 = time.time()
        context_blocks = self.controller.prepare_context(user_message)
        self.controller._record_stage_time("chat.prepare_context_total", (time.time() - _t0) * 1000)

        # Run extraction on the user message itself (outside benchmark gate)
        # so entities and facts from user input reach the timeline index
        # regardless of whether we store the reply.
        if _store_user_message:
            _user_block_for_extraction = MemoryBlock(
                content=user_message,
                tag=MemoryTag.EPISODIC,
                importance_score=5.0,
                tier=MemoryTier.L2_ACTIVE_RAM,
                source="user",
                session_id=self.config.session_id,
            )
            self.controller._run_extraction_pipeline(_user_block_for_extraction)

        # ------------------------------------------------------------------
        # MESSAGE BUILD
        # ------------------------------------------------------------------
        _t0 = time.time()
        prompt_messages = self._build_messages(
            context_blocks=context_blocks,
            user_message=user_message,
        )
        self.controller._record_stage_time("chat.build_messages", (time.time() - _t0) * 1000)

        # ------------------------------------------------------------------
        # LLM CALL
        # ------------------------------------------------------------------
        _t0 = time.time()
        reply = self._call_llm(prompt_messages)
        self.controller._record_stage_time("chat.llm_generate", (time.time() - _t0) * 1000)

        # ------------------------------------------------------------------
        # CHAT HISTORY  (append then trim so history never exceeds cap)
        # ------------------------------------------------------------------
        MAX_HISTORY = 6
        self._chat_history.append({"role": "user",      "content": user_message})
        self._chat_history.append({"role": "assistant", "content": reply})
        if len(self._chat_history) > MAX_HISTORY:
            self._chat_history = self._chat_history[-MAX_HISTORY:]

        # ------------------------------------------------------------------
        # INGEST RESPONSE
        # Disabled during benchmarks to avoid self-contamination.
        # Phase 1 hygiene: clean the reply before storage, discard noise.
        # ------------------------------------------------------------------
        if not self.config.benchmark_mode:
            # Gate the assistant reply through hygiene before any storage
            cleaned_reply = _hygiene.clean_response(reply)

            if cleaned_reply is not None:
                # Classify based on what the USER said (intent source),
                # not the reply text — this is the correct semantic.
                tag, score = self._classify_response(user_message)

                if _store_user_message:
                    # Keywords extracted from the user message — the topic
                    # of the conversation, not the LLM's filler words.
                    kw = self._extract_keywords(user_message)
                else:
                    kw = []

                self.controller.ingest_response(
                    content=cleaned_reply,
                    tag=tag,
                    importance_score=score,
                    keywords=kw,
                    session_id=self.config.session_id,
                )
            else:
                logger.debug(
                    "[TURN %d] Reply discarded by hygiene (noise/ignorance)",
                    self._turn_index,
                )

        # ------------------------------------------------------------------
        # TURN LOGGING
        # ------------------------------------------------------------------
        latency_ms = (time.time() - start) * 1000

        self._log_turn(
            user_message=user_message,
            reply=reply,
            context_blocks=context_blocks,
            paged_in_ids=[],
            evicted_ids=[],
            latency_ms=latency_ms,
        )

        logger.info("[TURN %d] %.0fms", self._turn_index, latency_ms)

        return reply

    # =========================================================================
    # MANUAL MEMORY
    # =========================================================================

    def add_memory(
        self,
        content: str,
        tag: MemoryTag,
        importance_score: float,
        keywords: Optional[list[str]] = None,
        tier: MemoryTier = MemoryTier.L2_ACTIVE_RAM,
    ) -> MemoryBlock:
        block = MemoryBlock(
            content=content,
            tag=tag,
            importance_score=importance_score,
            keywords=keywords or [],
            tier=tier,
            source="user",
            session_id=self.config.session_id,
        )
        self.controller.add(block)
        # Run extraction pipeline even in benchmark mode so the timeline index
        # and entity registry are populated for retrieval. This does NOT store
        # new memories — it only extracts structured facts from existing ones.
        self.controller._run_extraction_pipeline(block)
        return block

    # =========================================================================
    # MESSAGE BUILDING
    # =========================================================================

    def _build_messages(
        self,
        context_blocks: list[MemoryBlock],
        user_message: str,
    ) -> list[dict]:

        l1 = [b for b in context_blocks if b.tier == MemoryTier.L1_HOT_CACHE]
        # Inject everything that isn't L1 into the memory preamble.
        # _assemble_context() already decided relevance — filtering again
        # by tier here would silently drop any block whose .tier attribute
        # doesn't exactly match L2_ACTIVE_RAM (e.g. freshly hydrated blocks).
        memory_blocks = [b for b in context_blocks if b.tier != MemoryTier.L1_HOT_CACHE]

        logger.debug("[TURN %d] memory blocks in context: %d", self._turn_index, len(memory_blocks))

        # Memory preamble — injected whenever there are non-L1 blocks
        memory_lines: list[str] = []
        if memory_blocks:
            memory_lines.append("=== RELEVANT MEMORY ===")
            for block in memory_blocks:
                memory_lines.append(block.to_context_snippet())
            memory_lines.append("=== END MEMORY ===")

        memory_preamble = "\n".join(memory_lines)

        # System content = L1 (core rules + system prompt) + L2 preamble
        l1_content   = "\n\n".join(b.to_context_snippet() for b in l1)
        system_content = l1_content
        if memory_preamble:
            system_content += "\n\n" + memory_preamble

        # Final message stack
        result: list[dict] = [{"role": "system", "content": system_content}]
        result.extend(list(self._chat_history))
        result.append({"role": "user", "content": user_message})

        return result

    # =========================================================================
    # RESPONSE CLASSIFICATION
    # =========================================================================

    def _classify_response(self, user_message: str) -> tuple[MemoryTag, float]:
        """
        Classify the INTENT of the user's message to decide how to tag
        the resulting memory block. This correctly inspects the user's text,
        not the LLM reply.
        """
        text = user_message.lower()

        if any(s in text for s in [
            "always remember", "remember this", "never forget",
            "from now on", "rule:", "instruction:",
        ]):
            return MemoryTag.CORE_RULE, 9.0

        if any(s in text for s in [
            "my name", "i am", "i'm", "i live", "i prefer", "i like",
        ]):
            return MemoryTag.USER_PROFILE, 8.5

        if any(s in text for s in [
            "project", "almond", "research", "paper",
            "retrieval", "memory", "benchmark",
        ]):
            return MemoryTag.PROJECT_FACT, 7.5

        if any(s in text for s in [
            "todo", "need to", "should", "plan", "finish",
        ]):
            return MemoryTag.TASK, 6.5

        if any(s in text for s in ["haha", "lol", "joke"]):
            return MemoryTag.SMALL_TALK, 3.0

        return MemoryTag.EPISODIC, 5.0

    # =========================================================================
    # PROVIDER ROUTING
    # =========================================================================

    def _call_llm(self, messages: list[dict]) -> str:
        if self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(messages)
        if self.config.provider == LLMProvider.OLLAMA:
            return self._call_ollama(messages)
        return self._call_openai(messages)

    # =========================================================================
    # OLLAMA
    # =========================================================================

    def _call_ollama(self, messages: list[dict]) -> str:
        client = openai.OpenAI(
            api_key="ollama",
            base_url=f"{self.config.ollama_base_url}/v1",
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            seed=self.config.seed,
        )
        return response.choices[0].message.content or ""

    # =========================================================================
    # OPENAI
    # =========================================================================

    def _call_openai(self, messages: list[dict]) -> str:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            seed=self.config.seed,
        )
        return response.choices[0].message.content or ""

    # =========================================================================
    # ANTHROPIC
    # =========================================================================

    def _call_anthropic(self, messages: list[dict]) -> str:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # Anthropic API expects system content separate from the message list
        system_message = messages[0]["content"]
        response = client.messages.create(
            model=self.config.model,
            system=system_message,
            messages=messages[1:],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.content[0].text

    # =========================================================================
    # KEYWORDS
    # =========================================================================

    def _extract_keywords(self, text: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "i", "you", "we", "they",
            "was", "are", "be", "this", "that",
        }
        words = re.findall(r"\b[a-zA-Z0-9-]{2,}\b", text.lower())
        return list({w for w in words if w not in stopwords})[:15]

    # =========================================================================
    # TURN LOGGING
    # =========================================================================

    def _log_turn(
        self,
        user_message: str,
        reply: str,
        context_blocks: list[MemoryBlock],
        paged_in_ids: list[str],
        evicted_ids: list[str],
        latency_ms: float,
    ):
        self._turn_log.append(Turn(
            turn_index=self._turn_index,
            session_id=self.config.session_id,
            timestamp=time.time(),
            user_message=user_message,
            assistant_reply=reply,
            context_block_count=len(context_blocks),
            context_token_estimate=sum(
                len(b.to_context_snippet()) // 4
                for b in context_blocks
            ),
            paged_in_ids=paged_in_ids,
            evicted_ids=evicted_ids,
            latency_ms=latency_ms,
            provider=self.config.provider.value,
            model=self.config.model,
        ))

    # =========================================================================
    # EXPORTS
    # =========================================================================

    def export_turn_log(self) -> list[dict]:
        return [t.__dict__ for t in self._turn_log]

    # =========================================================================
    # TEARDOWN
    # =========================================================================

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass
        logger.info("[ALMOND] Closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()