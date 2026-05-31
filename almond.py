"""
Project Almond — LLM Middleware Wrapper
Refactored Benchmark-Safe Runtime
"""

from __future__ import annotations

import logging
import os
from pyexpat.errors import messages
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import anthropic
import openai

from memory_block import MemoryBlock, MemoryTag, MemoryTier
from memory_controller_v2 import MemoryController, EvictionPolicy
from memory_store import MemoryStore

logger = logging.getLogger(__name__)


# ============================================================================
# PROVIDERS
# ============================================================================

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class AlmondConfig:

    # ----------------------------------------------------------------------
    # Provider
    # ----------------------------------------------------------------------

    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.1:8b"

    ollama_base_url: str = "http://localhost:1234"

    max_tokens: int = 1024
    temperature: float = 0.3

    # ----------------------------------------------------------------------
    # Memory
    # ----------------------------------------------------------------------

    db_path: str = "almond.db"
    session_id: Optional[str] = None

    eviction_policy: EvictionPolicy = field(
        default_factory=EvictionPolicy
    )

    # ----------------------------------------------------------------------
    # Benchmark / Evaluation
    # ----------------------------------------------------------------------

    benchmark_mode: bool = False

    # ----------------------------------------------------------------------
    # Default ingestion behavior
    # ----------------------------------------------------------------------

    default_response_tag: MemoryTag = MemoryTag.EPISODIC
    default_response_importance: float = 5.0

    # ----------------------------------------------------------------------
    # System Prompt
    # ----------------------------------------------------------------------

    system_prompt: str = (
        "You are Almond, an intelligent assistant with persistent memory.\n"
        "You may receive retrieved memories from earlier conversations.\n"
        "Use them naturally and carefully.\n"
        "If memory context does not contain enough information to answer "
        "confidently, say you do not know instead of inventing details.\n"
        "Be concise, factual, and direct."
    )


# ============================================================================
# TURN LOGGING
# ============================================================================

@dataclass
class Turn:

    turn_index: int
    session_id: Optional[str]
    timestamp: float

    user_message: str
    assistant_reply: str

    context_block_count: int
    context_token_estimate: int

    paged_in_ids: list[str]
    evicted_ids: list[str]

    latency_ms: float

    provider: str
    model: str


# ============================================================================
# ALMOND
# ============================================================================

class Almond:

    def __init__(
        self,
        config: Optional[AlmondConfig] = None
    ):

        self.config = config or AlmondConfig()

        self.store = MemoryStore(
            db_path=Path(self.config.db_path)
        )

        self.controller = MemoryController(
            policy=self.config.eviction_policy,
            store=self.store,
        )

        self._turn_index = 0

        self._turn_log: list[Turn] = []

        # Smaller rolling history
        self._chat_history: list[dict] = []

        self._boot()

    # =========================================================================
    # BOOT
    # =========================================================================

    def _boot(self):

        existing_l1 = self.store.get_all(
            MemoryTier.L1_HOT_CACHE
        )

        if existing_l1:
            logger.info(
                f"[BOOT] Rehydrated {len(existing_l1)} L1 block(s)"
            )
            return

        system_block = MemoryBlock(
            content=self.config.system_prompt,
            tag=MemoryTag.CORE_RULE,
            importance_score=10.0,
            tier=MemoryTier.L1_HOT_CACHE,
            source="system",
            session_id=self.config.session_id,
            keywords=[]
        )

        self.controller.add(system_block)

        logger.info("[BOOT] System prompt injected")

    # =========================================================================
    # MAIN CHAT
    # =========================================================================

    def chat(
        self,
        user_message: str
    ) -> str:

        start = time.time()

        self._turn_index += 1

        # ------------------------------------------------------------------
        # MEMORY PREP
        # ------------------------------------------------------------------

        context_blocks = self.controller.prepare_context(
            user_message
        )

        # ------------------------------------------------------------------
        # MESSAGE BUILD
        # ------------------------------------------------------------------

        messages = self._build_messages(
            context_blocks=context_blocks,
            user_message=user_message
        )

        # ------------------------------------------------------------------
        # LLM
        # ------------------------------------------------------------------
        
        #Debug: print messages
        print("\n===== FINAL PROMPT =====")
        for msg in messages:
            print(f"\n[{msg['role'].upper()}]")
            print(msg["content"][:2000])

        reply = self._call_llm(messages)

        # ------------------------------------------------------------------
        # CHAT HISTORY
        # ------------------------------------------------------------------

        self._chat_history.append({
            "role": "user",
            "content": user_message
        })

        self._chat_history.append({
            "role": "assistant",
            "content": reply
        })

        # Reduced history size
        MAX_HISTORY = 6

        if len(self._chat_history) > MAX_HISTORY:
            self._chat_history = self._chat_history[-MAX_HISTORY:]

        # ------------------------------------------------------------------
        # INGEST RESPONSE
        # ------------------------------------------------------------------

        # IMPORTANT:
        # Disabled during benchmarks to avoid self-contamination.
        if not self.config.benchmark_mode:

            tag, score = self._classify_response(
                user_message
            )

            self.controller.ingest_response(
                content=reply,
                tag=tag,
                importance_score=score,
                keywords=self._extract_keywords(
                    user_message
                ),
                session_id=self.config.session_id
            )

        # ------------------------------------------------------------------
        # TURN LOGGING
        # ------------------------------------------------------------------

        latency_ms = (
            time.time() - start
        ) * 1000

        self._log_turn(
            user_message=user_message,
            reply=reply,
            context_blocks=context_blocks,
            paged_in_ids=[],
            evicted_ids=[],
            latency_ms=latency_ms
        )

        logger.info(
            f"[TURN {self._turn_index}] "
            f"{latency_ms:.0f}ms"
        )

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
        tier: MemoryTier = MemoryTier.L2_ACTIVE_RAM
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

        return block

    # =========================================================================
    # MESSAGE BUILDING
    # =========================================================================

    def _build_messages(
        self,
        context_blocks: list[MemoryBlock],
        user_message: str
    ) -> list[dict]:

        l1 = [
            b for b in context_blocks
            if b.tier == MemoryTier.L1_HOT_CACHE
        ]

        l2 = [
            b for b in context_blocks
            if b.tier == MemoryTier.L2_ACTIVE_RAM
        ]
        # Debug: print retrieved memory
        print(f"\n[L2 COUNT] {len(l2)}")

        # ------------------------------------------------------------------
        # MEMORY PREAMBLE
        # ------------------------------------------------------------------

        memory_lines = []

        if l2:

            memory_lines.append(
                "=== RELEVANT MEMORY ==="
            )

            for block in l2:

                # CLEANER MEMORY FORMAT
                # No telemetry leakage.
                memory_lines.append(
                    block.to_context_snippet()
                )

            memory_lines.append(
                "=== END MEMORY ==="
            )

        memory_preamble = "\n".join(
            memory_lines
        )

        # ------------------------------------------------------------------
        # SYSTEM CONTENT
        # ------------------------------------------------------------------

        l1_content = "\n\n".join(
            b.to_context_snippet()
            for b in l1
        )

        system_content = l1_content

        if memory_preamble:
            system_content += (
                "\n\n" + memory_preamble
            )

        # ------------------------------------------------------------------
        # FINAL MESSAGE STACK
        # ------------------------------------------------------------------

        history = list(self._chat_history)

        messages = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        messages.extend(history)

        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    # =========================================================================
    # RESPONSE CLASSIFICATION
    # =========================================================================

    def _classify_response(
        self,
        user_message: str
    ) -> tuple[MemoryTag, float]:

        text = user_message.lower()

        # ------------------------------------------------------------------
        # CORE RULE
        # ------------------------------------------------------------------

        if any(
            s in text
            for s in [
                "always remember",
                "remember this",
                "never forget",
                "from now on",
                "rule:",
                "instruction:"
            ]
        ):
            return MemoryTag.CORE_RULE, 9.0

        # ------------------------------------------------------------------
        # USER PROFILE
        # ------------------------------------------------------------------

        if any(
            s in text
            for s in [
                "my name",
                "i am",
                "i'm",
                "i live",
                "i prefer",
                "i like"
            ]
        ):
            return MemoryTag.USER_PROFILE, 8.5

        # ------------------------------------------------------------------
        # PROJECT FACT
        # ------------------------------------------------------------------

        if any(
            s in text
            for s in [
                "project",
                "almond",
                "research",
                "paper",
                "retrieval",
                "memory",
                "benchmark"
            ]
        ):
            return MemoryTag.PROJECT_FACT, 7.5

        # ------------------------------------------------------------------
        # TASK
        # ------------------------------------------------------------------

        if any(
            s in text
            for s in [
                "todo",
                "need to",
                "should",
                "plan",
                "finish"
            ]
        ):
            return MemoryTag.TASK, 6.5

        # ------------------------------------------------------------------
        # SMALL TALK
        # ------------------------------------------------------------------

        if any(
            s in text
            for s in [
                "haha",
                "lol",
                "joke"
            ]
        ):
            return MemoryTag.SMALL_TALK, 3.0

        return MemoryTag.EPISODIC, 5.0

    # =========================================================================
    # PROVIDER ROUTING
    # =========================================================================

    def _call_llm(
        self,
        messages: list[dict]
    ) -> str:

        if self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(messages)

        if self.config.provider == LLMProvider.OLLAMA:
            return self._call_ollama(messages)

        return self._call_openai(messages)

    # =========================================================================
    # OLLAMA
    # =========================================================================

    def _call_ollama(
        self,
        messages: list[dict]
    ) -> str:

        client = openai.OpenAI(
            api_key="ollama",
            base_url=f"{self.config.ollama_base_url}/v1",
        )

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return (
            response.choices[0]
            .message.content
            or ""
        )

    # =========================================================================
    # OPENAI
    # =========================================================================

    def _call_openai(
        self,
        messages: list[dict]
    ) -> str:

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return (
            response.choices[0]
            .message.content
            or ""
        )

    # =========================================================================
    # ANTHROPIC
    # =========================================================================

    def _call_anthropic(
        self,
        messages: list[dict]
    ) -> str:

        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

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

    def _extract_keywords(
        self,
        text: str
    ) -> list[str]:

        stopwords = {
            "the", "a", "an", "is", "it",
            "in", "on", "at", "to", "for",
            "of", "and", "or", "but",
            "i", "you", "we", "they",
            "was", "are", "be",
            "this", "that"
        }

        words = re.findall(
            r"\b[a-zA-Z0-9-]{2,}\b",
            text.lower()
        )

        return list({
            w for w in words
            if w not in stopwords
        })[:15]

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
        latency_ms: float
    ):

        turn = Turn(
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
            model=self.config.model
        )

        self._turn_log.append(turn)

    # =========================================================================
    # EXPORTS
    # =========================================================================

    def export_turn_log(self):

        return [
            t.__dict__
            for t in self._turn_log
        ]

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