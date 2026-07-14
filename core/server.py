"""
Almond Core API Server
======================
FastAPI bridge between Almond Desktop (Tauri/React) and the memory pipeline.

Run with:
    uvicorn server:app --host 127.0.0.1 --port 8472 --reload

This is the ONLY entry point the desktop uses. No pipeline code is imported
directly by the frontend. Keep this file thin — it translates HTTP ↔ core
function calls and nothing else.

REAL API surface (as of Almond 1.0):
  The server wraps the actual Almond class from almond.py, which exposes:
    - almond.chat(user_message: str) -> str
    - almond.add_memory(content, tag, importance_score, keywords, tier)
    - almond.controller  (MemoryController — gives access to store, registry, trace)
    - almond.store       (MemoryStore — get_all, tier_counts, get_all_facts)
    - almond.config      (AlmondConfig — model, session_id, db_path, etc.)
    - almond.close()

  There is no separate RetrievalEngine or standalone TimelineIndex import —
  the controller owns those internally.
"""

from __future__ import annotations

import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("almond.server")

# ── Lazy import of core so the server can start even if deps aren't ready ────
try:
    import sys, os
    # If server.py lives inside memory_pipeline_v2/, make sure the parent
    # is on sys.path so "from memory_pipeline_v2.xxx" imports work.
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from almond import Almond, AlmondConfig, LLMProvider
    from memory_block import MemoryTag, MemoryTier
    CORE_AVAILABLE = True
except Exception as _import_err:
    CORE_AVAILABLE = False
    logger.warning("Core modules not importable — running in stub mode: %s", _import_err)


# ── Singleton Almond instance ─────────────────────────────────────────────────
# One instance per server process. The desktop is single-user by design.
_almond: Optional["Almond"] = None
_start_time = time.time()
_trace_store: dict[str, dict] = {}   # trace_id → retrieval trace dict


def _get_almond() -> Optional["Almond"]:
    return _almond


def _boot_almond(config: "AlmondConfig") -> "Almond":
    global _almond
    if _almond is not None:
        _almond.close()
    _almond = Almond(config)
    logger.info("[server] Almond booted — model=%s db=%s", config.model, config.db_path)
    return _almond


# ── Lifespan: boot with sensible defaults on startup ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if CORE_AVAILABLE:
        _boot_almond(AlmondConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b",
            ollama_base_url="http://localhost:1234",
            db_path="almond_desktop.db",
            session_id="desktop",
        ))
    yield
    if _almond:
        _almond.close()


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Almond Core", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420"],   # Tauri dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None   # future: multi-session support


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class BootRequest(BaseModel):
    """Hot-swap the model or database without restarting the server."""
    model:           str = "llama3.1:8b"
    provider:        str = "ollama"          # "ollama" | "lmstudio" | "openai"
    ollama_base_url: str = "http://localhost:1234"
    db_path:         str = "almond_desktop.db"
    session_id:      str = "desktop"


class AddMemoryRequest(BaseModel):
    content:          str
    tag:              str = "EPISODIC"       # EPISODIC | SEMANTIC | PROCEDURAL
    importance_score: float = 6.0
    keywords:         List[str] = []


# ── /status ───────────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    """
    Desktop polls this every 5 s to show the connection indicator.
    Returns model info, memory counts per tier, and uptime.
    """
    a = _get_almond()
    if not a:
        return {
            "ok":            CORE_AVAILABLE,
            "core_available": CORE_AVAILABLE,
            "stub":          not CORE_AVAILABLE,
            "model":         None,
            "model_ready":   False,
            "memory_counts": {},
            "uptime":        round(time.time() - _start_time),
        }

    tier_counts = a.store.tier_counts() if hasattr(a, "store") else {}
    total = sum(tier_counts.values())

    return {
        "ok":            True,
        "core_available": True,
        "stub":          False,
        "model":         a.config.model,
        "model_ready":   True,
        "session_id":    a.config.session_id,
        "db_path":       a.config.db_path,
        "memory_counts": tier_counts,   # {"L2_ACTIVE_RAM": N, "L3_VIRTUAL_SWAP": M, ...}
        "total_memories": total,
        "uptime":        round(time.time() - _start_time),
        "l2_active":     a.controller.l2_count,
        "l1_hot":        a.controller.l1_count,
    }


# ── /boot ─────────────────────────────────────────────────────────────────────
@app.post("/boot")
def boot(req: BootRequest):
    """
    (Re)initialise Almond with a different model or database.
    Desktop settings panel calls this when the user changes config.
    """
    if not CORE_AVAILABLE:
        raise HTTPException(503, "Core modules not available")

    provider_map = {
        "ollama":   LLMProvider.OLLAMA,
        "lmstudio": LLMProvider.LM_STUDIO,
        "openai":   LLMProvider.OPENAI,
    }
    provider = provider_map.get(req.provider.lower(), LLMProvider.OLLAMA)

    cfg = AlmondConfig(
        provider=provider,
        model=req.model,
        ollama_base_url=req.ollama_base_url,
        db_path=req.db_path,
        session_id=req.session_id,
    )
    _boot_almond(cfg)
    return {"ok": True, "model": req.model, "db_path": req.db_path}


# ── /chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Send a message to Almond. Returns the response text, the top memory
    label used for the "memory hit" indicator, and a trace_id the dev
    panel can use to fetch full retrieval details.
    """
    a = _get_almond()
    if not a:
        return _stub_chat(req.text)

    trace_id = str(uuid.uuid4())

    # Update session_id if the frontend passes one (future multi-session)
    if req.session_id:
        a.config.session_id = req.session_id

    try:
        response_text = a.chat(req.text)
    except Exception as e:
        logger.exception("chat() failed")
        raise HTTPException(500, f"Chat failed: {e}")

    # Capture retrieval trace before it's overwritten by the next call
    trace = a.controller.get_retrieval_trace()
    _trace_store[trace_id] = trace

    # Build a human-readable "memory hit" label for the UI chip
    memory_hit = _make_memory_hit_label(trace)

    return {
        "response":   response_text,
        "memory_hit": memory_hit,
        "trace_id":   trace_id,
    }


def _make_memory_hit_label(trace: dict) -> Optional[str]:
    """Convert raw retrieval trace → short label for the desktop chip."""
    if not trace:
        return None
    candidates = trace.get("candidates_count", 0)
    intent = trace.get("intent", "")
    if candidates == 0:
        return None
    return f"{intent.capitalize()} · {candidates} memories"


# ── /memories ─────────────────────────────────────────────────────────────────
@app.get("/memories")
def list_memories(limit: int = 20, sort: str = "recent", tag: Optional[str] = None):
    """
    List stored memories. sort: "recent" | "importance" | "access"
    tag: filter by MemoryTag name (EPISODIC, SEMANTIC, PROCEDURAL, ...)
    """
    a = _get_almond()
    if not a:
        return _stub_memories()

    try:
        blocks = a.store.get_all()   # all tiers, ordered by created_at ASC
    except Exception as e:
        raise HTTPException(500, str(e))

    # Tag filter
    if tag:
        tag_upper = tag.upper()
        blocks = [b for b in blocks if b.tag.value == tag_upper]

    # Sort
    if sort == "importance":
        blocks = sorted(blocks, key=lambda b: b.importance_score, reverse=True)
    elif sort == "access":
        blocks = sorted(blocks, key=lambda b: b.access_count, reverse=True)
    else:  # "recent" — default, store already returns ASC so reverse
        blocks = list(reversed(blocks))

    blocks = blocks[:limit]

    return [
        {
            "id":               b.id,
            "tag":              b.tag.value,
            "tier":             b.tier.value,
            "content":          b.content,
            "preview":          b.content[:120] + ("…" if len(b.content) > 120 else ""),
            "importance_score": round(b.importance_score, 2),
            "access_count":     b.access_count,
            "p_eff":            round(b.p_eff, 4),
            "session_id":       b.session_id,
            "keywords":         b.keywords or [],
        }
        for b in blocks
    ]


@app.get("/memories/{memory_id}")
def get_memory(memory_id: str):
    a = _get_almond()
    if not a:
        raise HTTPException(503, "Core not available")
    block = a.store.get_block_by_id(memory_id) if hasattr(a.store, "get_block_by_id") else None
    if not block:
        # Fall back to controller method
        text = a.controller.get_by_id(memory_id)
        if not text:
            raise HTTPException(404, "Memory not found")
        return {"id": memory_id, "content": text}
    return {
        "id":               block.id,
        "tag":              block.tag.value,
        "tier":             block.tier.value,
        "content":          block.content,
        "importance_score": block.importance_score,
        "access_count":     block.access_count,
        "p_eff":            round(block.p_eff, 4),
        "delta_t_days":     round(block.delta_t, 2),
        "session_id":       block.session_id,
        "keywords":         block.keywords or [],
    }


@app.post("/memories")
def add_memory(req: AddMemoryRequest):
    """Manually inject a memory — useful for seeding the desktop."""
    a = _get_almond()
    if not a:
        raise HTTPException(503, "Core not available")

    tag_map = {t.value: t for t in MemoryTag}
    tag = tag_map.get(req.tag.upper(), MemoryTag.EPISODIC)

    a.add_memory(
        content=req.content,
        tag=tag,
        importance_score=req.importance_score,
        keywords=req.keywords,
    )
    return {"ok": True}


# ── /memories/search ──────────────────────────────────────────────────────────
@app.post("/memories/search")
def search_memories(req: SearchRequest):
    """
    Semantic search over stored memories via ChromaDB.
    Falls back to empty list if Chroma is unavailable.
    """
    a = _get_almond()
    if not a:
        return {"results": []}

    try:
        results = a.controller.semantic_search(req.query, top_k=req.limit)
        # semantic_search returns list of (memory_id, text, score)
        return {
            "results": [
                {"id": mid, "content": text, "score": round(score, 4)}
                for mid, text, score in results
            ]
        }
    except Exception as e:
        logger.warning("semantic_search failed: %s", e)
        return {"results": []}


# ── /timeline ─────────────────────────────────────────────────────────────────
@app.get("/timeline")
def get_timeline(limit: int = 50):
    """
    Returns timeline events from the controller's timeline index (almond_timeline.db).
    """
    a = _get_almond()
    if not a:
        return {"events": _stub_timeline()}

    try:
        # TimelineIndex is internal to MemoryController
        tl = a.controller._timeline
        # get_recent returns list of TimelineEvent objects
        events = tl.get_recent(limit=limit) if hasattr(tl, "get_recent") else []
        return {
            "events": [
                {
                    "id":        e.id if hasattr(e, "id") else str(uuid.uuid4()),
                    "predicate": e.predicate,
                    "object":    e.object_value,
                    "anchor":    str(e.anchor_date) if e.anchor_date else None,
                    "memory_id": e.memory_id,
                }
                for e in events
            ]
        }
    except Exception as e:
        logger.warning("timeline fetch failed: %s", e)
        return {"events": _stub_timeline()}


# ── /explorer ─────────────────────────────────────────────────────────────────
@app.get("/explorer/entities")
def explorer_entities(limit: int = 50):
    """Returns the entity registry — all known named entities and their mention counts."""
    a = _get_almond()
    if not a:
        return []

    try:
        registry = a.controller._entity_ext._registry
        entities = list(registry._entities.values())[:limit]
        return [
            {
                "id":           e.id,
                "name":         e.canonical_name,
                "type":         e.type,
                "mention_count": e.mention_count if hasattr(e, "mention_count") else 0,
                "first_seen":   str(e.first_seen) if hasattr(e, "first_seen") else None,
                "last_seen":    str(e.last_seen) if hasattr(e, "last_seen") else None,
                "aliases":      list(e.aliases) if hasattr(e, "aliases") else [],
            }
            for e in entities
        ]
    except Exception as e:
        logger.warning("entities fetch failed: %s", e)
        return []


@app.get("/explorer/facts")
def explorer_facts(limit: int = 50):
    """Returns structured facts extracted from memories."""
    a = _get_almond()
    if not a:
        return []

    try:
        facts = a.store.get_all_facts()[:limit]
        return [
            {
                "id":         f.id,
                "memory_id":  f.memory_id,
                "predicate":  f.predicate,
                "object":     f.object_value,
                "confidence": round(f.confidence, 3),
                "anchor":     str(f.anchor_date) if f.anchor_date else None,
            }
            for f in facts
        ]
    except Exception as e:
        logger.warning("facts fetch failed: %s", e)
        return []


@app.get("/explorer/pool")
def explorer_pool():
    """
    Returns the full memory pool with tier/score info.
    Useful for the memory inspector panel in the desktop.
    """
    a = _get_almond()
    if not a:
        return []
    return a.controller.dump_pool()


# ── /trace/{trace_id} ─────────────────────────────────────────────────────────
@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """Returns the full retrieval trace for the dev panel."""
    if trace_id not in _trace_store:
        a = _get_almond()
        if not a:
            return _stub_trace(trace_id)
        # Return current trace as fallback
        return a.controller.get_retrieval_trace()
    return _trace_store[trace_id]


# ── /stage-timings ────────────────────────────────────────────────────────────
@app.get("/stage-timings")
def stage_timings():
    """Returns per-stage runtime breakdown — for the dev/debug panel."""
    a = _get_almond()
    if not a:
        return {}
    return a.controller.get_stage_timing_report()


# ── /welcome ──────────────────────────────────────────────────────────────────
@app.get("/welcome")
def welcome():
    """
    Session initialisation — not a chat turn.
    Returns a presence signal the desktop uses to decide whether to show
    a greeting, a continuation thread, or stay silent.
    Called once when the Chat screen mounts.
    """
    import random

    a = _get_almond()
    if not a:
        return {"greeting": None, "continuation": None, "mood": "quiet", "references": []}

    try:
        recent = a.store.get_all()[-3:]   # last 3 memories across all tiers
    except Exception:
        recent = []

    try:
        tl = a.controller._timeline
        last_events = tl.get_recent(1) if hasattr(tl, "get_recent") else []
    except Exception:
        last_events = []

    if not recent:
        return {"greeting": None, "continuation": None, "mood": "quiet", "references": []}

    modes = ["silent", "remembered", "continued", "present"]
    # Weight silent lower so the feature actually shows up
    mode = random.choices(modes, weights=[1, 3, 3, 2], k=1)[0]

    if mode == "silent":
        return {"greeting": None, "continuation": None, "mood": "quiet", "references": []}

    if mode == "remembered" and recent:
        snippet = recent[-1].content[:80].rstrip()
        if len(recent[-1].content) > 80:
            snippet += "…"
        return {
            "greeting":     None,
            "continuation": f"I was thinking about something you said — \"{snippet}\"",
            "mood":         "reflective",
            "references":   [b.id for b in recent],
        }

    if mode == "continued" and last_events:
        e = last_events[0]
        obj = getattr(e, "object_value", None) or getattr(e, "object", "something")
        return {
            "greeting":     None,
            "continuation": f"You were working on {obj}.",
            "mood":         "calm",
            "references":   [getattr(e, "memory_id", "")],
        }

    # "present" — acknowledge being here without specifics
    return {"greeting": None, "continuation": None, "mood": "present", "references": []}


# ── Stubs ─────────────────────────────────────────────────────────────────────

def _stub_chat(text: str):
    return {
        "response":   f"[stub] Core not loaded. Your message: '{text}'",
        "memory_hit": None,
        "trace_id":   str(uuid.uuid4()),
    }

def _stub_memories():
    return [
        {"id": "stub-1", "tag": "EPISODIC", "tier": "L2_ACTIVE_RAM",
         "preview": "Core not loaded — this is stub data.", "content": "",
         "importance_score": 0, "access_count": 0, "p_eff": 0,
         "session_id": None, "keywords": []},
    ]

def _stub_timeline():
    return [
        {"predicate": "started",  "object": "Almond research",  "anchor": "2025-03-01"},
        {"predicate": "shipped",  "object": "Almond 1.0 paper", "anchor": None},
    ]

def _stub_trace(trace_id: str):
    return {
        "trace_id":        trace_id,
        "intent":          "temporal",
        "candidates_count": 0,
        "stub":            True,
        "log":             [{"level": "warn", "msg": "Core not loaded — stub trace"}],
    }