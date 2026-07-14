"""
Project Almond — Unified LongMemEval + Cognitive Runtime
========================================================

Features
--------
- Official LongMemEval evaluation
- Retrieval diagnostics
- Context pollution metrics
- Ablation support
- Replay entropy analysis
- Retrieval telemetry
- Latency tracking (Replay, Generation, Judge)
- Memory occupancy tracking
- Orchestrator compatible
- Streaming Dataset Parsing (ijson)
- Real-time RAM Telemetry (psutil)
- Auto-saving & Graceful Question Recovery
- Inference Watchdog Timeout

Changes from previous version
------------------------------
- FIX 1: reset_runtime() now deletes the correct Chroma directory
          ('almond_chroma_db' not 'longmem_almond_chroma').
          This was the root cause of Chroma/SQLite desync across runs.

- FIX 2: l2_peak now measured AFTER almond.chat() returns, not inside
          the replay loop. Page-in happens inside prepare_context() which
          is called by chat() — measuring before that always returned 0.

- FIX 3: add_memory() now uses tier=L2_ACTIVE_RAM instead of L3.
          Blocks in L3 are cold storage — they require a page-in step to
          reach context. Injecting directly to L2 means memories are
          immediately available without needing eviction to fire first.
          importance_score raised from 1.0 to 6.0 so P_eff stays above
          the l2_eviction threshold (2.0) during the benchmark window.

- FIX 4: apply_ablation() now wires into RankingEngine weight profiles.
          The retired RetrievalOptimizer flags are mapped to their V2
          equivalents: no_intent → force FACTUAL weights, no_keyword →
          zero entity_overlap, no_recency → zero timeline_relevance,
          no_peff → zero salience.

- FIX 5: l2_peak and retrieval trace read from controller.get_retrieval_trace()
          after chat() returns, giving the correct post-page-in snapshot.

Usage
-----
python eval_unified.py
python eval_unified.py --limit 50
python eval_unified.py --ablation no_intent
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import ijson
import psutil
import requests

from almond import Almond, AlmondConfig
from memory_block import MemoryTag, MemoryTier
from memory_controller_v2 import EvictionPolicy


# ============================================================================
# FULL TERMINAL LOGGING
# ============================================================================
# Long runs produce far more output than the terminal scrollback or any
# downstream tool can retain — for a 20-question run, only the last
# 3-4 questions' worth of [RANKING ENGINE INPUT/OUTPUT], [TRACE], and
# [FINAL PROMPT] blocks survive. Tee duplicates every byte written to
# stdout/stderr into a timestamped log file in OUTPUT_DIR so the complete
# run is always recoverable afterward, regardless of terminal limits.

class _Tee:
    """File-like object that writes to multiple streams at once."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()  # flush immediately so the log file stays current
                       # even if the process is killed mid-run
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # Preserve the terminal's isatty() so libraries that check for
        # interactive output (progress bars, color codes) behave normally.
        return self._streams[0].isatty() if hasattr(self._streams[0], "isatty") else False


def _setup_full_log(out_dir: Path) -> Path:
    """
    Duplicate stdout and stderr into a timestamped log file under out_dir.
    Returns the log file path. Call once at the start of a run.

    Also rebinds any existing logging.StreamHandler instances (created by
    logging.basicConfig() at module import time, before this function runs)
    to the new teed stderr. Without this, logger.debug/warning calls — like
    the [ENTITY_RESOLVE] lines in temporal_retriever — would bypass the tee
    and only reach the original terminal stream, not the log file.
    """
    out_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    log_path  = out_dir / f"full_run_log_{timestamp}.txt"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    # Rebind logging handlers created before this point so logger.* calls
    # also reach the log file.
    for logger_obj in [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values()):
        if not isinstance(logger_obj, logging.Logger):
            continue
        for handler in logger_obj.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stderr

    return log_path

# ============================================================================
# CONFIG
# ============================================================================

LLM_API_URL          = "http://localhost:1234/v1/chat/completions"
INFERENCE_TIMEOUT_SEC = 120

# FIX 1: correct Chroma directory name — matches memory_store.py
#         PersistentClient(path="./almond_chroma_db")
CHROMA_DIR = "almond_chroma_db"

logging.basicConfig(level=logging.ERROR)

# DIAGNOSTIC: enable DEBUG specifically for temporal_retriever so the
# [ENTITY_RESOLVE] line prints during this run. All other modules stay
# at ERROR to avoid log noise.
logging.getLogger("memory_pipeline_v2.temporal_retriever").setLevel(logging.DEBUG)
logging.getLogger("temporal_retriever").setLevel(logging.DEBUG)

OUTPUT_DIR = Path("longmem_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class RetrievalTrace:
    query:             str
    retrieved_count:   int
    rejected_count:    int
    retrieved_ids:     List[str]
    rejection_reasons: List[str]
    pollution_score:   float


@dataclass
class QuestionResult:
    index:            int
    question_type:    str
    question:         str
    expected_answer:  str
    model_response:   str
    passed:           bool
    latency_ms:       float
    replay_time_ms:   float
    judge_time_ms:    float
    l2_peak:          int
    l3_peak:          int
    avg_pollution:    float
    judge_gate:       str = ""    # which judge_v2 layer produced the verdict
    judge_reasoning:  str = ""    # human-readable explanation from that layer
    judge_extracted:  str = ""    # Layer 3 extracted claim, if reached


# ============================================================================
# INFRASTRUCTURE HELPERS
# ============================================================================

def get_ram_usage_mb() -> float:
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 2)


def save_partial_report(summary, question_results, retrieval_exports,
                        filename="partial_report.json"):
    report = {
        "summary":          summary,
        "results":          [asdict(x) for x in question_results],
        "retrieval_traces": retrieval_exports,
    }
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ============================================================================
# JUDGE
# ============================================================================
# Final-version judge: five-layer structural design in judge_v2.py.
# Replaces the single-LLM-call judge that produced repeated, distinct
# failure classes across every prior benchmark run (token-presence false
# positives, error-string false positives, abstention false negatives).
# See judge_v2.py module docstring for full design rationale.

from judge_v2 import judge as _judge_v2, JudgeResult


def llm_judge(question: str, expected_answer: str, model_response: str) -> bool:
    """
    Thin wrapper preserving the original call signature used throughout
    this file. Internally delegates to the five-layer judge_v2.judge(),
    which prints its own [JUDGE] diagnostic line showing exactly which
    layer produced the verdict and why.
    """
    return llm_judge_full(question, expected_answer, model_response).passed


def llm_judge_full(question: str, expected_answer: str, model_response: str) -> "JudgeResult":
    """Same as llm_judge() but returns the full JudgeResult with gate/reasoning."""
    return _judge_v2(
        question=question,
        expected_answer=expected_answer,
        model_response=model_response,
        llm_api_url=LLM_API_URL,
        verbose=True,
    )


# ============================================================================
# TAG INFERENCE
# ============================================================================

def infer_tag(content: str) -> MemoryTag:
    text = content.lower()
    if any(x in text for x in ["project","retrieval","memory","embedding","benchmark"]):
        return MemoryTag.PROJECT_FACT
    if any(x in text for x in ["my name","favorite","i prefer","i like"]):
        return MemoryTag.USER_PROFILE
    if any(x in text for x in ["todo","need to","remind"]):
        return MemoryTag.TASK
    return MemoryTag.EPISODIC   # was SMALL_TALK — EPISODIC decays slower, better for bench


# ============================================================================
# POLLUTION
# ============================================================================

def compute_pollution_score(accepts: List[Dict]) -> float:
    if not accepts:
        return 0.0
    tags       = [x.get("tag") for x in accepts]
    duplicates = len(tags) - len(set(tags))
    return duplicates / max(len(tags), 1)


# ============================================================================
# TRACE  (FIX 5: uses get_retrieval_trace() for post-page-in snapshot)
# ============================================================================

def extract_trace(almond: Almond, query: str) -> RetrievalTrace | None:
    try:
        # FIX 5: get_retrieval_trace() returns trace enriched with
        # l2_count_after_pagein — measured after prepare_context() completes.
        if hasattr(almond.controller, "get_retrieval_trace"):
            trace = almond.controller.get_retrieval_trace()
        else:
            trace = getattr(almond.controller, "last_retrieval_trace", None)

        if not trace:
            return None

        return RetrievalTrace(
            query=query,
            retrieved_count=trace.get("ranked_count", 0),
            rejected_count=max(
                0,
                trace.get("total_candidates", 0) - trace.get("ranked_count", 0),
            ),
            retrieved_ids=trace.get("paged_in", []),
            rejection_reasons=[],
            pollution_score=0.0,
        )
    except Exception:
        return None


# ============================================================================
# ALMOND FACTORY
# ============================================================================

def create_almond(chroma_path: str | None = None) -> Almond:
    policy = EvictionPolicy(
        l2_eviction=2.0,
        l3_eviction=0.5,
        l4_deletion=0.05,
        l2_max_blocks=20,
    )
    config = AlmondConfig(
        session_id="longmem_eval",
        db_path="longmem_almond.db",
        chroma_path=chroma_path if chroma_path is not None else fresh_chroma_dir(),
        eviction_policy=policy,
        max_tokens=512,
        benchmark_mode=True,
    )
    return Almond(config)


# ============================================================================
# ABLATIONS  (FIX 4: wired into RankingEngine weight profiles)
# ============================================================================

# Module-level backup of original weight profiles.
# Populated on first import so ablations are always reversible.
_ORIGINAL_WEIGHT_PROFILES: dict = {}

def apply_ablation(almond: Almond, ablation: str):
    """
    Map legacy RetrievalOptimizer ablation flags to RankingEngine equivalents.
    Ablations are REVERSIBLE — original weights are snapshotted on first call
    and restored when ablation="none".

    no_intent  → force all intents through FACTUAL weight profile
    no_keyword → zero entity_overlap (V2 equivalent of keyword precision)
    no_recency → zero timeline_relevance (V2 equivalent of recency decay)
    no_peff    → zero salience (V2 equivalent of P_eff in retrieval)
    """
    import copy
    try:
        from memory_pipeline_v2.ranking_engine import _WEIGHT_PROFILES
    except ImportError:
        print("[ABLATION] Warning: could not import _WEIGHT_PROFILES")
        return

    # Snapshot original weights on first call (before any mutation)
    if not _ORIGINAL_WEIGHT_PROFILES:
        _ORIGINAL_WEIGHT_PROFILES.update(copy.deepcopy(_WEIGHT_PROFILES))

    # Always restore first so consecutive ablations don't stack
    for key, profile in _ORIGINAL_WEIGHT_PROFILES.items():
        _WEIGHT_PROFILES[key] = copy.deepcopy(profile)

    if ablation == "none":
        print("[ABLATION] none: original weights restored")
        return

    if ablation == "no_intent":
        factual = copy.deepcopy(_WEIGHT_PROFILES["FACTUAL"])
        for key in list(_WEIGHT_PROFILES.keys()):
            _WEIGHT_PROFILES[key] = copy.deepcopy(factual)
        print("[ABLATION] no_intent: all intents → FACTUAL weights")

    elif ablation == "no_keyword":
        for profile in _WEIGHT_PROFILES.values():
            removed = profile.pop("entity_overlap", 0.0)
            profile["similarity"] = round(profile.get("similarity", 0.0) + removed, 4)
        print("[ABLATION] no_keyword: entity_overlap zeroed → similarity")

    elif ablation == "no_recency":
        for profile in _WEIGHT_PROFILES.values():
            removed = profile.pop("timeline_relevance", 0.0)
            profile["similarity"] = round(profile.get("similarity", 0.0) + removed, 4)
        print("[ABLATION] no_recency: timeline_relevance zeroed → similarity")

    elif ablation == "no_peff":
        for profile in _WEIGHT_PROFILES.values():
            removed = profile.pop("salience", 0.0)
            profile["similarity"] = round(profile.get("similarity", 0.0) + removed, 4)
        print("[ABLATION] no_peff: salience zeroed → similarity")

    else:
        print(f"[ABLATION] Unknown: {ablation!r} — no change applied")

    # Verify all profiles still sum to 1.0
    for intent, profile in _WEIGHT_PROFILES.items():
        total = sum(profile.values())
        if abs(total - 1.0) > 1e-6:
            print(f"[ABLATION WARNING] {intent} profile sums to {total:.6f}, not 1.0")


# ============================================================================
# RESET  (FIX 1: correct Chroma directory)
# FIX 5: stop deleting/recreating the same Chroma directory between
# iterations. The previous approach raced shutil.rmtree() against Windows
# file-handle release after almond.close() — when rmtree partially succeeded
# (some files deleted, others still locked) it left a directory that existed
# but was missing required Chroma tenant/database metadata, which surfaced
# as "Could not connect to tenant default_tenant" on every subsequent
# question once that happened. Instead, every call now returns a fresh,
# uniquely-named directory, so there is never a delete-then-recreate race on
# the same path. Old directories are best-effort cleaned up but failures to
# remove them are non-fatal.
# ============================================================================

_chroma_dir_counter = 0

def fresh_chroma_dir() -> str:
    """Return a new, never-before-used Chroma directory path for this run."""
    global _chroma_dir_counter
    _chroma_dir_counter += 1
    return f"{CHROMA_DIR}_{_chroma_dir_counter}"


def reset_runtime():
    import gc

    Path("longmem_almond.db").unlink(missing_ok=True)
    Path("almond_timeline.db").unlink(missing_ok=True)
    Path("almond_audit.db").unlink(missing_ok=True)

    # Best-effort cleanup of old chroma dirs from prior iterations. This is
    # now purely housekeeping (disk space), not a correctness requirement —
    # each iteration gets its own fresh directory via fresh_chroma_dir(), so
    # a failed cleanup here can never corrupt the directory the next
    # iteration actually uses.
    gc.collect()
    for old_dir in Path(".").glob(f"{CHROMA_DIR}_*"):
        try:
            shutil.rmtree(old_dir)
        except (PermissionError, OSError):
            pass  # leave it for the OS to reclaim later; non-fatal


# ============================================================================
# MAIN
# ============================================================================

def run_longmem_eval(dataset_path: str, limit: int, ablation: str = "none"):

    print("\n" + "=" * 70)
    print("PROJECT ALMOND — UNIFIED EVALUATION")
    print("=" * 70)
    print(f"[DATASET]  {dataset_path}")
    print(f"[ABLATION] {ablation}")
    print(f"[LIMIT]    {limit}")

    question_results: List[QuestionResult] = []
    retrieval_exports: list = []
    overall_start = time.time()

    # Aggregates per-stage timing across ALL questions. Each question gets a
    # brand-new Almond/MemoryController instance (see create_almond() inside
    # the loop), so controller.stage_timings_ms resets every iteration -
    # this dict is what survives across the whole run for the final report.
    global_stage_timings_ms: dict[str, float] = {}
    global_stage_call_counts: dict[str, int]  = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_iter = ijson.items(f, "item")

        for idx, instance in enumerate(dataset_iter, 1):
            if idx > limit:
                break

            print(f"\n{'='*40}")
            print(f"[{idx}/{limit}]")
            almond = None

            try:
                reset_runtime()
                almond = create_almond()
                apply_ablation(almond, ablation)

                question = instance["question"]
                answer   = instance["answer"]
                q_type   = instance["question_type"]
                sessions = instance["haystack_sessions"]

                # FIX 2: l2_peak measured AFTER chat() — initialise here
                l3_peak          = 0
                pollution_history = []

                # ================================================================
                # INGESTION CACHE
                # The same 20 LongMemEval sessions are replayed from scratch on
                # every eval run, paying ~91% of total runtime (1743 LLM calls)
                # for identical deterministic ingestion output (seed is fixed).
                # Cache the post-ingestion SQLite DB + Chroma dir under a hash of
                # the session content. On hit: restore in <1s instead of ~4min.
                # Cache lives at .eval_cache/ and persists across Python sessions.
                # ================================================================
                import hashlib, json as _json

                CACHE_DIR = Path(".eval_cache")
                CACHE_DIR.mkdir(exist_ok=True)

                # Hash is over the full serialized session content + ablation
                # so any change to session data or ablation mode busts the cache.
                _cache_key_src = _json.dumps(sessions, sort_keys=True) + str(ablation)
                _cache_key = hashlib.sha256(_cache_key_src.encode()).hexdigest()[:16]
                _cache_entry = CACHE_DIR / _cache_key
                _cache_db    = _cache_entry / "almond.db"
                _cache_chroma = _cache_entry / "chroma"

                _chroma_path = Path(almond.config.chroma_path) \
                               if almond.config.chroma_path else Path("almond_chroma_db")

                # ================================================================
                # REPLAY (or cache restore)
                # ================================================================
                print(f"[REPLAYING {len(sessions)} SESSIONS]")
                replay_t0 = time.time()

                if _cache_entry.exists():
                    # CACHE HIT: restore SQLite DB instead of re-ingesting.
                    # We intentionally skip copying the Chroma directory —
                    # ChromaDB tenant metadata doesn't survive shutil.copytree
                    # reliably on Windows (causes "default_tenant" errors).
                    # Chroma is only used for FACTUAL/AMBIGUOUS semantic search
                    # fallback; entity registry + timeline (from SQLite) handle
                    # the primary TEMPORAL/COMPARISON retrieval paths. Chroma
                    # will be empty on cache hit runs (vector fallback degrades
                    # to 0 candidates), but entity + timeline retrieval is intact.
                    print(f"  [CACHE HIT] {_cache_key} — restoring SQLite, skipping ingestion LLM calls")
                    almond.close()
                    for _db_name in ["longmem_almond.db", "almond_timeline.db", "almond_audit.db"]:
                        _cached_db = _cache_entry / _db_name
                        if _cached_db.exists():
                            shutil.copy2(str(_cached_db), _db_name)
                    # Fresh empty Chroma dir — avoids Windows tenant corruption.
                    # Vector semantic fallback unavailable on cache hits, but
                    # entity+timeline retrieval (from restored SQLite) is intact.
                    _restored_chroma = str(fresh_chroma_dir())
                    almond = create_almond(chroma_path=_restored_chroma)
                    apply_ablation(almond, ablation)
                    pool    = almond.controller.dump_pool()
                    l3_peak = sum(1 for x in pool if x["tier"] == "L3_VIRTUAL_SWAP")

                else:
                    # CACHE MISS: run full replay then snapshot for future runs
                    for s_idx, session in enumerate(sessions, 1):
                        if s_idx % 10 == 0 or s_idx == len(sessions):
                            print(f"  [SESSION {s_idx}/{len(sessions)}]")

                        for turn in session:
                            content = turn["content"]

                            # Inject to L3 (cold storage) so retrieval is genuinely
                            # tested — memories must be found by semantic search and
                            # paged in before they reach the prompt.
                            # importance_score=6.0 is derived from P_eff math:
                            # EPISODIC λ=0.08, stability=0.1 → p_eff after page-in
                            # stays above l2_eviction threshold (2.0) for 48h,
                            # so hydrated blocks survive the eviction sweep.
                            # importance=1.0 (original) gave p_eff=1.0 < 2.0 → immediate re-eviction.
                            almond.add_memory(
                                content=content,
                                tag=infer_tag(content),
                                importance_score=6.0,
                                keywords=[],
                                tier=MemoryTier.L3_VIRTUAL_SWAP,
                            )

                            # Track L3 peak during replay (L2 peak tracked after chat)
                            pool    = almond.controller.dump_pool()
                            l3_peak = max(
                                l3_peak,
                                sum(1 for x in pool if x["tier"] == "L3_VIRTUAL_SWAP"),
                            )

                    # Snapshot all SQLite DBs for future runs (Chroma excluded
                    # — its tenant structure doesn't survive copytree on Windows;
                    # semantic vector fallback will be unavailable on cache hits
                    # but entity+timeline retrieval works from SQLite alone).
                    try:
                        _cache_entry.mkdir(parents=True, exist_ok=True)
                        for _db_name in ["longmem_almond.db", "almond_timeline.db", "almond_audit.db"]:
                            if Path(_db_name).exists():
                                shutil.copy2(_db_name, str(_cache_entry / _db_name))
                        print(f"  [CACHE SAVE] {_cache_key}")
                    except Exception as _ce:
                        print(f"  [CACHE WARN] Could not save snapshot: {_ce}")

                replay_time_ms = (time.time() - replay_t0) * 1000
                print(f"[REPLAY TIME] {replay_time_ms:.0f} ms")
                print(f"[MEMORIES]    {len(almond.controller._l2)} in L2")

                # ================================================================
                # DIAGNOSTIC: post-replay extraction snapshot
                # Verifies whether fact_extractor / entity_extractor / timeline_index
                # actually ran during replay, before we ask the question.
                # Decision tree:
                #   timeline_count=0 AND entity_count=0 -> extraction not running
                #   timeline_count=0 BUT entity_count>0 -> entities found, no
                #       indexable temporal facts extracted from them
                #   both >0 but timeline_relevance=0 in ranking later ->
                #       entity resolution failure in temporal_retriever
                # ================================================================
                timeline_count = almond.controller._timeline.count()
                entity_count   = len(almond.controller._entity_reg)
                entity_names   = [e.name for e in almond.controller._entity_reg.all_entities()][:20]

                print(f"[DIAGNOSTIC] Timeline events indexed : {timeline_count}")
                print(f"[DIAGNOSTIC] Entities in registry    : {entity_count}")
                print(f"[DIAGNOSTIC] Entity names (first 20) : {entity_names}")

                # ================================================================
                # QUESTION  (with watchdog timeout)
                # ================================================================
                print("[GENERATING RESPONSE]")
                t0 = time.time()

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(almond.chat, question)
                        model_response = future.result(timeout=INFERENCE_TIMEOUT_SEC)
                except concurrent.futures.TimeoutError:
                    model_response = "[ERROR] Generation Timeout Exceeded"
                    print("  [!] Inference hung — Watchdog triggered.")
                except Exception as e:
                    import traceback
                    model_response = f"[ERROR] {e}"
                    print(f"[CHAT EXCEPTION] {type(e).__name__}: {e}")
                    print("[CHAT TRACEBACK]")
                    print(traceback.format_exc())

                latency_ms = (time.time() - t0) * 1000
                print(f"[LATENCY] {latency_ms:.0f} ms")

                # Measure l2_peak NOW — after chat() which calls prepare_context()
                # and page-in. Use controller.l2_count (O(1) dict len) instead
                # of dump_pool() which does a full SQLite scan just for counting.
                l2_peak = almond.controller.l2_count
                # l3_peak: use dump_pool only for L3 count (not in RAM)
                l3_pool = almond.controller.dump_pool(tier=MemoryTier.L3_VIRTUAL_SWAP)
                l3_peak = max(l3_peak, len(l3_pool))

                # ================================================================
                # TRACE  (FIX 5: post-page-in snapshot)
                # ================================================================
                trace = extract_trace(almond, question)
                if trace:
                    retrieval_exports.append(asdict(trace))
                    pollution_history.append(trace.pollution_score)

                # ================================================================
                # DIAGNOSTIC: full retrieval trace breakdown
                # Shows intent classification + per-memory ranking signals.
                # On TEMPORAL questions, if every top5 reasoning string shows
                # "timeline relevance (0.00)" -> timeline_index had nothing
                # for the resolved entities (see [DIAGNOSTIC] entity dump above
                # and [ENTITY_RESOLVE] from temporal_retriever for the cause).
                # ================================================================
                full_trace = almond.controller.get_retrieval_trace()
                print(f"[TRACE] intent={full_trace.get('intent_type')} "
                      f"conf={full_trace.get('intent_confidence')} "
                      f"fallback={full_trace.get('used_fallback')}")
                print(f"[TRACE] candidates={full_trace.get('total_candidates')} "
                      f"ranked={full_trace.get('ranked_count')} "
                      f"paged_in={len(full_trace.get('paged_in', []))}")
                for i, (score, reasoning) in enumerate(zip(
                    full_trace.get('top5_scores', []),
                    full_trace.get('top5_reasoning', []),
                ), 1):
                    print(f"[TRACE] rank{i} score={score:.4f} | {reasoning}")

                # ================================================================
                # JUDGE
                # ================================================================
                judge_t0      = time.time()
                judge_result  = llm_judge_full(question, answer, model_response)
                passed        = judge_result.passed
                judge_time_ms = (time.time() - judge_t0) * 1000

                print(f"\n{'='*30} JUDGE {'='*30}")
                print(f"EXPECTED  : {answer}")
                print(f"ACTUAL    : {model_response[:300]}")
                print(f"GATE      : {judge_result.gate}")
                print(f"REASONING : {judge_result.reasoning}")
                print(f"RESULT    : {'PASS ✓' if passed else 'FAIL ✗'}")

                # ================================================================
                # RESULT
                # ================================================================
                result = QuestionResult(
                    index=idx,
                    question_type=q_type,
                    question=question,
                    expected_answer=answer,
                    model_response=model_response[:500],
                    passed=passed,
                    latency_ms=round(latency_ms, 2),
                    replay_time_ms=round(replay_time_ms, 2),
                    judge_time_ms=round(judge_time_ms, 2),
                    l2_peak=l2_peak,
                    l3_peak=l3_peak,
                    avg_pollution=round(
                        statistics.mean(pollution_history) if pollution_history else 0.0, 3
                    ),
                    judge_gate=judge_result.gate,
                    judge_reasoning=judge_result.reasoning,
                    judge_extracted=judge_result.extracted_claim or "",
                )
                question_results.append(result)
                print(f"[RAM] {get_ram_usage_mb()} MB")

                # ================================================================
                # AUTOSAVE
                # ================================================================
                partial_summary = {
                    "completed_questions": len(question_results),
                    "current_accuracy": round(
                        sum(x.passed for x in question_results)
                        / len(question_results) * 100, 2
                    ) if question_results else 0.0,
                }
                save_partial_report(partial_summary, question_results, retrieval_exports)

            except Exception as e:
                print(f"[QUESTION FAILURE] Skipping instance {idx}: {e}")
                continue
            finally:
                if almond:
                    # Merge this question's per-stage timing into the global
                    # aggregator before the controller (and its timing dict)
                    # gets discarded by close(). This is the only place all
                    # per-question timing data is available, since each
                    # question runs on a fresh Almond/MemoryController.
                    try:
                        for stage, total_ms in almond.controller.stage_timings_ms.items():
                            global_stage_timings_ms[stage] = (
                                global_stage_timings_ms.get(stage, 0.0) + total_ms
                            )
                        for stage, count in almond.controller.stage_call_counts.items():
                            global_stage_call_counts[stage] = (
                                global_stage_call_counts.get(stage, 0) + count
                            )
                    except Exception as e:
                        print(f"  [WARN] Could not merge stage timings: {e}")
                    almond.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed      = time.time() - overall_start
    accuracy     = (
        sum(x.passed for x in question_results) / len(question_results) * 100
        if question_results else 0.0
    )

    summary = {
        "ablation":              ablation,
        "questions":             len(question_results),
        "accuracy":              round(accuracy, 2),
        "avg_latency_ms":        round(statistics.mean(x.latency_ms     for x in question_results), 2) if question_results else 0.0,
        "avg_replay_time_ms":    round(statistics.mean(x.replay_time_ms for x in question_results), 2) if question_results else 0.0,
        "avg_pollution":         round(statistics.mean(x.avg_pollution   for x in question_results), 3) if question_results else 0.0,
        "avg_retrieved_blocks":  round(statistics.mean(x["retrieved_count"] for x in retrieval_exports), 2) if retrieval_exports else 0.0,
        "avg_rejected_blocks":   round(statistics.mean(x["rejected_count"]  for x in retrieval_exports), 2) if retrieval_exports else 0.0,
        "avg_l2_peak":           round(statistics.mean(x.l2_peak for x in question_results), 2) if question_results else 0.0,
        "elapsed_seconds":       round(elapsed, 1),
    }

    report = {
        "summary":          summary,
        "results":          [asdict(x) for x in question_results],
        "retrieval_traces": retrieval_exports,
        "stage_timings":    {
            stage: {
                "total_ms": round(total_ms, 1),
                "calls":    global_stage_call_counts.get(stage, 0),
                "avg_ms":   round(total_ms / global_stage_call_counts[stage], 1)
                            if global_stage_call_counts.get(stage) else 0.0,
                "pct_of_total": round(
                    100 * total_ms / sum(global_stage_timings_ms.values()), 1
                ) if global_stage_timings_ms else 0.0,
            }
            for stage, total_ms in sorted(
                global_stage_timings_ms.items(), key=lambda kv: kv[1], reverse=True
            )
        },
    }

    timestamp = int(time.time())
    out_path  = OUTPUT_DIR / f"longmem_report_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    # ── Timing breakdown — this is what answers "where did the 90 minutes
    # go" without guessing. Sorted by total time descending so the dominant
    # bottleneck is always the first line.
    print("\n" + "=" * 70)
    print("STAGE TIMING BREAKDOWN (sorted by total time, descending)")
    print("=" * 70)
    grand_total_ms = sum(global_stage_timings_ms.values())
    if grand_total_ms > 0:
        print(f"{'STAGE':<32} {'TOTAL':>10} {'CALLS':>8} {'AVG':>10} {'% OF TOTAL':>12}")
        for stage, total_ms in sorted(
            global_stage_timings_ms.items(), key=lambda kv: kv[1], reverse=True
        ):
            calls = global_stage_call_counts.get(stage, 0)
            avg_ms = total_ms / calls if calls else 0.0
            pct = 100 * total_ms / grand_total_ms
            print(f"{stage:<32} {total_ms:>8.0f}ms {calls:>8} {avg_ms:>8.1f}ms {pct:>10.1f}%")
        print(f"\nSum of all instrumented stages: {grand_total_ms/1000:.1f}s "
              f"out of {elapsed:.1f}s wall-clock total "
              f"({100*grand_total_ms/1000/elapsed:.1f}% accounted for)")
    else:
        print("No timing data collected (stage_timings_ms was empty).")

    print(f"\nSaved to: {out_path}")


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main():
    # Tee all stdout/stderr to a full log file before anything else runs,
    # so the complete run is captured even for long benchmarks where the
    # terminal/tool only retains the last few questions' output.
    log_path = _setup_full_log(OUTPUT_DIR)
    print(f"[FULL LOG] Writing complete run output to: {log_path}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default="data/longmemeval_oracle.json")
    parser.add_argument("--limit",    type=int, default=50)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none","no_intent","no_keyword","no_recency","no_peff"])
    args = parser.parse_args()
    run_longmem_eval(dataset_path=args.dataset, limit=args.limit, ablation=args.ablation)

    print(f"\n[FULL LOG] Complete run output saved to: {log_path}")


if __name__ == "__main__":
    main()