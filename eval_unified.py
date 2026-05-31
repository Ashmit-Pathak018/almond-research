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
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
from urllib import response

import ijson
import psutil
import requests

from almond import Almond, AlmondConfig
from memory_block import MemoryTag, MemoryTier
from memory_controller_v2 import EvictionPolicy

# ============================================================================
# CONFIG
# ============================================================================

LLM_API_URL = "http://localhost:1234/v1/chat/completions"
INFERENCE_TIMEOUT_SEC = 120  # Watchdog timeout to prevent hung generations

logging.basicConfig(level=logging.ERROR)

OUTPUT_DIR = Path("longmem_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# METRICS
# ============================================================================

@dataclass
class RetrievalTrace:
    query: str
    retrieved_count: int
    rejected_count: int
    retrieved_ids: List[str]
    rejection_reasons: List[str]
    pollution_score: float


@dataclass
class QuestionResult:
    index: int
    question_type: str
    question: str
    expected_answer: str
    model_response: str
    passed: bool
    latency_ms: float
    replay_time_ms: float
    judge_time_ms: float
    l2_peak: int
    l3_peak: int
    avg_pollution: float


# ============================================================================
# INFRASTRUCTURE HELPERS
# ============================================================================

def get_ram_usage_mb():
    process = psutil.Process()
    return round(
        process.memory_info().rss / 1024 / 1024,
        2
    )

def save_partial_report(
    summary,
    question_results,
    retrieval_exports,
    filename="partial_report.json"
):
    report = {
        "summary": summary,
        "results": [
            asdict(x)
            for x in question_results
        ],
        "retrieval_traces": retrieval_exports
    }

    out_path = OUTPUT_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ============================================================================
# JUDGE
# ============================================================================

def llm_judge(
    question: str,
    expected_answer: str,
    model_response: str
):

    judge_prompt = (
        "I will give you a question, a correct answer, and a model response.\n"
        "Reply ONLY with yes or no.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {expected_answer}\n"
        f"Model Response: {model_response}\n"
    )

    payload = {
        "model": "llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator."
            },
            {
                "role": "user",
                "content": judge_prompt
            }
        ],
        "temperature": 0.0,
        "max_tokens": 5
    }

    try:

        response = requests.post(
            LLM_API_URL,
            json=payload,
            timeout=60
        ).json()

        decision = (
            response["choices"][0]["message"]["content"]
            .strip()
            .lower()
        )
        # debug: print judge response
        print(f"\n[JUDGE DECISION] {decision}")

        return "yes" in decision

    except Exception as e:

        print(f"[JUDGE ERROR] {e}")
        return False


# ============================================================================
# TAG INFERENCE
# ============================================================================

def infer_tag(content: str):

    text = content.lower()

    if any(
        x in text
        for x in [
            "project",
            "retrieval",
            "memory",
            "embedding",
            "benchmark"
        ]
    ):
        return MemoryTag.PROJECT_FACT

    if any(
        x in text
        for x in [
            "my name",
            "favorite",
            "i prefer",
            "i like"
        ]
    ):
        return MemoryTag.USER_PROFILE

    if any(
        x in text
        for x in [
            "todo",
            "need to",
            "remind"
        ]
    ):
        return MemoryTag.TASK

    return MemoryTag.SMALL_TALK


# ============================================================================
# POLLUTION
# ============================================================================

def compute_pollution_score(accepts: List[Dict]):

    if not accepts:
        return 0.0

    tags = [
        x.get("tag")
        for x in accepts
    ]

    duplicates = len(tags) - len(set(tags))

    return duplicates / max(len(tags), 1)


# ============================================================================
# TRACE
# ============================================================================

def extract_trace(
    almond: Almond,
    query: str
):

    try:

        optimizer = almond.controller.optimizer

        accepts = getattr(
            optimizer,
            "latest_accepts",
            []
        )

        rejects = getattr(
            optimizer,
            "latest_rejections",
            []
        )

        trace = RetrievalTrace(
            query=query,
            retrieved_count=len(accepts),
            rejected_count=len(rejects),
            retrieved_ids=[
                str(x.get("id", "unknown"))
                for x in accepts
            ],
            rejection_reasons=[
                x.get("reason", "unknown")
                for x in rejects
            ],
            pollution_score=round(
                compute_pollution_score(accepts),
                3
            )
        )

        return trace

    except Exception:
        return None


# ============================================================================
# ALMOND FACTORY
# ============================================================================

def create_almond():

    policy = EvictionPolicy(
        l2_eviction=2.0,
        l3_eviction=0.5,
        l4_deletion=0.05,
        l2_max_blocks=20
    )

    config = AlmondConfig(
        session_id="longmem_eval",
        db_path="longmem_almond.db",
        eviction_policy=policy,
        max_tokens=512
    )

    return Almond(config)


# ============================================================================
# ABLATIONS
# ============================================================================

def apply_ablation(
    almond: Almond,
    ablation: str
):

    optimizer = almond.controller.optimizer

    if ablation == "no_intent":
        optimizer.disable_intent = True

    elif ablation == "no_keyword":
        optimizer.disable_keyword = True

    elif ablation == "no_recency":
        optimizer.disable_recency = True

    elif ablation == "no_peff":
        optimizer.disable_peff = True


# ============================================================================
# RESET
# ============================================================================

def reset_runtime():

    Path("longmem_almond.db").unlink(
        missing_ok=True
    )

    chroma_path = Path("longmem_almond_chroma")

    if chroma_path.exists():
        shutil.rmtree(chroma_path)


# ============================================================================
# MAIN
# ============================================================================

def run_longmem_eval(
    dataset_path: str,
    limit: int,
    ablation: str = "none"
):

    print("\n" + "=" * 70)
    print("PROJECT ALMOND — UNIFIED EVALUATION")
    print("=" * 70)
    print(f"[DATASET] {dataset_path}")
    print(f"[ABLATION] {ablation}")

    question_results: List[QuestionResult] = []
    retrieval_exports = []

    overall_start = time.time()

    with open(dataset_path, "r", encoding="utf-8") as f:
        # ijson streams the file block-by-block, saving massive RAM
        dataset_iter = ijson.items(f, "item")

        for idx, instance in enumerate(dataset_iter, 1):
            if idx > limit:
                break

            print(f"\n[{idx}/{limit}]")
            almond = None

            try:
                reset_runtime()

                almond = create_almond()

                apply_ablation(
                    almond,
                    ablation
                )

                question = instance["question"]
                answer = instance["answer"]
                q_type = instance["question_type"]

                sessions = instance["haystack_sessions"]

                l2_peak = 0
                l3_peak = 0

                pollution_history = []

                # ====================================================================
                # REPLAY
                # ====================================================================
                print(f"[REPLAYING {len(sessions)} SESSIONS]")
                replay_t0 = time.time()
                
                for s_idx, session in enumerate(sessions, 1):
                    # Cleaner heartbeat to avoid terminal spam
                    if s_idx % 10 == 0 or s_idx == len(sessions):
                        print(f"  [SESSION {s_idx}/{len(sessions)}]")
                        
                    for turn in session:
                        content = turn["content"]

                        almond.add_memory(
                            content=content,
                            tag=infer_tag(content),
                            importance_score=1.0,
                            keywords=[],
                            tier=MemoryTier.L3_VIRTUAL_SWAP
                        )

                        pool = almond.controller.dump_pool()

                        l2_peak = max(
                            l2_peak,
                            sum(1 for x in pool if x["tier"] == "L2_ACTIVE_RAM")
                        )

                        l3_peak = max(
                            l3_peak,
                            sum(1 for x in pool if x["tier"] == "L3_VIRTUAL_SWAP")
                        )
                replay_time_ms = (time.time() - replay_t0) * 1000
                print(f"[REPLAY TIME] {round(replay_time_ms, 2)} ms")

                # ====================================================================
                # QUESTION (With Watchdog Timeout)
                # ====================================================================
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
                    model_response = f"[ERROR] {e}"

                latency_ms = (time.time() - t0) * 1000
                print(f"[LATENCY] {round(latency_ms, 2)} ms")

                # ====================================================================
                # TRACE
                # ====================================================================

                trace = extract_trace(almond, question)

                if trace:
                    retrieval_exports.append(asdict(trace))
                    pollution_history.append(trace.pollution_score)

                # ====================================================================
                # JUDGE
                # ====================================================================

                judge_t0 = time.time()
                passed = llm_judge(question, answer, model_response)
                judge_time_ms = (time.time() - judge_t0) * 1000

                if passed:
                    print("[PASS]")
                else:
                    print("[FAIL]")
                print("\n===== JUDGE =====")
                print("EXPECTED:")
                print(answer)

                print("\nACTUAL:")
                print(model_response)

                print("\nPASS:")
                print(passed)

                # ====================================================================
                # RESULT
                # ====================================================================

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
                    )
                )

                question_results.append(result)
                print(f"[RAM] {get_ram_usage_mb()} MB")

                # ====================================================================
                # AUTOSAVE
                # ====================================================================
                partial_summary = {
                    "completed_questions": len(question_results),
                    "current_accuracy": round(
                        (sum(x.passed for x in question_results) / len(question_results)) * 100, 2
                    ) if question_results else 0.0
                }

                save_partial_report(partial_summary, question_results, retrieval_exports)

            except Exception as e:
                print(f"[QUESTION FAILURE] Skipping instance {idx} due to error: {e}")
                continue
            finally:
                if almond:
                    almond.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    elapsed = time.time() - overall_start

    accuracy = (
        (sum(x.passed for x in question_results) / len(question_results)) * 100
        if question_results else 0.0
    )

    avg_latency = statistics.mean(x.latency_ms for x in question_results) if question_results else 0.0
    avg_replay_time = statistics.mean(x.replay_time_ms for x in question_results) if question_results else 0.0
    avg_pollution = statistics.mean(x.avg_pollution for x in question_results) if question_results else 0.0
    
    # Enhanced Retrieval Stats
    avg_retrieved = statistics.mean(x["retrieved_count"] for x in retrieval_exports) if retrieval_exports else 0.0
    avg_rejected = statistics.mean(x["rejected_count"] for x in retrieval_exports) if retrieval_exports else 0.0

    summary = {
        "ablation": ablation,
        "questions": len(question_results),
        "accuracy": round(accuracy, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_replay_time_ms": round(avg_replay_time, 2),
        "avg_pollution": round(avg_pollution, 3),
        "avg_retrieved_blocks": round(avg_retrieved, 2),
        "avg_rejected_blocks": round(avg_rejected, 2),
        "elapsed_seconds": round(elapsed, 1)
    }

    report = {
        "summary": summary,
        "results": [asdict(x) for x in question_results],
        "retrieval_traces": retrieval_exports
    }

    timestamp = int(time.time())
    out_path = OUTPUT_DIR / f"longmem_report_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    print(f"\nSaved final report to: {out_path}")


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/longmemeval_oracle.json"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50
    )

    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "no_intent",
            "no_keyword",
            "no_recency",
            "no_peff"
        ]
    )

    args = parser.parse_args()

    run_longmem_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        ablation=args.ablation
    )


if __name__ == "__main__":
    main()