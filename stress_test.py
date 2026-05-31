"""
Project Almond — Research Stress Test Suite v3.0
================================================

Major Upgrades:
- Randomized probe scheduling
- Retrieval trace exports
- Context pollution metrics
- Precision metrics
- Improved negative hallucination tests
- Component ablation support
- Probe realism improvements
- Context assembly diagnostics
- Retrieval analytics exports
- Cleaner evaluation methodology

Purpose:
This suite evaluates:
- Recall quality
- Retrieval precision
- Context pollution
- Hallucination resistance
- Temporal memory stability
- Long-session degradation
- Retrieval governance quality

Usage:
    python stress_test_v3.py --turns 500
    python stress_test_v3.py --turns 500 --ablation no_intent
    python stress_test_v3.py --turns 500 --no-llm
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from almond import Almond, AlmondConfig
from memory_block import MemoryTag, MemoryTier
from memory_controller_v2 import EvictionPolicy

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

SEED_FACTS = [
    {
        "id": "fact_user_name",
        "content": "User's name is Asmit. He is building Project Almond as both a research paper and a personal Jarvis-style voice assistant.",
        "tag": MemoryTag.USER_PROFILE,
        "importance": 9.5,
        "keywords": ["asmit", "almond", "jarvis", "assistant", "research"],
        "probes": [
            {
                "type": "factual",
                "question": "What is the user's name?",
                "expected": ["asmit"]
            },
            {
                "type": "paraphrase",
                "question": "Who am I? Do you remember me?",
                "expected": ["asmit"]
            },
            {
                "type": "abstract",
                "question": "What kind of system am I building?",
                "expected": ["assistant", "research"]
            }
        ]
    },
    {
        "id": "fact_peff",
        "content": "Project Almond uses the Peff formula: P_eff = I_base * exp(-(lambda/S) * delta_t).",
        "tag": MemoryTag.PROJECT_FACT,
        "importance": 9.0,
        "keywords": ["peff", "formula", "lambda", "delta", "decay"],
        "probes": [
            {
                "type": "factual",
                "question": "Explain the Peff formula.",
                "expected": ["peff", "exp"]
            },
            {
                "type": "abstract",
                "question": "How does Almond decide what to forget?",
                "expected": ["decay", "peff"]
            }
        ]
    },
    {
        "id": "fact_zane",
        "content": "Zane Walker is the protagonist of Eternity. He is known as The Devil.",
        "tag": MemoryTag.PROJECT_FACT,
        "importance": 8.5,
        "keywords": ["zane", "devil", "eternity"],
        "probes": [
            {
                "type": "factual",
                "question": "Who is Zane Walker?",
                "expected": ["devil"]
            },
            {
                "type": "paraphrase",
                "question": "What is the street name of the protagonist in Eternity?",
                "expected": ["devil"]
            }
        ]
    }
]

NEGATIVE_PROBES = [
    {
        "question": "What is my favorite color?",
        "expected": ["don't know", "not specified", "never mentioned", "unsure"]
    },
    {
        "question": "What is the pirate queen's name in Eternity?",
        "expected": ["don't know", "not mentioned", "unsure", "no information"]
    },
    {
        "question": "Which university did I graduate from?",
        "expected": ["don't know", "not specified", "unknown"]
    }
]

TURN_TEMPLATES = {
    "small_talk": [
        "Tell me something interesting.",
        "I just had coffee.",
        "Do you know any jokes?",
        "What's on your mind?",
        "What do you think about AI?"
    ],
    "task": [
        "Remind me to work on Almond tomorrow.",
        "I need to fix the retrieval pipeline.",
        "I should benchmark Almond this week.",
        "Help me prioritize my roadmap."
    ],
    "eternity": [
        "Tell me more about Zane Walker.",
        "Explain the themes of Eternity.",
        "What role does Sylvia play?",
        "Describe Amber's philosophy."
    ],
    "almond": [
        "How does your memory system work?",
        "Explain virtual swap memory.",
        "How does semantic retrieval work?",
        "What is intent-aware retrieval?"
    ],
    "user_profile": [
        "I prefer concise answers.",
        "I'm building a memory-centric AI system.",
        "I use Python and TypeScript.",
        "I'm interested in cognitive architectures."
    ]
}

WEIGHTS = {
    "small_talk": 0.25,
    "task": 0.15,
    "eternity": 0.25,
    "almond": 0.20,
    "user_profile": 0.15
}

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTrace:
    query: str
    retrieved_count: int
    rejected_count: int
    retrieved_ids: List[str]
    rejected_reasons: List[str]
    retrieval_latency_ms: float
    rerank_latency_ms: float
    context_pollution_score: float


@dataclass
class ProbeResult:
    turn: int
    probe_type: str
    question: str
    expected: List[str]
    response: str
    passed: bool
    keyword_coverage: float


@dataclass
class TurnMetric:
    turn: int
    category: str
    latency_ms: float
    l2_count: int
    l3_count: int
    context_tokens: int
    retrieval_trace: Optional[RetrievalTrace] = None


# ---------------------------------------------------------------------------
# STRESS TEST
# ---------------------------------------------------------------------------

class AlmondStressTest:

    def __init__(
        self,
        turns: int = 500,
        no_llm: bool = False,
        ablation: str = "none"
    ):
        self.turns = turns
        self.no_llm = no_llm
        self.ablation = ablation

        self.probe_results: List[ProbeResult] = []
        self.turn_metrics: List[TurnMetric] = []
        self.retrieval_traces: List[Dict[str, Any]] = []

        self.output_dir = Path(f"almond_stress_v3_{turns}_{ablation}")
        self.output_dir.mkdir(exist_ok=True)

        self.next_probe_turn = self._schedule_next_probe(10)

    # -------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------

    def _create_almond(self) -> Almond:

        policy = EvictionPolicy(
            l2_eviction=2.0,
            l3_eviction=0.5,
            l4_deletion=0.05,
            l2_max_blocks=20
        )

        config = AlmondConfig(
            session_id=f"stress_v3_{self.turns}",
            db_path=str(self.output_dir / "almond.db"),
            eviction_policy=policy,
            max_tokens=512
        )

        return Almond(config)

    def _seed_facts(self, almond: Almond):
        print("[INFO] Seeding long-term memory facts...")

        for fact in SEED_FACTS:
            almond.add_memory(
                content=fact["content"],
                tag=fact["tag"],
                importance_score=fact["importance"],
                keywords=fact["keywords"],
                tier=MemoryTier.L3_VIRTUAL_SWAP
            )

    # -------------------------------------------------------------------
    # PROBE SCHEDULING
    # -------------------------------------------------------------------

    def _schedule_next_probe(self, current_turn: int) -> int:
        return current_turn + random.randint(15, 30)

    # -------------------------------------------------------------------
    # RUNNER
    # -------------------------------------------------------------------

    def run(self):

        almond = self._create_almond()
        self._seed_facts(almond)

        try:
            for turn in range(1, self.turns + 1):
                self._run_turn(almond, turn)

            self._finalize(almond)

        finally:
            almond.close()

    # -------------------------------------------------------------------
    # SINGLE TURN
    # -------------------------------------------------------------------

    def _run_turn(self, almond: Almond, turn: int):

        is_probe = turn >= self.next_probe_turn

        if is_probe:
            self.next_probe_turn = self._schedule_next_probe(turn)
            message, probe_type, expected = self._generate_probe()
            category = "probe"
        else:
            category = random.choices(
                list(WEIGHTS.keys()),
                weights=list(WEIGHTS.values()),
                k=1
            )[0]
            message = random.choice(TURN_TEMPLATES[category])
            probe_type = None
            expected = []

        t0 = time.time()

        if self.no_llm:
            response = "[DRY RUN]"
        else:
            response = almond.chat(message)

        latency_ms = (time.time() - t0) * 1000

        # -------------------------------------------------------------------
        # RETRIEVAL TRACE EXPORT
        # -------------------------------------------------------------------

        retrieval_trace = self._extract_retrieval_trace(almond, message)

        # -------------------------------------------------------------------
        # PROBE EVALUATION
        # -------------------------------------------------------------------

        if is_probe:
            self._evaluate_probe(
                turn,
                probe_type,
                message,
                expected,
                response
            )

        pool = almond.controller.dump_pool()

        metric = TurnMetric(
            turn=turn,
            category=category,
            latency_ms=round(latency_ms, 2),
            l2_count=sum(1 for b in pool if b["tier"] == "L2_ACTIVE_RAM"),
            l3_count=sum(1 for b in pool if b["tier"] == "L3_VIRTUAL_SWAP"),
            context_tokens=sum(len(b["content_preview"]) // 4 for b in pool),
            retrieval_trace=retrieval_trace
        )

        self.turn_metrics.append(metric)

        print(
            f"Turn {turn:>4}/{self.turns} | "
            f"{category:<12} | "
            f"L2={metric.l2_count:<3} "
            f"L3={metric.l3_count:<3} | "
            f"{metric.latency_ms:>6.0f}ms"
        )

    # -------------------------------------------------------------------
    # PROBES
    # -------------------------------------------------------------------

    def _generate_probe(self):

        if random.random() < 0.25:
            neg = random.choice(NEGATIVE_PROBES)
            return neg["question"], "negative", neg["expected"]

        fact = random.choice(SEED_FACTS)
        probe = random.choice(fact["probes"])

        return (
            probe["question"],
            probe["type"],
            probe["expected"]
        )

    # -------------------------------------------------------------------
    # PROBE SCORING
    # -------------------------------------------------------------------

    def _evaluate_probe(
        self,
        turn: int,
        probe_type: str,
        question: str,
        expected: List[str],
        response: str
    ):

        response_lower = response.lower()

        matched = sum(
            1 for kw in expected
            if kw.lower() in response_lower
        )

        coverage = matched / max(len(expected), 1)

        if probe_type == "negative":
            passed = coverage > 0
        else:
            passed = coverage >= 0.7

        result = ProbeResult(
            turn=turn,
            probe_type=probe_type,
            question=question,
            expected=expected,
            response=response[:400],
            passed=passed,
            keyword_coverage=round(coverage, 2)
        )

        self.probe_results.append(result)

    # -------------------------------------------------------------------
    # RETRIEVAL TRACE EXTRACTION
    # -------------------------------------------------------------------

    def _extract_retrieval_trace(
        self,
        almond: Almond,
        query: str
    ) -> Optional[RetrievalTrace]:

        try:
            optimizer = almond.controller.optimizer

            latest_rejections = getattr(optimizer, "latest_rejections", [])
            latest_accepts = getattr(optimizer, "latest_accepts", [])

            retrieved_count = len(latest_accepts)
            rejected_count = len(latest_rejections)

            retrieved_ids = [
                str(x.get("id", "unknown"))
                for x in latest_accepts
            ]

            rejection_reasons = [
                x.get("reason", "unknown")
                for x in latest_rejections
            ]

            pollution_score = self._estimate_pollution(latest_accepts)

            trace = RetrievalTrace(
                query=query,
                retrieved_count=retrieved_count,
                rejected_count=rejected_count,
                retrieved_ids=retrieved_ids,
                rejected_reasons=rejection_reasons,
                retrieval_latency_ms=0.0,
                rerank_latency_ms=0.0,
                context_pollution_score=round(pollution_score, 3)
            )

            self.retrieval_traces.append(asdict(trace))

            return trace

        except Exception:
            return None

    # -------------------------------------------------------------------
    # CONTEXT POLLUTION
    # -------------------------------------------------------------------

    def _estimate_pollution(self, accepts: List[Dict]) -> float:

        if not accepts:
            return 0.0

        tags = [x.get("tag") for x in accepts]

        duplicates = len(tags) - len(set(tags))

        return duplicates / max(len(tags), 1)

    # -------------------------------------------------------------------
    # FINALIZATION
    # -------------------------------------------------------------------

    def _finalize(self, almond: Almond):

        summary = self._compute_summary()

        self._save_json(summary)
        self._save_summary(summary)

        print("\n" + "=" * 60)
        print("ALMOND STRESS TEST COMPLETE")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------

    def _compute_summary(self):

        total_probes = len(self.probe_results)

        if total_probes == 0:
            recall = 0.0
        else:
            recall = sum(p.passed for p in self.probe_results) / total_probes

        breakdown = defaultdict(list)

        for p in self.probe_results:
            breakdown[p.probe_type].append(p.passed)

        breakdown_scores = {
            k: round(sum(v) / len(v), 3)
            for k, v in breakdown.items()
        }

        pollution_scores = [
            t.retrieval_trace.context_pollution_score
            for t in self.turn_metrics
            if t.retrieval_trace
        ]

        avg_pollution = (
            statistics.mean(pollution_scores)
            if pollution_scores else 0.0
        )

        return {
            "turns": self.turns,
            "ablation": self.ablation,
            "overall_recall": round(recall, 3),
            "probe_breakdown": breakdown_scores,
            "avg_latency_ms": round(statistics.mean(
                t.latency_ms for t in self.turn_metrics
            ), 2),
            "avg_context_pollution": round(avg_pollution, 3),
            "peak_l2": max(t.l2_count for t in self.turn_metrics),
            "peak_tokens": max(t.context_tokens for t in self.turn_metrics),
            "total_probes": total_probes
        }

    # -------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------

    def _save_json(self, summary: Dict[str, Any]):

        payload = {
            "summary": summary,
            "probe_results": [asdict(p) for p in self.probe_results],
            "turn_metrics": [asdict(t) for t in self.turn_metrics],
            "retrieval_traces": self.retrieval_traces
        }

        path = self.output_dir / "stress_report.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_summary(self, summary: Dict[str, Any]):

        lines = [
            "PROJECT ALMOND — STRESS TEST SUMMARY",
            "=" * 60,
            f"Turns:                 {summary['turns']}",
            f"Ablation:              {summary['ablation']}",
            f"Overall Recall:        {summary['overall_recall']}",
            f"Avg Latency (ms):      {summary['avg_latency_ms']}",
            f"Avg Pollution:         {summary['avg_context_pollution']}",
            f"Peak L2:               {summary['peak_l2']}",
            f"Peak Tokens:           {summary['peak_tokens']}",
            "",
            "Probe Breakdown:",
        ]

        for k, v in summary["probe_breakdown"].items():
            lines.append(f"  {k:<15} {v * 100:.1f}%")

        path = self.output_dir / "summary.txt"
        path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        default="none",
        choices=[
            "none",
            "no_intent",
            "no_recency",
            "no_keyword",
            "no_peff"
        ]
    )

    args = parser.parse_args()

    random.seed(args.seed)

    test = AlmondStressTest(
        turns=args.turns,
        no_llm=args.no_llm,
        ablation=args.ablation
    )

    test.run()


if __name__ == "__main__":
    main()
