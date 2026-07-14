"""
judge_diagnostic.py
-------------------
Drop this file next to eval_unified.py and run it against any eval results JSON.

Usage:
    python judge_diagnostic.py results.json

What it does:
    1. Reads the eval results JSON
    2. For every failed question, checks whether the model response
       actually contains the semantic content of the expected answer
    3. Flags cases where the judge likely failed (correct answer, wrong verdict)
    4. Reports the adjusted accuracy if those cases were counted correctly
    5. Explains what to fix in the judge prompt

The core problem this addresses:
    eval_unified.py judge is likely doing substring/exact match.
    Model responses like:
        "You got the Samsung Galaxy S22 from Best Buy on Feb 20th, before the Dell"
    should pass for expected answer "Samsung Galaxy S22" but fail substring match
    because the response doesn't START with "Samsung Galaxy S22".
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Semantic overlap check (no LLM needed)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def _key_tokens(text: str) -> set[str]:
    stopwords = {"the","a","an","i","you","it","was","is","did","do",
                 "my","your","first","before","after","when","which","what"}
    return {w for w in _normalise(text).split() if w not in stopwords and len(w) > 2}

def semantic_overlap(expected: str, response: str) -> float:
    """
    Fraction of key tokens from expected_answer that appear in model_response.
    Score >= 0.60 strongly suggests the model gave the correct answer.
    """
    exp_tokens = _key_tokens(expected)
    if not exp_tokens:
        return 0.0
    res_tokens = _key_tokens(response)
    hit = exp_tokens & res_tokens
    return len(hit) / len(exp_tokens)

def likely_correct(expected: str, response: str,
                   threshold: float = 0.60) -> tuple[bool, float]:
    score = semantic_overlap(expected, response)
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Diagnosis per question
# ---------------------------------------------------------------------------

@dataclass
class QuestionDiagnosis:
    index:            int
    question:         str
    expected:         str
    response:         str
    judge_passed:     bool
    overlap_score:    float
    likely_correct:   bool
    verdict:          str   # "TRUE_FAIL" | "JUDGE_ERROR" | "TRUE_PASS"
    note:             str


def diagnose(result: dict) -> QuestionDiagnosis:
    idx      = result.get("index", 0)
    question = result.get("question", "")
    expected = result.get("expected_answer", "")
    response = result.get("model_response", "")
    passed   = result.get("passed", False)

    prob, score = likely_correct(expected, response)

    if passed:
        verdict = "TRUE_PASS"
        note    = "Judge and semantic check agree: correct."
    elif prob:
        verdict = "JUDGE_ERROR"
        note    = (
            f"Model response has {score:.0%} token overlap with expected answer. "
            f"Judge likely used exact/substring match and missed a correct paraphrase."
        )
    else:
        verdict = "TRUE_FAIL"
        note    = f"Model response has only {score:.0%} overlap. Retrieval or reasoning failed."

    return QuestionDiagnosis(
        index=idx, question=question, expected=expected,
        response=response, judge_passed=passed,
        overlap_score=score, likely_correct=prob,
        verdict=verdict, note=note,
    )


# ---------------------------------------------------------------------------
# Retrieval trace analysis
# ---------------------------------------------------------------------------

def analyse_trace(trace: dict, result: dict) -> list[str]:
    issues = []
    ret_count = trace.get("retrieved_count", 0)
    ret_ids   = trace.get("retrieved_ids", [])
    l2_peak   = result.get("l2_peak", 0)
    l3_peak   = result.get("l3_peak", 0)

    if ret_count > 0 and not ret_ids:
        issues.append(
            f"retrieved_count={ret_count} but retrieved_ids=[]. "
            f"Chroma found candidates but SQLite hydration returned nothing. "
            f"Likely Chroma/SQLite desync — orphaned Chroma entries."
        )

    if ret_ids and l2_peak == 0:
        issues.append(
            f"{len(ret_ids)} IDs paged in but l2_peak=0. "
            f"Eval is reading l2_peak before page-in completes, "
            f"or reading from old RetrievalOptimizer trace field. "
            f"Fix: read controller.l2_count AFTER prepare_context() returns."
        )

    if l3_peak > 0 and l2_peak == 0 and not ret_ids:
        issues.append(
            f"l3_peak={l3_peak} means memories exist in cold storage. "
            f"Page-in failed entirely — check Chroma/SQLite sync and tier filter."
        )

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(path: str):
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    traces  = {t["query"]: t for t in data.get("retrieval_traces", [])}
    summary = data.get("summary", {})

    print("=" * 70)
    print("JUDGE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Questions: {summary.get('questions', len(results))}")
    print(f"Reported accuracy: {summary.get('accuracy', 0.0):.1%}")
    print()

    diagnoses      = [diagnose(r) for r in results]
    judge_errors   = [d for d in diagnoses if d.verdict == "JUDGE_ERROR"]
    true_fails     = [d for d in diagnoses if d.verdict == "TRUE_FAIL"]
    true_passes    = [d for d in diagnoses if d.verdict == "TRUE_PASS"]

    adjusted_correct = len(true_passes) + len(judge_errors)
    adjusted_accuracy = adjusted_correct / len(diagnoses) if diagnoses else 0.0

    print(f"True passes (judge + semantic agree) : {len(true_passes)}")
    print(f"Likely judge errors (correct answer) : {len(judge_errors)}")
    print(f"True failures (wrong answer)         : {len(true_fails)}")
    print(f"Adjusted accuracy                    : {adjusted_accuracy:.1%}  "
          f"(vs reported {summary.get('accuracy',0.0):.1%})")
    print()

    # Per-question detail
    for d in diagnoses:
        trace  = traces.get(d.question, {})
        issues = analyse_trace(trace, next(
            (r for r in results if r.get("question") == d.question), {}
        ))

        icon = {"TRUE_PASS": "✓", "JUDGE_ERROR": "⚠", "TRUE_FAIL": "✗"}[d.verdict]
        print(f"{icon} Q{d.index} [{d.verdict}]")
        print(f"  Question : {d.question[:70]}")
        print(f"  Expected : {d.expected}")
        print(f"  Response : {d.response[:120]}")
        print(f"  Overlap  : {d.overlap_score:.0%}")
        print(f"  Note     : {d.note}")
        for issue in issues:
            print(f"  TRACE    : {issue}")
        print()

    # Judge fix recommendation
    if judge_errors:
        print("=" * 70)
        print("JUDGE FIX REQUIRED")
        print("=" * 70)
        print()
        print("The judge is marking correct paraphrases as failures.")
        print("This is a substring/exact-match problem.")
        print()
        print("Replace the judge evaluation with this prompt pattern:")
        print()
        print('  JUDGE_PROMPT = """')
        print('  Question: {question}')
        print('  Expected answer: {expected}')
        print('  Model response: {response}')
        print()
        print('  Does the model response correctly answer the question?')
        print('  The response does not need to use the exact wording of the expected answer.')
        print('  It just needs to convey the same factual information.')
        print()
        print('  Answer with exactly one word: YES or NO')
        print('  """')
        print()
        print("Key change: 'does not need to use the exact wording'")
        print("This is the line that fixes Q2 and Q4.")
        print()

    # Retrieval fix recommendation
    all_traces = [traces.get(r.get("question",""), {}) for r in results]
    has_desync = any(
        t.get("retrieved_count",0) > 0 and not t.get("retrieved_ids",[])
        for t in all_traces
    )
    if has_desync:
        print("=" * 70)
        print("CHROMA/SQLITE DESYNC DETECTED")
        print("=" * 70)
        print()
        print("Some queries found Chroma candidates but got empty paged_in.")
        print("This means Chroma has entries from a previous run that SQLite")
        print("no longer has. The orphan cleanup in memory_controller_v2.py")
        print("will handle this automatically on next run by deleting ghost")
        print("Chroma entries when get_blocks_by_ids() returns nothing for them.")
        print()
        print("To force a clean state now:")
        print("  1. Delete ./almond_chroma_db/ directory")
        print("  2. Delete almond.db")
        print("  3. Re-run benchmark — both stores rebuild from scratch in sync")
        print()
        print("OR if you want to keep existing data:")
        print("  controller._collection.delete(where={'tier': {'$exists': True}})")
        print("  Then re-run save() on all blocks in SQLite to rebuild Chroma index.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python judge_diagnostic.py results.json")
        sys.exit(1)
    run(sys.argv[1])