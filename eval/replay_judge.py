import sys, os, json, argparse, inspect
from pathlib import Path

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.judge_v2 import judge as judge_v2

REPORT_PATH = Path(__file__).parent / "longmem_eval_results" / "partial_report.json"

def main():
    parser = argparse.ArgumentParser(description="Replay judge on existing eval results")
    parser.add_argument("--report", type=str, default=str(REPORT_PATH), help="Path to partial_report.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM-dependent gates (extraction, comparison)")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", [])
    print(f"Loaded {len(results)} results from {args.report}")
    print(f"Note: 15 crashed instances (61,62,66,71,74,78,80,81,82,84,89,91,94,97,99) are NOT in this file.")
    print(f"      They require a full re-run with the ingestion cache.\n")

    flipped_to_pass = []
    flipped_to_fail = []
    unchanged = []
    old_pass = 0
    new_pass = 0

    judge_sig = inspect.signature(judge_v2)

    for r in results:
        idx = r["index"]
        q = r["question"]
        expected = str(r["expected_answer"])
        response = r["model_response"]
        old_passed = r["passed"]
        old_gate = r.get("judge_gate", "")
        q_type = r.get("question_type", None)

        if old_passed:
            old_pass += 1

        if args.no_llm:
            # Only re-run if the old gate was deterministic
            deterministic_gates = {"ERROR_GATE", "ABSTENTION_GATE", "NUMERIC_GATE", "SUBSTRING_MATCH"}
            if old_gate not in deterministic_gates:
                # Can't re-judge without LLM, keep old result
                if old_passed:
                    new_pass += 1
                unchanged.append(idx)
                continue

        try:
            judge_kwargs = {
                "question": q,
                "expected_answer": expected,
                "model_response": response,
                "verbose": False,
            }
            if "question_type" in judge_sig.parameters:
                judge_kwargs["question_type"] = q_type

            new_result = judge_v2(**judge_kwargs)
            new_passed = new_result.passed
            new_gate = new_result.gate
        except Exception as e:
            print(f"  [ERROR] Instance {idx}: {e}")
            new_passed = False
            new_gate = "REPLAY_ERROR"

        if new_passed:
            new_pass += 1

        if old_passed and not new_passed:
            flipped_to_fail.append((idx, old_gate, new_gate, q[:80]))
        elif not old_passed and new_passed:
            flipped_to_pass.append((idx, old_gate, new_gate, q[:80]))
        else:
            unchanged.append(idx)

    print(f"{'='*60}")
    print(f"REPLAY JUDGE RESULTS")
    print(f"{'='*60}")
    print(f"Total re-scored: {len(results)}")
    print(f"Old accuracy: {old_pass}/{len(results)} = {old_pass/len(results)*100:.1f}%")
    print(f"New accuracy: {new_pass}/{len(results)} = {new_pass/len(results)*100:.1f}%")
    print(f"")
    print(f"Flipped PASS -> FAIL: {len(flipped_to_fail)}")
    for idx, old_g, new_g, q in flipped_to_fail:
        print(f"  [{idx}] {old_g} -> {new_g} | {q}")
    print(f"")
    print(f"Flipped FAIL -> PASS: {len(flipped_to_pass)}")
    for idx, old_g, new_g, q in flipped_to_pass:
        print(f"  [{idx}] {old_g} -> {new_g} | {q}")
    print(f"")
    print(f"Unchanged: {len(unchanged)}")
    print(f"")
    print(f"True accuracy estimate (including 15 crashed as failures):")
    print(f"  {new_pass}/100 = {new_pass/100*100:.1f}%")

if __name__ == "__main__":
    main()
