"""
Project Almond — Unified Evaluation Orchestrator
================================================

Purpose
-------
Automates overnight benchmark execution for:
- Baseline runs
- Ablation studies
- Replay evaluations
- LongMemEval benchmarking
- Failure recovery
- Incremental checkpointing
- Crash-safe execution
- Summary leaderboard generation

Run Once:
---------
python eval_runner.py

Custom:
-------
python eval_runner.py --dataset longmemeval_dataset.json --limit 50

What This Does:
---------------
1. Runs all configured evaluations automatically
2. Retries failed runs safely
3. Saves incremental checkpoints
4. Exports JSON + CSV summaries
5. Continues after crashes
6. Generates final leaderboard

Folder Structure:
-----------------
results/
├── baseline/
├── no_intent/
├── no_keyword/
├── no_recency/
├── no_peff/
├── failures/
├── checkpoints/
└── leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

# ==========================================================================
# CONFIG
# ==========================================================================

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FAILURE_DIR = RESULTS_DIR / "failures"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
FAILURE_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

# IMPORTANT:
# This must match your evaluation file name.
EVAL_SCRIPT = "eval_unified.py"

# ==========================================================================
# RUN CONFIGS
# ==========================================================================

RUN_CONFIGS = [
    {
        "name": "baseline",
        "ablation": "none"
    },
    {
        "name": "no_intent",
        "ablation": "no_intent"
    },
    {
        "name": "no_keyword",
        "ablation": "no_keyword"
    },
    {
        "name": "no_recency",
        "ablation": "no_recency"
    },
    {
        "name": "no_peff",
        "ablation": "no_peff"
    }
]

# ==========================================================================
# DATA STRUCTURES
# ==========================================================================

@dataclass
class RunResult:
    name: str
    success: bool
    accuracy: float
    avg_latency_ms: float
    avg_pollution: float
    elapsed_seconds: float
    retries_used: int
    error: str = ""


# ==========================================================================
# CHECKPOINTS
# ==========================================================================

CHECKPOINT_FILE = CHECKPOINT_DIR / "progress.json"


def load_checkpoint() -> Dict:

    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "completed": []
    }



def save_checkpoint(data: Dict):

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ==========================================================================
# RUNNER
# ==========================================================================


def run_single_eval(
    config: Dict,
    dataset_path: str,
    limit: int
) -> RunResult:

    name = config["name"]
    ablation = config["ablation"]

    print("\n" + "=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)

    run_dir = RESULTS_DIR / name
    run_dir.mkdir(exist_ok=True)

    retries = 0

    while retries < MAX_RETRIES:

        try:
            start = time.time()

            command = [
                sys.executable,
                EVAL_SCRIPT,
                "--dataset",
                dataset_path,
                "--limit",
                str(limit),
                "--ablation",
                ablation
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours max per run
            )

            elapsed = time.time() - start

            # --------------------------------------------------------------
            # STDOUT / STDERR EXPORT
            # --------------------------------------------------------------

            (run_dir / "stdout.txt").write_text(
                process.stdout,
                encoding="utf-8"
            )

            (run_dir / "stderr.txt").write_text(
                process.stderr,
                encoding="utf-8"
            )

            # --------------------------------------------------------------
            # FAILURE
            # --------------------------------------------------------------

            if process.returncode != 0:

                raise RuntimeError(
                    f"Evaluation failed with return code {process.returncode}"
                )

            # --------------------------------------------------------------
            # LOAD REPORT
            # --------------------------------------------------------------

            report_path = Path("longmem_eval_results") / "longmem_report.json"

            if not report_path.exists():
                raise RuntimeError("Missing report JSON output")

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            # copy stable snapshot
            shutil.copy(
                report_path,
                run_dir / "report.json"
            )

            summary = report["summary"]

            return RunResult(
                name=name,
                success=True,
                accuracy=summary.get("accuracy", 0.0),
                avg_latency_ms=summary.get("avg_latency_ms", 0.0),
                avg_pollution=summary.get("avg_pollution", 0.0),
                elapsed_seconds=round(elapsed, 2),
                retries_used=retries
            )

        except subprocess.TimeoutExpired:

            error_msg = f"Timeout after 2 hours"

        except Exception as e:

            error_msg = str(e)

        # --------------------------------------------------------------
        # RETRY HANDLING
        # --------------------------------------------------------------

        retries += 1

        print(f"[RETRY {retries}/{MAX_RETRIES}] {error_msg}")

        failure_payload = {
            "run": name,
            "retry": retries,
            "error": error_msg,
            "timestamp": time.time()
        }

        with open(
            FAILURE_DIR / f"{name}_retry_{retries}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(failure_payload, f, indent=2)

        time.sleep(RETRY_DELAY_SECONDS)

    # ----------------------------------------------------------------------
    # FINAL FAILURE
    # ----------------------------------------------------------------------

    return RunResult(
        name=name,
        success=False,
        accuracy=0.0,
        avg_latency_ms=0.0,
        avg_pollution=0.0,
        elapsed_seconds=0.0,
        retries_used=retries,
        error=error_msg
    )


# ==========================================================================
# LEADERBOARD
# ==========================================================================


def export_leaderboard(results: List[RunResult]):

    leaderboard_path = RESULTS_DIR / "leaderboard.csv"

    with open(leaderboard_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "run",
            "success",
            "accuracy",
            "avg_latency_ms",
            "avg_pollution",
            "elapsed_seconds",
            "retries_used",
            "error"
        ])

        for r in results:
            writer.writerow([
                r.name,
                r.success,
                r.accuracy,
                r.avg_latency_ms,
                r.avg_pollution,
                r.elapsed_seconds,
                r.retries_used,
                r.error
            ])


# ==========================================================================
# FINAL SUMMARY
# ==========================================================================


def print_summary(results: List[RunResult]):

    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)

    ranked = sorted(
        results,
        key=lambda x: x.accuracy,
        reverse=True
    )

    for idx, r in enumerate(ranked, 1):

        status = "PASS" if r.success else "FAIL"

        print(
            f"[{idx}] {r.name:<15} | "
            f"{status:<5} | "
            f"Acc={r.accuracy:>6.2f}% | "
            f"Pollution={r.avg_pollution:<5} | "
            f"Latency={r.avg_latency_ms:<8}ms | "
            f"Retries={r.retries_used}"
        )

    print("=" * 70)


# ==========================================================================
# MAIN
# ==========================================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="longmemeval_dataset.json"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoints and rerun everything"
    )

    args = parser.parse_args()

    checkpoint = load_checkpoint()

    if args.fresh:
        checkpoint = {"completed": []}

    completed = set(checkpoint["completed"])

    results: List[RunResult] = []

    overall_start = time.time()

    # ======================================================================
    # RUN ALL CONFIGS
    # ======================================================================

    for config in RUN_CONFIGS:

        name = config["name"]

        if name in completed:
            print(f"[SKIP] {name} already completed")
            continue

        result = run_single_eval(
            config=config,
            dataset_path=args.dataset,
            limit=args.limit
        )

        results.append(result)

        # ------------------------------------------------------------------
        # CHECKPOINT SAVE
        # ------------------------------------------------------------------

        if result.success:
            completed.add(name)

        save_checkpoint({
            "completed": list(completed)
        })

        # incremental export
        export_leaderboard(results)

    # ======================================================================
    # FINAL EXPORTS
    # ======================================================================

    elapsed = round(time.time() - overall_start, 2)

    summary_payload = {
        "elapsed_seconds": elapsed,
        "results": [
            asdict(r)
            for r in results
        ]
    }

    with open(
        RESULTS_DIR / "final_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary_payload, f, indent=2)

    export_leaderboard(results)
    print_summary(results)

    print(f"\nAll results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
