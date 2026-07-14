"""
Project Almond — Official LongMemEval Adapter
Parses the huggingface longmemeval_s_cleaned.json schema.

Usage: 
    python eval_longmem.py
"""

import json
import logging
import time
from pathlib import Path
import requests

from almond import Almond, AlmondConfig
from memory_block import MemoryTag, MemoryTier
from memory_controller_v2 import EvictionPolicy
from memory_store import MemoryStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR) # Suppress standard logs to keep console clean

# LM Studio Local API endpoint
LLM_API_URL = "http://localhost:1234/v1/chat/completions"

def llm_judge(question: str, expected_answer: str, model_response: str) -> bool:
    """Uses local LM Studio to grade the response based on the official ICLR 2025 prompt."""
    judge_prompt = (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer 'yes' if the response contains the correct answer. Otherwise, answer 'no'. "
        "If the response is equivalent to the correct answer or contains all the intermediate steps "
        "to get the correct answer, you should also answer 'yes'. If the response only contains a "
        "subset of the information required by the answer, answer 'no'.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {expected_answer}\n"
        f"Model Response: {model_response}\n\n"
        "Decision (yes/no):"
    )

    payload = {
        "model": "llama-3.1-8b-instruct", 
        "messages": [
            {"role": "system", "content": "You are an impartial, strict academic evaluator."},
            {"role": "user", "content": judge_prompt}
        ],
        "temperature": 0.0, 
        "max_tokens": 10
    }

    try:
        response = requests.post(LLM_API_URL, json=payload).json()
        decision = response['choices'][0]['message']['content'].strip().lower()
        return "yes" in decision
    except Exception as e:
        print(f"[JUDGE ERROR] Failed to parse LLM judgment: {e}")
        return False

def run_longmem_eval(dataset_path: str = "longmemeval_dataset.json"):
    print(f"\n{'='*60}")
    print("  PROJECT ALMOND — OFFICIAL LONGMEMEVAL BENCHMARK")
    print(f"{'='*60}\n")

    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Init config for Full T-MMU
    policy = EvictionPolicy(l2_eviction=2.0, l3_eviction=0.5, l4_deletion=0.05, l2_max_blocks=20)
    config = AlmondConfig(
        session_id="longmem_eval",
        db_path="longmem_almond.db",
        eviction_policy=policy
    )

    total_questions = len(dataset)
    passed_questions = 0
    start_time = time.time()

    print(f"Loaded {total_questions} evaluation instances from {dataset_path}...")
    
    # We will test the first 50 questions to keep the run-time reasonable 
    # (You can remove the [:50] slice later to run all 500)
    dataset_subset = dataset[:50] 

    for idx, instance in enumerate(dataset_subset, 1):
        question = instance['question']
        expected_answer = instance['answer']
        history_sessions = instance['haystack_sessions'] # Official schema key
        q_type = instance['question_type']

        print(f"\n--- Q{idx}/{len(dataset_subset)} [{q_type}] ---")
        
        # Start a fresh brain for every single question
        Path("longmem_almond.db").unlink(missing_ok=True)
        Path("longmem_almond_chroma").unlink(missing_ok=True)
        almond = Almond(config)

        # 1. Ingest the "Haystack" (History)
        for session in history_sessions:
            for turn in session:
                content = turn['content']
                # Feed to Almond as L2_ACTIVE_RAM, forcing the T-MMU to actively manage the bloat
                almond.add_memory(
                    content=content, 
                    tag=MemoryTag.SMALL_TALK, 
                    importance_score=1.0, 
                    keywords=[],
                    tier=MemoryTier.L2_ACTIVE_RAM
                )
        # 2. Ask the Benchmark Question
        try:
            model_response = almond.chat(question)
        except Exception as e:
            model_response = f"[OOM CRASH] {e}"

        # 3. Judge the Answer
        is_correct = llm_judge(question, expected_answer, model_response)
        
        if is_correct:
            passed_questions += 1
            print(f"[PASS] ✓ {question}")
        else:
            print(f"[FAIL] ✗ {question}")
            print(f"         Expected: {expected_answer}")
            print(f"         Got: {model_response[:120]}...")

        almond.close()

    # --- Final Results ---
    elapsed = time.time() - start_time
    accuracy = (passed_questions / len(dataset_subset)) * 100

    print(f"\n{'='*60}")
    print("  LONGMEMEVAL RESULTS (SUBSET: 50)")
    print(f"{'='*60}")
    print(f"  Passed:           {passed_questions} / {len(dataset_subset)}")
    print(f"  Accuracy:         {accuracy:.2f}%")
    print(f"  Time Elapsed:     {elapsed:.1f} seconds")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_longmem_eval()