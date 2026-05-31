"""
Project Almond — CLI Test Harness
Smoke-tests the full T-MMU stack and provides an interactive REPL.

Modes:
    python cli.py test       — run automated smoke tests (no API key needed)
    python cli.py chat       — interactive session with live Almond
    python cli.py dump       — print current memory pool state from DB
    python cli.py reset      — wipe almond.db and start fresh

Usage:
    pip install anthropic openai pydantic
    export ANTHROPIC_API_KEY=sk-...
    python cli.py chat
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — readable single-line format for CLI output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR  = "─" * 60
DIM        = "\033[2m"
CYAN       = "\033[96m"
GREEN      = "\033[92m"
YELLOW     = "\033[93m"
RED        = "\033[91m"
BOLD       = "\033[1m"
RESET      = "\033[0m"

def c(color: str, text: str) -> str:
    """Colorize text for terminal output."""
    return f"{color}{text}{RESET}"

def banner() -> None:
    print(c(CYAN, BOLD + """
    ╔═══════════════════════════════════════╗
    ║       PROJECT ALMOND  —  T-MMU        ║
    ║   Temporal Memory Management Unit     ║
    ╚═══════════════════════════════════════╝
    """ + RESET))


# ---------------------------------------------------------------------------
# Mode 1: Smoke Tests (no API key required)
# ---------------------------------------------------------------------------

def run_smoke_tests() -> None:
    """
    Validates the full stack without hitting any LLM API.
    Tests: schema, scoring, eviction, persistence, page-in, rehydration.
    """
    from memory_block import MemoryBlock, MemoryTag, MemoryTier
    from memory_store import MemoryStore
    from memory_controller_v2 import MemoryController, EvictionPolicy

    DB_PATH = Path("almond_test.db")
    DB_PATH.unlink(missing_ok=True)  # fresh slate

    passed = 0
    failed = 0

    def test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(c(GREEN, f"  ✓ {name}"))
            passed += 1
        else:
            print(c(RED, f"  ✗ {name}" + (f" — {detail}" if detail else "")))
            failed += 1

    print(f"\n{c(BOLD, 'SMOKE TESTS')}\n{SEPARATOR}")

    # ── 1. MemoryBlock: Peff sanity ──────────────────────────────────────
    print(c(YELLOW, "\n[1] MemoryBlock & Peff Formula"))

    fresh = MemoryBlock(
        content="Project Almond is a T-MMU for LLMs.",
        tag=MemoryTag.PROJECT_FACT,
        importance_score=8.0,
    )
    test("Peff ≈ I_base when Δt ≈ 0", abs(fresh.p_eff - 8.0) < 0.01,
         f"got {fresh.p_eff:.4f}")

    # Simulate 50-day-old block manually
    old = MemoryBlock(
        content="This is very old.",
        tag=MemoryTag.SMALL_TALK,
        importance_score=5.0,
    )
    object.__setattr__(old, "last_accessed_at", time.time() - (50 * 86400))
    expected = 5.0 * math.exp(-(0.20 / 0.1) * 50)
    test("Peff decays correctly after 50 days",
         abs(old.p_eff - expected) < 0.001, f"got {old.p_eff:.6f}")

    test("CORE_RULE decays slower than SMALL_TALK", (
        MemoryBlock(content="rule", tag=MemoryTag.CORE_RULE, importance_score=5.0).p_eff
        > old.p_eff
    ))

    touch_block = MemoryBlock(
        content="touch test", tag=MemoryTag.TASK, importance_score=5.0
    )
    prev_access = touch_block.access_count
    touch_block.touch()
    test("touch() increments access_count", touch_block.access_count == prev_access + 1)

    # ── 2. MemoryStore: persistence ──────────────────────────────────────
    print(c(YELLOW, "\n[2] MemoryStore — SQLite Persistence"))

    with MemoryStore(db_path=DB_PATH) as store:
        b1 = MemoryBlock(
            content="Almond uses exponential decay scoring.",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=9.0,
            keywords=["almond", "decay", "scoring"],
            tier=MemoryTier.L3_VIRTUAL_SWAP,
        )
        store.save(b1)
        retrieved = store.get_by_id(b1.id)
        test("save() + get_by_id() round-trip", retrieved is not None)
        test("Content preserved after round-trip",
             retrieved.content == b1.content if retrieved else False)

        # Keyword match
        matches = store.query_by_keywords(["decay", "scoring"], MemoryTier.L3_VIRTUAL_SWAP)
        test("query_by_keywords() finds matching block", len(matches) == 1)

        no_matches = store.query_by_keywords(["voice", "jarvis"], MemoryTier.L3_VIRTUAL_SWAP)
        test("query_by_keywords() returns empty on no match", len(no_matches) == 0)

        # Tier update
        store.update_tier(b1.id, MemoryTier.L4_ARCHIVE)
        updated = store.get_by_id(b1.id)
        test("update_tier() persists correctly",
             updated.tier == MemoryTier.L4_ARCHIVE if updated else False)

        # Tier counts
        counts = store.tier_counts()
        test("tier_counts() returns dict", isinstance(counts, dict))

    # ── 3. MemoryController: eviction ────────────────────────────────────
    print(c(YELLOW, "\n[3] MemoryController — Eviction & Page-In"))

    DB_PATH.unlink(missing_ok=True)
    policy = EvictionPolicy(l2_eviction=9.5)  # very aggressive — evicts almost everything

    with MemoryStore(db_path=DB_PATH) as store:
        ctrl = MemoryController(policy=policy, store=store)

        # Add a high-importance L1 block (should never evict)
        l1 = MemoryBlock(
            content="System: You are Almond.",
            tag=MemoryTag.CORE_RULE,
            importance_score=10.0,
            tier=MemoryTier.L1_HOT_CACHE,
        )
        ctrl.add(l1)

        # Add a low-importance L2 block (should evict immediately)
        low = MemoryBlock(
            content="User asked about the weather.",
            tag=MemoryTag.SMALL_TALK,
            importance_score=2.0,
            keywords=["weather"],
            tier=MemoryTier.L2_ACTIVE_RAM,
        )
        ctrl.add(low)

        # Add a high-importance L2 block (should survive)
        high = MemoryBlock(
            content="Project Almond core architecture defined.",
            tag=MemoryTag.CORE_RULE,
            importance_score=10.0,
            keywords=["almond", "architecture"],
            tier=MemoryTier.L2_ACTIVE_RAM,
        )
        ctrl.add(high)

        context = ctrl.prepare_context("hello")
        l1_ids = {b.id for b in context if b.tier == MemoryTier.L1_HOT_CACHE}
        test("L1 block always present in context", l1.id in l1_ids)
        test("Low-importance block evicted from context",
             low.id not in {b.id for b in context})

        # Page-in test — mention "weather" to trigger L3 retrieval
        context2 = ctrl.prepare_context("what was the weather we discussed?")
        paged_ids = {b.id for b in context2}
        test("Keyword page-in retrieves evicted block", low.id in paged_ids)

    # ── 4. Rehydration ───────────────────────────────────────────────────
    print(c(YELLOW, "\n[4] Session Rehydration (50-day gap simulation)"))

    DB_PATH.unlink(missing_ok=True)
    with MemoryStore(db_path=DB_PATH) as store:
        ctrl1 = MemoryController(store=store)
        fact = MemoryBlock(
            content="The T-MMU paper deadline is Q3.",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=8.5,
            tier=MemoryTier.L2_ACTIVE_RAM,
            keywords=["deadline", "paper", "q3"],
        )
        ctrl1.add(fact)

    # New controller — simulates restarting Almond after a gap
    with MemoryStore(db_path=DB_PATH) as store:
        ctrl2 = MemoryController(store=store)
        l2_ids = set(ctrl2._l2.keys())
        test("L2 block rehydrated from DB after restart", fact.id in l2_ids)

    # ── Summary ──────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{SEPARATOR}")
    print(c(GREEN if failed == 0 else RED,
            f"  {passed}/{total} tests passed" + (" 🎉" if failed == 0 else f" ({failed} failed)")))
    print(SEPARATOR)

    DB_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Mode 2: Interactive Chat REPL
# ---------------------------------------------------------------------------

def run_chat(session_id: str) -> None:
    from almond import Almond, AlmondConfig
    from memory_block import MemoryTag, MemoryTier

    from almond import AlmondConfig, LLMProvider
    config = AlmondConfig(session_id=session_id)

    # Only enforce API key check for cloud providers
    if config.provider == LLMProvider.ANTHROPIC and not os.environ.get("ANTHROPIC_API_KEY"):
        print(c(RED, "  Error: ANTHROPIC_API_KEY not set. Export it or switch provider to OLLAMA."))
        sys.exit(1)
    if config.provider == LLMProvider.OPENAI and not os.environ.get("OPENAI_API_KEY"):
        print(c(RED, "  Error: OPENAI_API_KEY not set. Export it or switch provider to OLLAMA."))
        sys.exit(1)

    # Verify Ollama is reachable before starting session
    if config.provider == LLMProvider.OLLAMA:
        import urllib.request
        try:
            urllib.request.urlopen(f"{config.ollama_base_url}/v1/models", timeout=3)
        except Exception:
            print(c(RED, "  Error: LM Studio server not reachable at " + config.ollama_base_url))
            print(c(DIM, "  Start it via LM Studio → Local Server → Start Server"))
            sys.exit(1)

    banner()
    print(c(DIM, f"  Session : {session_id}"))
    print(c(DIM, f"  Provider: {config.provider.value}  |  Model: {config.model}"))
    print(c(DIM, f"  DB      : almond.db\n"))
    print(c(DIM, "  Commands: /dump  /export  /quit\n"))
    print(SEPARATOR)

    with Almond(config) as almond:

        # Seed user profile on fresh sessions
        existing = almond.store.tier_counts()
        if not existing:
            almond.add_memory(
                content="User is building Project Almond — a T-MMU for LLMs. "
                        "Goal: research paper, then personal voice assistant (Jarvis-style), "
                        "then potential product.",
                tag=MemoryTag.USER_PROFILE,
                importance_score=9.5,
                keywords=["almond", "project", "tmu", "paper", "jarvis", "voice"],
                tier=MemoryTier.L1_HOT_CACHE,
            )

        while True:
            try:
                user_input = input(c(CYAN, "\n  You › ") + RESET).strip()
            except (EOFError, KeyboardInterrupt):
                print(c(DIM, "\n  Session ended."))
                break

            if not user_input:
                continue

            # ── Commands ──
            if user_input == "/quit":
                break

            if user_input == "/dump":
                pool = almond.controller.dump_pool()
                print(c(YELLOW, f"\n  Memory Pool ({len(pool)} blocks)\n  {SEPARATOR}"))
                for b in pool:
                    bar_len  = int(b["p_eff"] / 10 * 20)
                    bar      = "█" * bar_len + "░" * (20 - bar_len)
                    print(
                        f"  {b['tier']:<18} {b['tag']:<14} "
                        f"peff={b['p_eff']:<8.4f} [{bar}] "
                        f"{DIM}{b['content_preview'][:40]}{RESET}"
                    )
                continue

            if user_input == "/export":
                path = f"almond_turns_{session_id}.json"
                with open(path, "w") as f:
                    json.dump(almond.export_turn_log(), f, indent=2)
                print(c(GREEN, f"  Turn log exported → {path}"))
                continue

            # ── Normal turn ──
            t0    = time.time()
            reply = almond.chat(user_input)
            ms    = (time.time() - t0) * 1000

            pool_summary = almond.controller.dump_pool()
            l2_count     = sum(1 for b in pool_summary if b["tier"] == "L2_ACTIVE_RAM")
            tokens        = sum(
                len(b["content_preview"]) // 4
                for b in pool_summary
                if b["tier"] in ("L1_HOT_CACHE", "L2_ACTIVE_RAM")
            )

            print(c(GREEN, f"\n  Almond › ") + reply)
            print(c(DIM, f"\n  [{ms:.0f}ms | L2 blocks: {l2_count} | ~{tokens} tokens]"))


# ---------------------------------------------------------------------------
# Mode 3: Dump DB State
# ---------------------------------------------------------------------------

def run_dump() -> None:
    from memory_store import MemoryStore

    if not Path("almond.db").exists():
        print(c(YELLOW, "  No almond.db found. Run 'chat' first."))
        return

    with MemoryStore() as store:
        metrics = store.export_metrics()
        counts  = store.tier_counts()

    print(f"\n{c(BOLD, 'MEMORY POOL DUMP')}\n{SEPARATOR}")
    print(c(DIM, f"  Total blocks: {len(metrics)}"))
    for tier, cnt in counts.items():
        print(c(DIM, f"  {tier}: {cnt}"))
    print()

    for b in metrics:
        # 1. Pad the raw strings FIRST (so the spacing is mathematically perfect)
        tier_padded = f"{b['tier']:<16}"
        tag_padded  = f"{b['tag']:<15}"
        peff_padded = f"{b['p_eff']:<8.4f}"  # pad the number to 8 spaces
    
        # 2. Apply colors SECOND
        tier_colored = c(CYAN, tier_padded)
        peff_colored = c(GREEN, peff_padded)
    
        # 3. Print the pre-formatted variables
        print(
            f"  {tier_colored} {tag_padded} "
            f"peff={peff_colored} "
            f"Δt={b['delta_t_days']:<6.2f}d  "
            f"{c(DIM, b['content_preview'][:50])}"
        )
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Mode 4: Reset
# ---------------------------------------------------------------------------

def run_reset() -> None:
    db = Path("almond.db")
    if db.exists():
        db.unlink()
        print(c(GREEN, "  almond.db deleted. Fresh start on next run."))
    else:
        print(c(YELLOW, "  No almond.db found — already clean."))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Almond — T-MMU CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  test    Run smoke tests (no API key needed)
  chat    Interactive session with live Almond
  dump    Print current memory pool from DB
  reset   Wipe almond.db
        """
    )
    parser.add_argument(
        "mode",
        choices=["test", "chat", "dump", "reset"],
        help="Run mode"
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for chat mode (default: 'default')"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Log verbosity (default: WARNING)"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.mode == "test":
        run_smoke_tests()
    elif args.mode == "chat":
        run_chat(args.session)
    elif args.mode == "dump":
        run_dump()
    elif args.mode == "reset":
        run_reset()


if __name__ == "__main__":
    main()