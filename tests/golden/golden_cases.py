"""
Project Almond V3 — 10 Executable Golden Case Specifications
Authoritative test definitions for architectural invariants and retrieval regression prevention.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class GoldenPreconditionMemory:
    id: str
    content: str
    tag: str
    event_time: Optional[float] = None
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    importance_score: float = 5.0
    access_count: int = 1
    session_id: str = "default_session"

@dataclass
class GoldenTestCase:
    case_id: str
    title: str
    description: str
    query: str
    reference_time: float
    preconditions: List[GoldenPreconditionMemory]
    expected_retrieved_ids: List[str]
    expected_order_strict: bool
    expected_abstention: bool = False
    expected_lifecycle_tier: Optional[str] = None
    expected_facts_state: Optional[Dict[str, str]] = None
    verification_notes: str = ""

GOLDEN_CASES: List[GoldenTestCase] = [
    # -------------------------------------------------------------------------
    # CASE-001: Strict Temporal Chronology
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-001",
        title="Strict Temporal Chronology",
        description="Verify retrieval selects events prior to a temporal boundary in strict reverse-chronological order.",
        query="What did I do before switching to PostgreSQL?",
        reference_time=1709294400.0, # 2024-03-01 12:00:00 UTC
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_mysql_setup",
                content="I set up our primary MySQL database with master-replica replication.",
                tag="PROJECT_FACT",
                event_time=1704888000.0, # 2024-01-10 12:00:00 UTC
                created_at=1704888000.0,
                last_accessed_at=1704888000.0
            ),
            GoldenPreconditionMemory(
                id="mem_postgres_plan",
                content="I drafted the architectural migration plan to switch from MySQL to PostgreSQL due to JSONB requirements.",
                tag="PROJECT_FACT",
                event_time=1707998400.0, # 2024-02-15 12:00:00 UTC
                created_at=1707998400.0,
                last_accessed_at=1707998400.0
            ),
            GoldenPreconditionMemory(
                id="mem_postgres_live",
                content="Completed the database cutover: PostgreSQL is now live in production.",
                tag="PROJECT_FACT",
                event_time=1709294400.0, # 2024-03-01 12:00:00 UTC
                created_at=1709294400.0,
                last_accessed_at=1709294400.0
            )
        ],
        expected_retrieved_ids=["mem_postgres_plan", "mem_mysql_setup"],
        expected_order_strict=True,
        verification_notes="Must filter out mem_postgres_live and return plan before initial setup."
    ),

    # -------------------------------------------------------------------------
    # CASE-002: Entity Alias Resolution
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-002",
        title="Entity Alias Resolution",
        description="Verify entity aliases (e.g. 'Bob' -> 'Robert Vance') retrieve canonical entity facts.",
        query="How is Bob doing on the client infrastructure project?",
        reference_time=1710000000.0,
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_robert_vance",
                content="Robert Vance delivered the client infrastructure security audit report on time.",
                tag="PROJECT_FACT",
                event_time=1709900000.0,
                created_at=1709900000.0,
                last_accessed_at=1709900000.0
            ),
            GoldenPreconditionMemory(
                id="mem_unrelated_bob",
                content="Watched a funny video of Bob the builder.",
                tag="SMALL_TALK",
                event_time=1709900000.0,
                created_at=1709900000.0,
                last_accessed_at=1709900000.0
            )
        ],
        expected_retrieved_ids=["mem_robert_vance"],
        expected_order_strict=False,
        verification_notes="Entity retriever must link 'Bob' to 'Robert Vance' in project context."
    ),

    # -------------------------------------------------------------------------
    # CASE-003: Multi-Point Comparison
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-003",
        title="Multi-Point Comparative Reasoning",
        description="Verify query spanning distinct planning and execution intervals retrieves both periods.",
        query="Compare what I planned for Q1 vs. what was actually finished.",
        reference_time=1712000000.0, # Early April 2024
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_q1_plan",
                content="Q1 roadmap goals: Ship OAuth2 login, add dark mode UI, and reduce latency by 30%.",
                tag="TASK",
                event_time=1704200000.0, # Early Jan 2024
                created_at=1704200000.0,
                last_accessed_at=1704200000.0
            ),
            GoldenPreconditionMemory(
                id="mem_q1_review",
                content="Q1 retrospective: We successfully shipped OAuth2 and dark mode, but latency reduction was delayed to Q2.",
                tag="PROJECT_FACT",
                event_time=1711800000.0, # Late March 2024
                created_at=1711800000.0,
                last_accessed_at=1711800000.0
            )
        ],
        expected_retrieved_ids=["mem_q1_plan", "mem_q1_review"],
        expected_order_strict=False,
        verification_notes="Both planning and retrospective memories must be retrieved into comparison clusters."
    ),

    # -------------------------------------------------------------------------
    # CASE-004: Abstention on Absent Memory
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-004",
        title="Abstention on Absent Memory",
        description="Verify system abstains when queried about facts not present in memory store.",
        query="What did I say my favorite musical artist was?",
        reference_time=1710000000.0,
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_fav_food",
                content="My favorite food is spicy vegetarian ramen.",
                tag="USER_PROFILE",
                created_at=1709000000.0,
                last_accessed_at=1709000000.0
            )
        ],
        expected_retrieved_ids=[],
        expected_order_strict=False,
        expected_abstention=True,
        verification_notes="Must return empty/low-confidence retrieval with abstention flag rather than forcing false match."
    ),

    # -------------------------------------------------------------------------
    # CASE-005: Contradiction & Superseded Truth
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-005",
        title="Contradiction and Superseded Truth",
        description="Verify newer fact supersedes older contradictory fact without silent corruption.",
        query="Where do I live currently?",
        reference_time=1715000000.0, # May 2024
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_city_old",
                content="I live in Seattle, Washington.",
                tag="USER_PROFILE",
                event_time=1640995200.0, # 2022
                created_at=1640995200.0,
                last_accessed_at=1640995200.0
            ),
            GoldenPreconditionMemory(
                id="mem_city_new",
                content="I recently relocated and now live in Boston, Massachusetts.",
                tag="USER_PROFILE",
                event_time=1714521600.0, # May 2024
                created_at=1714521600.0,
                last_accessed_at=1714521600.0
            )
        ],
        expected_retrieved_ids=["mem_city_new"],
        expected_order_strict=True,
        expected_facts_state={"mem_city_old": "SUPERSEDED", "mem_city_new": "VERIFIED"},
        verification_notes="Old fact must be marked superseded and Boston returned as current ground truth."
    ),

    # -------------------------------------------------------------------------
    # CASE-006: Time-Travel Decay Under Simulated Clock
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-006",
        title="Time-Travel Decay Under Simulated Clock",
        description="Verify memory decays strictly according to reference_time, not physical wall-clock.",
        query="What was my daily routine task in January?",
        reference_time=1705320000.0, # Jan 15 2024 (14 days after creation)
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_jan_task",
                content="Daily task: check server uptime at 9am every morning.",
                tag="TASK",
                event_time=1704067200.0, # Jan 1 2024
                created_at=1704067200.0,
                last_accessed_at=1704067200.0,
                importance_score=5.0
            )
        ],
        expected_retrieved_ids=["mem_jan_task"],
        expected_order_strict=False,
        expected_lifecycle_tier="L2_ACTIVE_RAM",
        verification_notes="At reference_time Jan 15, Delta_t is 14 days; memory remains in L2 active tier."
    ),

    # -------------------------------------------------------------------------
    # CASE-007: Cascading Deletion Integrity
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-007",
        title="Cascading Deletion Integrity",
        description="Verify deleting a memory purges all relational links, entity links, facts, and vectors.",
        query="Tell me about the secret project.",
        reference_time=1710000000.0,
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_secret_project",
                content="Project Obsidian is a stealth quantum computing initiative.",
                tag="PROJECT_FACT",
                created_at=1709000000.0,
                last_accessed_at=1709000000.0
            )
        ],
        expected_retrieved_ids=[], # After deletion
        expected_order_strict=False,
        expected_abstention=True,
        verification_notes="After calling delete_memory('mem_secret_project'), 0 rows and 0 vectors remain."
    ),

    # -------------------------------------------------------------------------
    # CASE-008: Cold Recovery Rebuild
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-008",
        title="Cold Recovery Rebuild",
        description="Verify wiping derived Chroma vector index and rebuilding from SQLite restores 100% search.",
        query="What framework do we use for web APIs?",
        reference_time=1710000000.0,
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_fastapi_fact",
                content="Our backend microservices are built exclusively using FastAPI and Python 3.10+.",
                tag="PROJECT_FACT",
                created_at=1709000000.0,
                last_accessed_at=1709000000.0
            )
        ],
        expected_retrieved_ids=["mem_fastapi_fact"],
        expected_order_strict=False,
        verification_notes="Must succeed after complete drop and rebuild_indexes() of vector store."
    ),

    # -------------------------------------------------------------------------
    # CASE-009: Out-of-Order Ingestion Replay
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-009",
        title="Out-of-Order Ingestion Replay",
        description="Verify memories ingested in reverse chronological order maintain correct timeline order.",
        query="List our deployment history in chronological order.",
        reference_time=1715000000.0,
        preconditions=[
            # Ingested in reverse: May event first, then March, then January
            GoldenPreconditionMemory(
                id="mem_deploy_v3",
                content="Deployed v1.2 release with bug fixes.",
                tag="EPISODIC",
                event_time=1714521600.0, # May 1 2024
                created_at=1714521600.0
            ),
            GoldenPreconditionMemory(
                id="mem_deploy_v2",
                content="Deployed v1.1 release with dark mode.",
                tag="EPISODIC",
                event_time=1709251200.0, # March 1 2024
                created_at=1709251200.0
            ),
            GoldenPreconditionMemory(
                id="mem_deploy_v1",
                content="Initial launch of v1.0 in production.",
                tag="EPISODIC",
                event_time=1704067200.0, # January 1 2024
                created_at=1704067200.0
            )
        ],
        expected_retrieved_ids=["mem_deploy_v1", "mem_deploy_v2", "mem_deploy_v3"],
        expected_order_strict=True,
        verification_notes="Timeline must order strictly by event_time (Jan -> Mar -> May), not ingestion order."
    ),

    # -------------------------------------------------------------------------
    # CASE-010: Multi-Session Recall & Context Window Assembly
    # -------------------------------------------------------------------------
    GoldenTestCase(
        case_id="CASE-010",
        title="Multi-Session Recall & Context Assembly",
        description="Verify context assembler draws and orders relevant evidence across multiple sessions.",
        query="Synthesize what we discussed across sessions regarding the database architecture.",
        reference_time=1715000000.0,
        preconditions=[
            GoldenPreconditionMemory(
                id="mem_sess1_db",
                content="Session 1: Identified connection pool exhaustion under high traffic load.",
                tag="PROJECT_FACT",
                session_id="session_alpha",
                created_at=1708000000.0
            ),
            GoldenPreconditionMemory(
                id="mem_sess2_db",
                content="Session 2: Tested PgBouncer and verified it eliminated connection spikes.",
                tag="PROJECT_FACT",
                session_id="session_beta",
                created_at=1709000000.0
            ),
            GoldenPreconditionMemory(
                id="mem_sess3_db",
                content="Session 3: Approved PgBouncer deployment to production cluster.",
                tag="PROJECT_FACT",
                session_id="session_gamma",
                created_at=1710000000.0
            )
        ],
        expected_retrieved_ids=["mem_sess1_db", "mem_sess2_db", "mem_sess3_db"],
        expected_order_strict=False,
        verification_notes="Context assembler must integrate evidence from alpha, beta, and gamma sessions."
    )
]
