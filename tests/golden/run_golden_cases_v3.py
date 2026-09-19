"""
Project Almond V3 — 10 Executable Golden Cases Runner
Validates all 10 architectural invariants and retrieval regression cases against V3 Retrieval Engine.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.golden.golden_cases import GOLDEN_CASES, GoldenTestCase
from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_store import MemoryStore
from core.knowledge.entity_store import EntityStore
from core.knowledge.timeline_store import TimelineStore, EVENT_TYPE_POINT
from core.knowledge.fact_store import FactStore
from core.retrieval.contracts import RetrievalQuery
from core.retrieval.retrieval_engine import RetrievalEngine


def setup_knowledge_for_case(
    case: GoldenTestCase,
    store: MemoryStore,
    entity_store: EntityStore,
    timeline_store: TimelineStore,
    fact_store: FactStore
):
    ns = "default"
    if case.case_id == "CASE-001":
        for p in case.preconditions:
            if p.event_time:
                title = p.content[:25]
                ev = timeline_store.create_event(ns, title, EVENT_TYPE_POINT, p.event_time)
                timeline_store.link_memory(p.id, ev.event_id)

    elif case.case_id == "CASE-002":
        ent = entity_store.create_entity(ns, "Robert Vance", "PERSON")
        entity_store.add_alias(ent.entity_id, "Bob")
        for p in case.preconditions:
            if "Robert Vance" in p.content:
                entity_store.link_memory(p.id, ent.entity_id, 0.95)

    elif case.case_id == "CASE-003":
        for p in case.preconditions:
            if p.event_time:
                title = p.content[:25]
                ev = timeline_store.create_event(ns, title, EVENT_TYPE_POINT, p.event_time)
                timeline_store.link_memory(p.id, ev.event_id)

    elif case.case_id == "CASE-005":
        f_old = fact_store.create_fact(ns, "mem_city_old", "user", "lives_in", "Seattle")
        f_new = fact_store.create_fact(ns, "mem_city_new", "user", "lives_in", "Boston")
        fact_store.supersede_fact(f_old.fact_id, f_new.fact_id)

    elif case.case_id == "CASE-007":
        # Delete memory block to verify cascading deletion integrity
        store.delete("mem_secret_project")

    elif case.case_id == "CASE-008":
        # Wipe derived Chroma collection and test rebuild_indexes() from SQLite ground truth
        store.rebuild_indexes()

    elif case.case_id in ["CASE-009", "CASE-010"]:
        for p in case.preconditions:
            if p.event_time:
                title = p.content[:25]
                ev = timeline_store.create_event(ns, title, EVENT_TYPE_POINT, p.event_time)
                timeline_store.link_memory(p.id, ev.event_id)


def run_golden_cases():
    print("==================================================")
    print("RUNNING ALL 10 GOLDEN CASES AGAINST V3 RETRIEVAL ENGINE")
    print("==================================================")
    passed_count = 0
    total_count = len(GOLDEN_CASES)

    for case in GOLDEN_CASES:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "golden_test.db")
            chroma_path = os.path.join(tmpdir, "chroma_golden")
            store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
            entity_store = EntityStore(store._conn)
            timeline_store = TimelineStore(store._conn)
            fact_store = FactStore(store._conn)
            engine = RetrievalEngine(store, entity_store, timeline_store)

            for p in case.preconditions:
                try:
                    tag = MemoryTag(p.tag)
                except ValueError:
                    tag = MemoryTag.USER_PROFILE

                mb = MemoryBlock(
                    id=p.id,
                    content=p.content,
                    tag=tag,
                    event_time=p.event_time,
                    created_at=p.created_at or time.time(),
                    last_accessed_at=p.last_accessed_at or time.time(),
                    importance_score=p.importance_score,
                )
                if case.case_id == "CASE-005" and p.id == "mem_city_old":
                    mb.tier = MemoryTier.L4_ARCHIVE
                    mb.summary = "Lived in Seattle."
                store.save(mb)

            setup_knowledge_for_case(case, store, entity_store, timeline_store, fact_store)

            q = RetrievalQuery(
                query_text=case.query,
                namespace_id="default",
                reference_time=case.reference_time,
                top_k=10
            )
            res = engine.query(q)
            matched_ids = [c.memory_id for c in res.candidates]
            
            success = True
            error_msg = ""
            if case.expected_abstention:
                if not res.is_abstention:
                    success = False
                    error_msg = f"Expected abstention, but got candidates: {matched_ids}"
            else:
                if res.is_abstention:
                    success = False
                    error_msg = "Expected matches, but system abstained."
                else:
                    if case.expected_order_strict:
                        indices = []
                        for exp_id in case.expected_retrieved_ids:
                            if exp_id in matched_ids:
                                indices.append(matched_ids.index(exp_id))
                            else:
                                indices.append(-1)
                        if -1 in indices:
                            success = False
                            error_msg = f"Missing expected IDs. Expected {case.expected_retrieved_ids}, got {matched_ids}"
                        elif indices != sorted(indices):
                            success = False
                            error_msg = f"Incorrect strict ordering. Expected {case.expected_retrieved_ids}, got {matched_ids}"
                    else:
                        for exp_id in case.expected_retrieved_ids:
                            if exp_id not in matched_ids:
                                success = False
                                error_msg = f"Missing expected ID '{exp_id}'. Got {matched_ids}"
                                break

            # Specific case invariant validations
            if success and case.case_id == "CASE-006":
                # Verify memory block remains in expected lifecycle tier
                retrieved_cand = next((c for c in res.candidates if c.memory_id == "mem_jan_task"), None)
                if not retrieved_cand or retrieved_cand.tier != case.expected_lifecycle_tier:
                    success = False
                    error_msg = f"Expected tier {case.expected_lifecycle_tier}, got {getattr(retrieved_cand, 'tier', None)}"

            if success and case.case_id == "CASE-007":
                # Verify SQLite relational links and primary row are purged
                row = store._conn.execute("SELECT id FROM memory_blocks WHERE id = 'mem_secret_project'").fetchone()
                if row is not None:
                    success = False
                    error_msg = "Primary SQLite record not deleted"
                count = store._collection.count()
                if count != 0:
                    success = False
                    error_msg = f"Chroma vector not purged (count={count})"

            if success:
                print(f"PASS {case.case_id}: {case.title}")
                passed_count += 1
            else:
                print(f"FAIL {case.case_id}: {case.title}")
                print(f"   Reason: {error_msg}")
                print(f"   Trace: {res.trace.to_dict()}")
                print(f"   Candidates: {[c.memory_id for c in res.candidates]}")
                print(f"   Collection count: {store._collection.count() if hasattr(store, '_collection') and store._collection else None}")
            store.close()
            
    print("==================================================")
    print(f"RESULTS: {passed_count}/{total_count} Golden Cases Passed")
    print("==================================================")
    if passed_count != total_count:
        sys.exit(1)


if __name__ == "__main__":
    run_golden_cases()
