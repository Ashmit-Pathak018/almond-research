"""
Test harness for Almond Golden Cases.
Verifies structure, completeness, and executable invariants of CASE-001 through CASE-010.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tests.golden.golden_cases import GOLDEN_CASES, GoldenTestCase

def test_golden_cases_completeness():
    """Ensure exactly 10 golden cases are defined and properly numbered."""
    assert len(GOLDEN_CASES) == 10
    expected_ids = [f"CASE-{i:03d}" for i in range(1, 11)]
    actual_ids = [c.case_id for c in GOLDEN_CASES]
    assert actual_ids == expected_ids

def test_golden_cases_invariants():
    """Ensure every golden case defines non-empty query, reference_time, and preconditions (except abstention/deletion)."""
    for case in GOLDEN_CASES:
        assert case.query.strip(), f"{case.case_id} has empty query"
        assert case.reference_time > 0, f"{case.case_id} has invalid reference_time"
        assert len(case.preconditions) > 0, f"{case.case_id} has no preconditions"
        for p in case.preconditions:
            assert p.id.strip(), f"Precondition in {case.case_id} missing ID"
            assert p.content.strip(), f"Precondition {p.id} has empty content"
            assert p.tag.strip(), f"Precondition {p.id} missing tag"

if __name__ == "__main__":
    test_golden_cases_completeness()
    test_golden_cases_invariants()
    print("All Golden Case specifications validated successfully.")
