import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.judge_v2 import _extract_numbers, _check_numeric_gate, _looks_purely_numeric, _check_substring_match, _check_word_prefix_match, _check_abstention_gate, _normalize_for_substring_check

def test_int_expected_answer_does_not_crash():
    """Fix 1: str(expected_answer) prevents AttributeError on int answers."""
    # These should not raise - they test the deterministic gates only
    result = _looks_purely_numeric("3")
    assert result == True
    result = _looks_purely_numeric("21 days")
    assert result == True
    print("PASS: test_int_expected_answer_does_not_crash")

def test_comma_thousands_parsed_as_single_number():
    """Fix 2b: $5,850 should parse as [5850.0], not [5.0, 850.0]."""
    nums = _extract_numbers("$5,850")
    assert 5850.0 in nums, f"Expected 5850.0 in {nums}"
    assert 5.0 not in nums, f"5.0 should NOT be in {nums}"
    assert 850.0 not in nums, f"850.0 should NOT be in {nums}"
    print("PASS: test_comma_thousands_parsed_as_single_number")

def test_comma_thousands_various():
    """Fix 2b: Various comma-grouped number formats."""
    assert 1000.0 in _extract_numbers("$1,000")
    assert 1234567.0 in _extract_numbers("1,234,567")
    assert 5850.0 in _extract_numbers("total of $5,850")
    # Plain numbers still work
    assert 42.0 in _extract_numbers("42 days")
    assert 3.5 in _extract_numbers("3.5 hours")
    print("PASS: test_comma_thousands_various")

def test_substring_match_rejects_short_fragments():
    """Fix 3: Short fragments should not match long strings."""
    # "1" should not match inside "10"
    result = _check_substring_match("10 days", "1")
    assert result is None, f"Expected None (defer to LLM), got {result}"
    # "no" should not match inside "innovative option"
    result = _check_substring_match("innovative option", "no")
    assert result is None, f"Expected None, got {result}"
    print("PASS: test_substring_match_rejects_short_fragments")

def test_substring_match_allows_genuine_matches():
    """Fix 3: Genuine near-matches should still pass."""
    result = _check_substring_match("Data Analysis using Python", "Data Analysis using Python webinar")
    assert result == True, f"Expected True, got {result}"
    print("PASS: test_substring_match_allows_genuine_matches")

def test_word_prefix_rejects_generic_nouns():
    """Fix 3b: Generic words like 'trip' should not singlehandedly trigger a match."""
    result = _check_word_prefix_match(
        _normalize_for_substring_check("the european trip was more recent"),
        _normalize_for_substring_check("the solo trip to thailand")
    )
    assert result == False, f"Expected False (generic 'trip' should not match), got {result}"
    print("PASS: test_word_prefix_rejects_generic_nouns")

def test_word_prefix_allows_specific_matches():
    """Fix 3b: Specific words should still match their variants."""
    result = _check_word_prefix_match(
        _normalize_for_substring_check("tomato seeds were started first"),
        _normalize_for_substring_check("tomatoes")
    )
    # This should match because 'tomato' prefix-matches 'tomatoes'
    # (both are specific nouns, not generic stopwords)
    assert result == True, f"Expected True, got {result}"
    print("PASS: test_word_prefix_allows_specific_matches")

def test_abstention_gate_detects_refusal():
    """Abstention gate catches common refusal patterns."""
    result = _check_abstention_gate("I don't have enough information to answer that question.")
    assert result is not None and result.passed == False
    result = _check_abstention_gate("Based on your memories, the answer is Paris.")
    assert result is None  # not an abstention
    print("PASS: test_abstention_gate_detects_refusal")

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []
    tests = [
        test_int_expected_answer_does_not_crash,
        test_comma_thousands_parsed_as_single_number,
        test_comma_thousands_various,
        test_substring_match_rejects_short_fragments,
        test_substring_match_allows_genuine_matches,
        test_word_prefix_rejects_generic_nouns,
        test_word_prefix_allows_specific_matches,
        test_abstention_gate_detects_refusal,
    ]
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"FAIL: {test.__name__}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    sys.exit(0 if failed == 0 else 1)
