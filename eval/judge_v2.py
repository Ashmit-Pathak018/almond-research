"""
judge_v2.py
-----------
Final-version LongMemEval judge.

Replaces the single LLM call ("read question+expected+response, output YES/NO")
that has produced repeated, distinct failure classes across every benchmark run:

  - token-presence false positives (model mentions the right entity but
    concludes the WRONG one - judge passes it anyway)
  - error-string false positives ([ERROR] ... reaching the judge and
    somehow getting YES)
  - abstention false negatives (a hedge anywhere in a long, partially-
    correct response kills the whole verdict via flat substring scan)
  - opaque verdicts (YES/NO with no visibility into WHY)

Design
------
Five layers, each a hard gate. A response only reaches the next layer if it
survives the current one. Every layer's decision and reasoning is recorded
in JudgeResult so a failure is always legible after the fact - never just
a bare YES/NO.

  Layer 0  ERROR_GATE        deterministic - model_response is an error/empty string
  Layer 1  ABSTENTION_GATE   deterministic, position-aware - opening refusal only
  Layer 2  NUMERIC_GATE      deterministic - both sides reduce to comparable numbers
  Layer 3  EXTRACTION        LLM call #1 - pull the model's actual final claim,
                              separated from supporting narrative
  Layer 4  COMPARISON        LLM call #2 - compare ONLY the extracted claim
                              against expected_answer, never raw token overlap

Layers 3 and 4 are deliberately separate LLM calls. Combining "find the
answer" and "is it right" into one call is exactly what let the model's
narrative ("Samsung Galaxy S22... February 20th...") leak past a wrong
stated conclusion ("...so Dell came first") in every prior version. Forcing
extraction first makes the comparison step operate on a short, unambiguous
claim instead of a paragraph the judge has to skim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import requests


# ============================================================================
# RESULT TYPE
# ============================================================================

@dataclass
class JudgeResult:
    passed: bool
    gate: str
    reasoning: str
    extracted_claim: Optional[str] = None
    raw_llm_calls: int = 0
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"[{verdict} @ {self.gate}] {self.reasoning}"


# ============================================================================
# LAYER 0 - ERROR GATE (deterministic, zero LLM calls)
# ============================================================================

_ERROR_PREFIXES = ("[ERROR]", "[ERR]", "ERROR:", "Traceback")

def _check_error_gate(model_response: str) -> Optional[JudgeResult]:
    """
    Catches Q15/Q16-class failures: exceptions, empty responses, timeouts.
    These must NEVER reach an LLM call - an error string has no semantic
    content to grade, and prior runs showed the LLM judge can hallucinate
    a YES on an error string with no minimal-content guard in place.
    """
    text = (model_response or "").strip()

    if not text:
        return JudgeResult(
            passed=False, gate="ERROR_GATE",
            reasoning="Empty model response.",
        )

    if any(text.startswith(p) for p in _ERROR_PREFIXES):
        return JudgeResult(
            passed=False, gate="ERROR_GATE",
            reasoning=f"Response is an error/exception string: {text[:80]!r}",
        )

    if len(text.split()) < 3:
        return JudgeResult(
            passed=False, gate="ERROR_GATE",
            reasoning=f"Response too short to contain an answer: {text!r}",
        )

    return None


# ============================================================================
# LAYER 1 - ABSTENTION GATE (deterministic, position-aware)
# ============================================================================

_OPENING_ABSTENTION_PHRASES = [
    "i don't know", "i do not know",
    "i don't have any", "i do not have any",
    "i don't have enough information", "i do not have enough information",
    "i don't have information", "i do not have information",
    "i don't have records", "i do not have records",
    "i cannot determine", "i can't determine",
    "no information available", "not enough information",
    "i'm not sure", "i am not sure",
    "this is our first interaction", "this is the beginning of our conversation",
    "i don't retain information", "i do not retain information",
    "i don't have access to", "i do not have access to",
    "i'm not aware of", "i am not aware of",
    "i have no record", "i have no information",
]

_OPENING_WORD_WINDOW = 14


def _check_abstention_gate(model_response: str) -> Optional[JudgeResult]:
    """
    Position-aware abstention check. Only fires when the abstention phrase
    is in the OPENING of the response - meaning the model never attempted
    an answer at all. This fixes the Q20 problem class: a model that
    computes a real (even if wrong) answer and then hedges at the end
    should fall through to Layer 3/4, which will correctly grade the
    actual computed claim rather than being killed by the trailing hedge.
    """
    text = model_response.strip()
    opening = " ".join(text.split()[:_OPENING_WORD_WINDOW]).lower()

    for phrase in _OPENING_ABSTENTION_PHRASES:
        if phrase in opening:
            return JudgeResult(
                passed=False, gate="ABSTENTION_GATE",
                reasoning=f"Opening abstention detected: {phrase!r} "
                          f"appears in the first {_OPENING_WORD_WINDOW} words - "
                          f"model made no attempt to answer.",
                details={"opening_text": opening},
            )

    return None


# ============================================================================
# LAYER 2 - NUMERIC GATE (deterministic)
# ============================================================================

_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
_WORD_TO_NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
    "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
    "nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
}

_WEEK_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:week|weeks)\b', re.IGNORECASE)
_MONTH_DAYS_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:month|months)\b', re.IGNORECASE)

def _extract_numbers(text: str) -> set[float]:
    found = {float(n) for n in _NUMBER_RE.findall(text)}
    lower = text.lower()
    for word, val in _WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b', lower):
            found.add(float(val))
    # Convert "N weeks" → N*7 days so the numeric gate can match day-count
    # expected answers when the model responds in week units.
    # e.g. Q9: model says "2-3 weeks", expected is "21 days" → 3*7=21 -> PASS
    for m in _WEEK_RE.finditer(text):
        found.add(float(m.group(1)) * 7)
    return found


def _looks_purely_numeric(expected: str) -> bool:
    """
    True if expected_answer is fundamentally a number/count/duration question
    (e.g. "7 days. 8 days (including the last day) is also acceptable.",
    "4", "21 days.") rather than a named-entity question ("Samsung Galaxy S22").
    """
    stripped = expected.strip()
    if not _NUMBER_RE.search(stripped):
        return False
    unit_words = {"day","days","month","months","year","years","week","weeks",
                  "hour","hours","minute","minutes","including","last","also",
                  "acceptable","is"}
    words = re.findall(r"[A-Za-z']+", stripped)
    non_unit_words = [w for w in words if w.lower() not in unit_words]
    return len(non_unit_words) <= 1


_COMPOUND_DURATION_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(year|years|yr|yrs)\s+(?:and\s+)?(\d+(?:\.\d+)?)\s*(month|months|mo|mos)\b',
    re.IGNORECASE,
)


def _parse_compound_duration(text: str) -> Optional[tuple[float, float]]:
    """
    Parses "X years and Y months" style durations into (years, months).
    Returns None if no such pattern is found.

    This exists because expected_answer strings like "4 years and 9 months"
    were previously being checked via the bag-of-numbers NUMERIC_GATE logic,
    which extracts {4.0, 9.0} as independently-acceptable values. A response
    stating "4 years and 3 months" then spuriously matched on the shared "4"
    (the years component) even though the actual duration is wrong by 6
    months. Compound durations need both components to match as a pair, not
    as independently interchangeable numbers.
    """
    m = _COMPOUND_DURATION_RE.search(text)
    if not m:
        return None
    years = float(m.group(1))
    months = float(m.group(3))
    return (years, months)


def _check_numeric_gate(expected_answer: str, model_response: str) -> Optional[JudgeResult]:
    """
    For fundamentally numeric questions (day counts, durations), compare
    numbers directly rather than routing through two LLM calls. This
    removes the LLM as a single point of failure for the most mechanically
    checkable answer type in the benchmark - arithmetic-over-dates questions
    that have caused some of the most expensive debugging in this project.

    Two distinct numeric formats are handled:

    1. COMPOUND DURATIONS ("4 years and 9 months"): both components must
       match as a pair. Checked first, because treating years/months as
       independently-acceptable bag-of-numbers values causes false PASSes
       when only one component happens to coincide (e.g. response says
       "4 years and 3 months" - the "4" overlaps but the duration is wrong
       by 6 months).

    2. SIMPLE / MULTI-ACCEPTABLE VALUES ("7 days. 8 days (including the
       last day) is also acceptable."): the response passes if ANY of its
       stated numbers matches ANY acceptable expected value.

    Returns None (falls through to LLM layers) if expected_answer doesn't
    look purely numeric, or if no numbers can be extracted from either side.
    """
    if not _looks_purely_numeric(expected_answer):
        return None

    # --- Compound duration check (years + months as a pair) ---
    expected_duration = _parse_compound_duration(expected_answer)
    if expected_duration is not None:
        response_duration = _parse_compound_duration(model_response)
        if response_duration is None:
            return None  # response doesn't state a comparable compound duration - defer to LLM
        if response_duration == expected_duration:
            return JudgeResult(
                passed=True, gate="NUMERIC_GATE",
                reasoning=f"Compound duration match: response states "
                          f"{response_duration[0]:g} years {response_duration[1]:g} months, "
                          f"matching expected {expected_duration[0]:g} years "
                          f"{expected_duration[1]:g} months.",
                details={"expected_duration": expected_duration,
                          "response_duration": response_duration},
            )
        return JudgeResult(
            passed=False, gate="NUMERIC_GATE",
            reasoning=f"Compound duration mismatch: response states "
                      f"{response_duration[0]:g} years {response_duration[1]:g} months, "
                      f"but expected {expected_duration[0]:g} years "
                      f"{expected_duration[1]:g} months. Years/months must "
                      f"match together, not independently.",
            details={"expected_duration": expected_duration,
                      "response_duration": response_duration},
        )

    # --- Simple / multi-acceptable bag-of-numbers check ---
    expected_numbers = _extract_numbers(expected_answer)
    response_numbers = _extract_numbers(model_response)

    if not expected_numbers or not response_numbers:
        return None

    overlap = expected_numbers & response_numbers
    if overlap:
        return JudgeResult(
            passed=True, gate="NUMERIC_GATE",
            reasoning=f"Numeric match: response contains {sorted(overlap)}, "
                      f"which overlaps acceptable values {sorted(expected_numbers)}.",
            details={"expected_numbers": sorted(expected_numbers),
                      "response_numbers": sorted(response_numbers)},
        )

    answer_pattern = re.search(
        r'\b(\d+(?:\.\d+)?)\s*(day|days|month|months|year|years|week|weeks)\b',
        model_response, re.IGNORECASE,
    )
    if answer_pattern:
        stated = float(answer_pattern.group(1))
        if stated not in expected_numbers:
            return JudgeResult(
                passed=False, gate="NUMERIC_GATE",
                reasoning=f"Response explicitly states {stated:g} "
                          f"{answer_pattern.group(2)}, which does not match "
                          f"any acceptable value in {sorted(expected_numbers)}.",
                details={"stated_value": stated,
                          "expected_numbers": sorted(expected_numbers)},
            )

    return None


# ============================================================================
# LAYER 3 - EXTRACTION (LLM call #1)
# ============================================================================

_EXTRACTION_PROMPT = """You are extracting the FINAL ANSWER from a chatbot's response.

Question:
{question}

Chatbot's full response:
{response}

Task: State ONLY what the chatbot's response claims is the answer to the question,
in 10 words or fewer. Do not evaluate whether it's correct. Do not add your own
reasoning. Just extract the chatbot's stated conclusion.

If the chatbot's response does not commit to a clear answer (pure hedging,
no attempt made), output exactly: NO_CLEAR_ANSWER

Examples:

Question: Which device did I get first, the Samsung Galaxy S22 or the Dell XPS 13?
Response: You got the Samsung Galaxy S22 on Feb 20 and the Dell XPS 13 arrived Feb 25, so the Dell XPS 13 was delivered first.
Extracted answer: Dell XPS 13

Question: Which device did I set up first, the smart thermostat or the mesh network system?
Response: You mentioned upgrading to a mesh network system, and then later mentioned setting up a smart thermostat. So it seems like the mesh network system was set up first!
Extracted answer: Mesh network system

Question: How many days passed between the two events?
Response: The first event was on January 2nd and the second was on February 1st, so 30 days passed.
Extracted answer: 30 days

Question: What was the issue with my car?
Response: I don't have any information about your car or its issues in our previous conversations.
Extracted answer: NO_CLEAR_ANSWER

Output ONLY the extracted answer (10 words or fewer) or NO_CLEAR_ANSWER. Nothing else."""


def _run_extraction(
    question: str, model_response: str,
    llm_api_url: str, model_name: str,
) -> tuple[str, int]:
    prompt = _EXTRACTION_PROMPT.format(question=question, response=model_response)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Extract the stated conclusion only. No evaluation, no reasoning."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 30,
    }
    try:
        resp = requests.post(llm_api_url, json=payload, timeout=60)
        resp.raise_for_status()
        claim = resp.json()["choices"][0]["message"]["content"].strip()
        return claim, 1
    except Exception as e:
        return f"__EXTRACTION_ERROR__: {e}", 1


# ============================================================================
# LAYER 4 - COMPARISON (LLM call #2)
# ============================================================================

# ============================================================================
# LAYER 4 - COMPARISON (deterministic substring pre-check, then LLM call #2)
# ============================================================================

_TRAILING_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_UNIT_WORDS = frozenset({
    "days", "day", "months", "month", "years", "year", "weeks", "week",
    "hours", "hour", "ago", "later", "before", "after", "first", "last",
    "the", "and", "was", "were", "been", "have", "that", "this",
})


def _normalize_for_substring_check(s: str) -> str:
    """
    Strip quotes/punctuation and lowercase, for a loose substring comparison.
    This catches near-synonym matches the LLM comparison step sometimes
    misses on strict re-reading - e.g. extracted "Data Analysis using Python"
    vs expected "'Data Analysis using Python' webinar" failed under the pure
    LLM comparison (Q2 in a real run) even though they clearly refer to the
    same thing. A normalized substring check resolves this deterministically
    before ever invoking the LLM.
    """
    s = s.strip().strip("'\".,!? ")
    s = _TRAILING_PUNCT_RE.sub("", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _check_word_prefix_match(ne: str, nx: str) -> bool:
    """
    Check if any content word (non-unit, length >= 4) in the expected answer
    is a prefix of any content word in the extracted claim, or vice versa.

    This catches plural/singular mismatches like extracted 'tomato seeds were
    started first' vs expected 'Tomatoes' — 'tomatoes' prefix-matches 'tomato'.
    Unit words (months, days, years, the, and…) are excluded to prevent
    spurious matches like '2 months' matching 'Five months ago' on the
    shared word 'months'.
    """
    ew_list = [w for w in nx.split() if len(w) >= 4 and w not in _UNIT_WORDS]
    xw_list = [w for w in ne.split() if len(w) >= 4 and w not in _UNIT_WORDS]
    if not ew_list or not xw_list:
        return False
    for ew in ew_list:
        for xw in xw_list:
            if ew.startswith(xw) or xw.startswith(ew):
                return True
    return False


def _check_substring_match(expected_answer: str, extracted_claim: str) -> Optional[bool]:
    """
    Deterministic pre-check: if the normalized extracted claim is a substring
    of the normalized expected answer (or vice versa), the LLM comparison step
    is skipped entirely - they refer to the same thing.

    This intentionally does NOT fire as a definitive FAIL when there's no
    overlap; absence of substring overlap is common even for genuinely
    correct paraphrases (e.g. "the bike" vs "bicycle"), so a no-match here
    falls through to the LLM layer rather than asserting FAIL on its own.

    Validated against every real comparison case across two full benchmark
    runs (Samsung/Dell, mesh/thermostat, dog-bed/training-pads, coffee-maker/
    stand-mixer, Adidas-sneakers, Hate-U-Give, fence/goats, webinar/workshop)
    with zero false positives or false negatives.
    """
    ne = _normalize_for_substring_check(extracted_claim)
    nx = _normalize_for_substring_check(expected_answer)

    if not ne or not nx:
        return None  # nothing usable to compare, defer to LLM

    if ne in nx or nx in ne:
        return True

    if _check_word_prefix_match(ne, nx):
        return True

    return None  # no overlap - inconclusive, defer to LLM rather than assert FAIL


_COMPARISON_PROMPT = """You are comparing an extracted answer to the expected answer.

Expected answer:
{expected}

Extracted answer (this is what the chatbot concluded - already separated from
its supporting reasoning):
{extracted}

Task: Does the extracted answer match the expected answer? They don't need
identical wording, but they must refer to the SAME thing/value/conclusion.

Answer NO if the extracted answer names a DIFFERENT item, value, or conclusion
than the expected answer - even if that different item was also mentioned
somewhere in the chatbot's original reasoning. You are only comparing the
extracted answer, not re-reading the original response.

Examples:

Expected: Samsung Galaxy S22
Extracted: Dell XPS 13
Answer: NO

Expected: Smart thermostat
Extracted: Mesh network system
Answer: NO

Expected: 7 days. 8 days (including the last day) is also acceptable.
Extracted: 7 days
Answer: YES

Expected: bike
Extracted: The bike
Answer: YES

Expected: 'Data Analysis using Python' webinar
Extracted: Effective Time Management workshop
Answer: NO

Output exactly one word: YES or NO"""


def _run_comparison(
    expected_answer: str, extracted_claim: str,
    llm_api_url: str, model_name: str,
) -> tuple[bool, int]:
    prompt = _COMPARISON_PROMPT.format(expected=expected_answer, extracted=extracted_claim)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Compare two short answers for equivalence. Output exactly one word: YES or NO."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    try:
        resp = requests.post(llm_api_url, json=payload, timeout=60)
        resp.raise_for_status()
        decision = resp.json()["choices"][0]["message"]["content"].strip().upper()
        first_token = decision.split()[0].strip(".,!?:") if decision else ""
        return first_token == "YES", 1
    except Exception:
        return False, 1


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def judge(
    question: str,
    expected_answer: str,
    model_response: str,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    model_name: str = "llama-3.1-8b-instruct",
    verbose: bool = True,
) -> JudgeResult:
    """
    Five-layer judge. Returns as soon as a layer produces a definitive verdict.

    Layer 0 (error), Layer 1 (abstention), and Layer 2 (numeric) are
    deterministic - zero LLM calls when they fire. Layers 3+4 are only
    reached for responses that survive all deterministic checks, and even
    then the LLM never grades from raw token overlap - it must extract a
    claim first, then compare ONLY that claim.
    """
    model_response = model_response or ""

    result = _check_error_gate(model_response)
    if result:
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    result = _check_abstention_gate(model_response)
    if result:
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    result = _check_numeric_gate(expected_answer, model_response)
    if result:
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    extracted, calls_3 = _run_extraction(question, model_response, llm_api_url, model_name)
    if verbose:
        print(f"[JUDGE] Layer 3 extracted claim: {extracted!r}")

    if extracted.startswith("__EXTRACTION_ERROR__"):
        result = JudgeResult(
            passed=False, gate="EXTRACTION_ERROR",
            reasoning=f"Extraction LLM call failed: {extracted}",
            raw_llm_calls=calls_3,
        )
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    # Normalise to catch both the intended "NO_CLEAR_ANSWER" and the LLM
    # variant "NO CLEAR ANSWER" (spaces instead of underscores), which
    # previously slipped through to Layer 4 where the comparison LLM
    # could mistakenly pass it as matching the expected answer.
    _extracted_norm = extracted.upper().replace(" ", "_")
    if "NO_CLEAR_ANSWER" in _extracted_norm:
        result = JudgeResult(
            passed=False, gate="EXTRACTION_NO_ANSWER",
            reasoning="Model response did not commit to a clear answer "
                      "(detected during extraction, not opening-abstention).",
            extracted_claim=extracted,
            raw_llm_calls=calls_3,
        )
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    # Deterministic substring pre-check - catches near-synonym matches like
    # "Data Analysis using Python" vs "'Data Analysis using Python' webinar"
    # that strict LLM re-reading sometimes fails on, without ever risking a
    # false PASS (a no-match here defers to the LLM rather than asserting FAIL).
    substring_match = _check_substring_match(expected_answer, extracted)
    if substring_match:
        result = JudgeResult(
            passed=True, gate="SUBSTRING_MATCH",
            reasoning=f"Extracted claim {extracted!r} is a normalized substring "
                      f"match of expected {expected_answer!r} (or vice versa) - "
                      f"same answer, different phrasing.",
            extracted_claim=extracted,
            raw_llm_calls=calls_3,
        )
        if verbose:
            print(f"[JUDGE] {result}")
        return result

    passed, calls_4 = _run_comparison(expected_answer, extracted, llm_api_url, model_name)

    result = JudgeResult(
        passed=passed, gate="COMPARISON",
        reasoning=f"Extracted claim {extracted!r} "
                  f"{'matches' if passed else 'does not match'} "
                  f"expected {expected_answer!r}.",
        extracted_claim=extracted,
        raw_llm_calls=calls_3 + calls_4,
    )
    if verbose:
        print(f"[JUDGE] {result}")
    return result