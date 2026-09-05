"""
Robust numeric answer extraction for GSM8K-style model responses.

GSM8K reference answers end with "#### <number>". Model responses (free-
form text) express the final answer in varied natural-language forms:
    "The answer is 42."
    "Therefore, the final answer is 42."
    "#### 42"
    "= 42"
    "42 dollars"

This module extracts a normalized numeric string from either the
reference format or free-form model text, and compares two normalized
numbers for equality. Deliberately deterministic — no LLM judge.
"""

import re


# Ordered from most to least specific. Matched against the response with
# re.IGNORECASE. The first pattern that matches wins.
_ANSWER_PHRASE_PATTERNS = [
    r"####\s*([\-\$]?[\d,]*\.?\d+%?)",
    r"final answer is[:\s]*([\-\$]?[\d,]*\.?\d+%?)",
    r"the answer is[:\s]*([\-\$]?[\d,]*\.?\d+%?)",
    r"answer:\s*([\-\$]?[\d,]*\.?\d+%?)",
]

# Fallback: last number-looking token anywhere in the text.
_ANY_NUMBER_PATTERN = r"[\-\$]?[\d,]*\.?\d+%?"

# Base models (no instruction tuning, no learned EOS-on-answer behavior)
# routinely keep generating past their answer to the asked question and
# hallucinate an entirely new, unrelated Q&A pair. Truncating at the
# first such marker before searching for the answer prevents picking up
# the answer to a self-invented follow-up question instead of the one
# actually asked. This only affects extraction — the full raw response
# is still preserved untouched in the saved record.
_CONTINUATION_MARKERS = [
    r"\n\s*\[Question\]",
    r"\n\s*Question\s*:",
    r"\n\s*Q\s*:",
]
_CONTINUATION_PATTERN = re.compile("|".join(_CONTINUATION_MARKERS), flags=re.IGNORECASE)


def _truncate_at_continuation(response: str) -> str:
    """Cut off text at the first sign the model started a new, unrelated
    question rather than continuing to answer the one it was asked."""
    match = _CONTINUATION_PATTERN.search(response)
    if match:
        return response[: match.start()]
    return response


def normalize_number(raw: str) -> str | None:
    """Normalize a raw numeric string to a canonical decimal string.

    Handles: integers, decimals, negative numbers, comma-separated
    thousands, basic currency ($), and percentages (%, converted to the
    equivalent fraction value, e.g. "50%" -> "0.5").

    Returns None if `raw` does not contain a parseable number.
    """
    if raw is None:
        return None

    s = raw.strip()
    if not s:
        return None

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()

    s = s.replace("$", "").replace(",", "").strip()

    if not s:
        return None

    try:
        value = float(s)
    except ValueError:
        return None

    if is_percent:
        value = value / 100.0

    # Canonicalize: integers render without a trailing ".0"; otherwise
    # strip trailing zeros so "9.50" and "9.5" compare equal.
    if value == int(value):
        return str(int(value))
    normalized = f"{value:.10f}".rstrip("0").rstrip(".")
    return normalized


def extract_predicted_answer(response: str) -> str | None:
    """Extract the model's final numeric answer from free-form text.

    Tries specific "answer is / #### / final answer" phrasings first
    (most reliable), then falls back to the last number-like substring
    in the response. Returns the normalized numeric string, or None if
    no number could be found at all.

    This is the method-agnostic extractor, kept for simple callers and
    existing tests. See `extract_predicted_answer_with_method` for the
    version that also reports *how* the answer was found, and
    `extract_final_answer` for the version that additionally decides
    whether a fallback-only match should be trusted at all.
    """
    answer, _method = extract_predicted_answer_with_method(response)
    return answer


def extract_predicted_answer_with_method(response: str) -> tuple[str | None, str]:
    """Like `extract_predicted_answer`, but also reports how the answer
    was found: 'phrase' (matched an explicit "the answer is/####/..."
    pattern — reliable), 'fallback' (no explicit phrase; just the last
    number-looking token in the text — unreliable on its own, since a
    degenerate repetition loop or truncated mid-calculation text will
    always contain *some* trailing number), or 'none' (no number at all).
    """
    if not response:
        return None, "none"

    truncated = _truncate_at_continuation(response)

    for pattern in _ANSWER_PHRASE_PATTERNS:
        matches = re.findall(pattern, truncated, flags=re.IGNORECASE)
        if matches:
            normalized = normalize_number(matches[-1])
            if normalized is not None:
                return normalized, "phrase"

    fallback_matches = re.findall(_ANY_NUMBER_PATTERN, truncated)
    # Filter out matches that are just "-" or "$" with no digits, which
    # the regex's optional groups can otherwise let through as ''.
    fallback_matches = [m for m in fallback_matches if re.search(r"\d", m)]
    if fallback_matches:
        normalized = normalize_number(fallback_matches[-1])
        if normalized is not None:
            return normalized, "fallback"

    return None, "none"


def extract_final_answer(response: str, hit_max_new_tokens: bool) -> dict:
    """Decide the authoritative extracted answer and a termination_status
    label, given whether generation was truncated at the token cap.

    A fallback (non-phrase) match found in a response that was cut off at
    the token cap is NOT trusted as a genuine answer — empirically (see
    docs/research_log.md) these are overwhelmingly stray numbers from a
    degenerate repetition loop or an arithmetic drift that never reached
    a stated conclusion, not a real final answer that just lacked
    "the answer is" phrasing. A fallback match in a response that
    stopped on its own (no cap hit) IS trusted, since GSM8K solutions
    sometimes end in a bare number without explicit phrasing.

    Returns a dict with:
        raw_extracted_answer: str|None   whatever extraction found, unfiltered
        extraction_method:    'phrase' | 'fallback' | 'none'
        extracted_answer:     str|None   authoritative answer for scoring
        termination_status:   'stopped_with_answer' | 'stopped_no_answer'
                             | 'capped_with_answer'  | 'capped_no_answer'
    """
    raw_answer, method = extract_predicted_answer_with_method(response)

    if method == "phrase":
        authoritative = raw_answer
        status = "capped_with_answer" if hit_max_new_tokens else "stopped_with_answer"
    elif hit_max_new_tokens:
        # fallback or none, and the response was truncated at the cap:
        # do not trust a bare trailing number as a genuine answer.
        authoritative = None
        status = "capped_no_answer"
    elif raw_answer is not None:
        authoritative = raw_answer
        status = "stopped_with_answer"
    else:
        authoritative = None
        status = "stopped_no_answer"

    return {
        "raw_extracted_answer": raw_answer,
        "extraction_method": method,
        "extracted_answer": authoritative,
        "termination_status": status,
    }


def is_correct(predicted: str | None, reference: str | None) -> bool:
    """Compare a predicted and reference answer after normalization.

    Both inputs are expected to already be raw (un-normalized) strings;
    this function normalizes both before comparing so callers don't need
    to remember to do so themselves.
    """
    if predicted is None or reference is None:
        return False
    pred_norm = normalize_number(predicted)
    ref_norm = normalize_number(reference)
    if pred_norm is None or ref_norm is None:
        return False
    return pred_norm == ref_norm
