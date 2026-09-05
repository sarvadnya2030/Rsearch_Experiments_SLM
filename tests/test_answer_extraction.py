import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extraction import (
    extract_final_answer,
    extract_predicted_answer,
    extract_predicted_answer_with_method,
    is_correct,
    normalize_number,
)


class TestNormalizeNumber:
    def test_integer(self):
        assert normalize_number("42") == "42"

    def test_decimal(self):
        assert normalize_number("3.14") == "3.14"

    def test_negative(self):
        assert normalize_number("-7") == "-7"

    def test_comma_separated(self):
        assert normalize_number("1,234") == "1234"

    def test_comma_and_decimal(self):
        assert normalize_number("1,234.50") == "1234.5"

    def test_currency(self):
        assert normalize_number("$42") == "42"

    def test_percentage(self):
        assert normalize_number("50%") == "0.5"

    def test_trailing_zero_decimal(self):
        assert normalize_number("9.50") == "9.5"

    def test_integral_float_renders_as_int(self):
        assert normalize_number("18.0") == "18"

    def test_none_input(self):
        assert normalize_number(None) is None

    def test_empty_string(self):
        assert normalize_number("") is None

    def test_garbage_input(self):
        assert normalize_number("abc") is None

    def test_whitespace_padded(self):
        assert normalize_number("  42  ") == "42"


class TestExtractPredictedAnswer:
    def test_gsm8k_marker_format(self):
        assert extract_predicted_answer("Some reasoning.\n#### 42") == "42"

    def test_the_answer_is(self):
        assert extract_predicted_answer("The answer is 42.") == "42"

    def test_final_answer_is(self):
        assert extract_predicted_answer("Therefore, the final answer is 42.") == "42"

    def test_answer_colon(self):
        assert extract_predicted_answer("Answer: 42") == "42"

    def test_case_insensitive(self):
        assert extract_predicted_answer("THE ANSWER IS 42.") == "42"

    def test_negative_answer(self):
        assert extract_predicted_answer("The answer is -5.") == "-5"

    def test_decimal_answer(self):
        assert extract_predicted_answer("The answer is 3.5.") == "3.5"

    def test_comma_separated_answer(self):
        assert extract_predicted_answer("The answer is 1,234.") == "1234"

    def test_currency_answer(self):
        assert extract_predicted_answer("The answer is $18.") == "18"

    def test_fallback_last_number_in_text(self):
        # No recognized phrase, but a trailing number exists.
        assert extract_predicted_answer("She has 16 eggs and sells 9 of them for 18") == "18"

    def test_multiple_matches_takes_last(self):
        text = "The answer is 10. Wait, actually the answer is 42."
        assert extract_predicted_answer(text) == "42"

    def test_no_number_present(self):
        assert extract_predicted_answer("I don't know.") is None

    def test_empty_response(self):
        assert extract_predicted_answer("") is None

    def test_none_response(self):
        assert extract_predicted_answer(None) is None

    def test_prefers_marker_over_fallback(self):
        text = "Step 1 uses 16 eggs.\n#### 18"
        assert extract_predicted_answer(text) == "18"

    def test_ignores_hallucinated_followup_question_bracketed(self):
        # Base models often keep generating past their answer and invent
        # an unrelated new question — must not pick up its answer.
        text = "2 + 1 = 3\nSo the answer is 3.\n\n[Question]A man is 20 years older...\n[Answer]So the answer is 18."
        assert extract_predicted_answer(text) == "3"

    def test_ignores_hallucinated_followup_question_plain(self):
        text = "So the answer is 3.\n\nQuestion: A man is 20 years older...\nAnswer: So the answer is 18."
        assert extract_predicted_answer(text) == "3"

    def test_ignores_hallucinated_followup_q_colon(self):
        text = "The answer is 55.\nQ: A car travels 40 kph...\nA: The answer is 140."
        assert extract_predicted_answer(text) == "55"


class TestIsCorrect:
    def test_matching_integers(self):
        assert is_correct("42", "42") is True

    def test_mismatched_integers(self):
        assert is_correct("42", "41") is False

    def test_matching_after_normalization(self):
        assert is_correct("1,234", "1234") is True
        assert is_correct("9.50", "9.5") is True

    def test_none_predicted(self):
        assert is_correct(None, "42") is False

    def test_none_reference(self):
        assert is_correct("42", None) is False

    def test_both_none(self):
        assert is_correct(None, None) is False


class TestExtractPredictedAnswerWithMethod:
    def test_phrase_match_reports_phrase(self):
        answer, method = extract_predicted_answer_with_method("The answer is 42.")
        assert answer == "42"
        assert method == "phrase"

    def test_fallback_match_reports_fallback(self):
        answer, method = extract_predicted_answer_with_method("She has 16 eggs and sells 9 of them for 18")
        assert answer == "18"
        assert method == "fallback"

    def test_no_number_reports_none(self):
        answer, method = extract_predicted_answer_with_method("I don't know.")
        assert answer is None
        assert method == "none"


class TestExtractFinalAnswer:
    def test_stopped_with_answer(self):
        result = extract_final_answer("The answer is 42.", hit_max_new_tokens=False)
        assert result["extracted_answer"] == "42"
        assert result["extraction_method"] == "phrase"
        assert result["termination_status"] == "stopped_with_answer"

    def test_capped_with_answer_still_trusted(self):
        # Model answered explicitly, then kept rambling until the cap —
        # the phrase-based answer is still trustworthy.
        text = "The answer is 42. " + ("more rambling text " * 50)
        result = extract_final_answer(text, hit_max_new_tokens=True)
        assert result["extracted_answer"] == "42"
        assert result["extraction_method"] == "phrase"
        assert result["termination_status"] == "capped_with_answer"

    def test_capped_fallback_number_is_not_trusted(self):
        # Real example from a degenerate repetition loop (exp00 run
        # 20260905_183615, example 7): no answer phrase anywhere, just a
        # stray trailing number from mid-loop arithmetic. Must NOT be
        # treated as a genuine answer.
        text = (
            "200 / 2 = 100 GB downloaded. 100 * 2 = 200 GB downloaded. "
            "200 - 200 = 0 GB downloaded. 200 / 2 = 100 GB downloaded. "
            "100 * 2 = 200 GB downloaded. 200 - 200 = 0 GB downloaded. 10"
        )
        result = extract_final_answer(text, hit_max_new_tokens=True)
        assert result["raw_extracted_answer"] == "10"  # what fallback found, preserved for transparency
        assert result["extraction_method"] == "fallback"
        assert result["extracted_answer"] is None  # NOT trusted as authoritative
        assert result["termination_status"] == "capped_no_answer"

    def test_capped_no_number_at_all(self):
        text = "I am thinking about this problem carefully " * 30
        result = extract_final_answer(text, hit_max_new_tokens=True)
        assert result["extracted_answer"] is None
        assert result["termination_status"] == "capped_no_answer"

    def test_stopped_fallback_is_trusted(self):
        # Model stopped on its own (no cap hit) with a bare trailing
        # number and no explicit phrase — this IS trusted, since short
        # natural completions sometimes end this way.
        result = extract_final_answer("She has 16 eggs and sells 9 of them for 18", hit_max_new_tokens=False)
        assert result["extracted_answer"] == "18"
        assert result["extraction_method"] == "fallback"
        assert result["termination_status"] == "stopped_with_answer"

    def test_stopped_no_answer_at_all(self):
        result = extract_final_answer("I don't know.", hit_max_new_tokens=False)
        assert result["extracted_answer"] is None
        assert result["termination_status"] == "stopped_no_answer"
