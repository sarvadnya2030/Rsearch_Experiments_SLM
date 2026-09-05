import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extraction import extract_predicted_answer, is_correct, normalize_number


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
