import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_metrics


def make_record(is_correct, response_tokens, generation_time=1.0, tokens_per_second=10.0, termination_status=None):
    record = {
        "is_correct": is_correct,
        "response_tokens": response_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
    }
    if termination_status is not None:
        record["termination_status"] = termination_status
    return record


class TestComputeMetrics:
    def test_all_correct(self):
        records = [make_record(True, 50) for _ in range(5)]
        m = compute_metrics(records)
        assert m.total_examples == 5
        assert m.correct == 5
        assert m.incorrect == 0
        assert m.accuracy == 1.0

    def test_all_incorrect(self):
        records = [make_record(False, 50) for _ in range(5)]
        m = compute_metrics(records)
        assert m.accuracy == 0.0
        assert m.incorrect == 5

    def test_mixed_accuracy(self):
        records = [make_record(True, 50), make_record(False, 50), make_record(True, 50), make_record(False, 50)]
        m = compute_metrics(records)
        assert m.accuracy == 0.5
        assert m.correct == 2
        assert m.incorrect == 2

    def test_response_length_stats(self):
        records = [make_record(True, 100), make_record(True, 200), make_record(True, 300)]
        m = compute_metrics(records)
        assert m.avg_response_length == 200
        assert m.median_response_length == 200

    def test_single_record_stdev_zero(self):
        records = [make_record(True, 100)]
        m = compute_metrics(records)
        assert m.stdev_response_length == 0.0

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_length_buckets_assignment(self):
        records = [
            make_record(True, 50),    # 0-100
            make_record(False, 150),  # 100-200
            make_record(True, 350),   # 200-400
            make_record(True, 900),   # 800+
        ]
        m = compute_metrics(records)
        buckets = m.accuracy_by_length_bucket
        assert buckets["0-100"]["count"] == 1
        assert buckets["0-100"]["accuracy"] == 1.0
        assert buckets["100-200"]["count"] == 1
        assert buckets["100-200"]["accuracy"] == 0.0
        assert buckets["200-400"]["count"] == 1
        assert buckets["400-800"]["count"] == 0
        assert buckets["400-800"]["accuracy"] is None
        assert buckets["800+"]["count"] == 1

    def test_bucket_boundary_is_half_open(self):
        # A record of exactly 100 tokens should land in the 100-200
        # bucket, not 0-100 (bucket ranges are [lo, hi)).
        records = [make_record(True, 100)]
        m = compute_metrics(records)
        assert m.accuracy_by_length_bucket["0-100"]["count"] == 0
        assert m.accuracy_by_length_bucket["100-200"]["count"] == 1

    def test_tokens_per_second_averaged(self):
        records = [make_record(True, 50, tokens_per_second=10.0), make_record(True, 50, tokens_per_second=20.0)]
        m = compute_metrics(records)
        assert m.avg_tokens_per_second == 15.0

    def test_termination_status_counts(self):
        records = [
            make_record(True, 50, termination_status="stopped_with_answer"),
            make_record(True, 50, termination_status="capped_with_answer"),
            make_record(False, 50, termination_status="capped_no_answer"),
            make_record(False, 50, termination_status="capped_no_answer"),
        ]
        m = compute_metrics(records)
        assert m.termination_status_counts == {
            "stopped_with_answer": 1,
            "capped_with_answer": 1,
            "capped_no_answer": 2,
        }

    def test_termination_status_missing_counts_as_unknown(self):
        records = [make_record(True, 50), make_record(False, 50)]
        m = compute_metrics(records)
        assert m.termination_status_counts == {"unknown": 2}
