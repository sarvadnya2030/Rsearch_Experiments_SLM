"""
Aggregate metrics over a set of per-example generation records.

Deliberately descriptive only. The response-length buckets below are for
observation, not causal claims — see project notes: do not infer that
response length *causes* correctness from these buckets.
"""

import statistics
from dataclasses import dataclass, field

LENGTH_BUCKETS = [
    (0, 100),
    (100, 200),
    (200, 400),
    (400, 800),
    (800, float("inf")),
]


def _bucket_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f"{int(lo)}+"
    return f"{int(lo)}-{int(hi)}"


@dataclass
class AggregateMetrics:
    total_examples: int
    correct: int
    incorrect: int
    accuracy: float
    avg_response_length: float
    median_response_length: float
    stdev_response_length: float
    avg_generation_time: float
    avg_tokens_per_second: float
    accuracy_by_length_bucket: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "avg_response_length": self.avg_response_length,
            "median_response_length": self.median_response_length,
            "stdev_response_length": self.stdev_response_length,
            "avg_generation_time": self.avg_generation_time,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "accuracy_by_length_bucket": self.accuracy_by_length_bucket,
        }


def compute_metrics(records: list[dict]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of per-example record dicts.

    Each record is expected to have at minimum:
        is_correct: bool
        response_tokens: int
        generation_time: float
        tokens_per_second: float
    """
    if not records:
        raise ValueError("compute_metrics called with an empty record list")

    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    incorrect = total - correct

    lengths = [r["response_tokens"] for r in records]
    gen_times = [r["generation_time"] for r in records]
    tps = [r["tokens_per_second"] for r in records]

    bucket_stats: dict[str, dict] = {}
    for lo, hi in LENGTH_BUCKETS:
        label = _bucket_label(lo, hi)
        bucket_records = [r for r in records if lo <= r["response_tokens"] < hi]
        if bucket_records:
            bucket_correct = sum(1 for r in bucket_records if r["is_correct"])
            bucket_stats[label] = {
                "count": len(bucket_records),
                "correct": bucket_correct,
                "accuracy": bucket_correct / len(bucket_records),
            }
        else:
            bucket_stats[label] = {"count": 0, "correct": 0, "accuracy": None}

    return AggregateMetrics(
        total_examples=total,
        correct=correct,
        incorrect=incorrect,
        accuracy=correct / total,
        avg_response_length=statistics.mean(lengths),
        median_response_length=statistics.median(lengths),
        stdev_response_length=statistics.stdev(lengths) if total > 1 else 0.0,
        avg_generation_time=statistics.mean(gen_times),
        avg_tokens_per_second=statistics.mean(tps),
        accuracy_by_length_bucket=bucket_stats,
    )
