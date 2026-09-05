"""
Failure-record collection for Experiment 0.

Deliberately does NOT auto-classify failures. `error_category` is fixed
to "unclassified" for every record — real taxonomy design comes only
after a human manually inspects a sample of actual failures (see project
notes section 13).
"""

import json
from pathlib import Path

UNCLASSIFIED = "unclassified"


def build_error_record(record: dict) -> dict:
    """Build an errors.jsonl record from a full generation record.

    Expects `record` to already contain example_id, question,
    reference_solution, reference_answer, model_response,
    extracted_answer (all produced during generation/scoring).
    """
    return {
        "example_id": record["example_id"],
        "question": record["question"],
        "reference_solution": record["reference_solution"],
        "reference_answer": record["reference_answer"],
        "model_response": record["model_response"],
        "extracted_answer": record["extracted_answer"],
        "error_category": UNCLASSIFIED,
    }


def write_errors_jsonl(records: list[dict], output_path: Path) -> int:
    """Write incorrect records to errors.jsonl. Returns count written."""
    errors = [build_error_record(r) for r in records if not r["is_correct"]]
    with open(output_path, "w") as f:
        for err in errors:
            f.write(json.dumps(err) + "\n")
    return len(errors)
