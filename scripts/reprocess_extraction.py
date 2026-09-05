"""
Reprocess an existing run's generations.jsonl with the current answer-
extraction logic, WITHOUT rerunning inference (raw model_response text is
already saved in full, per the raw-preservation requirement).

Writes generations_reprocessed.jsonl, metrics_reprocessed.json, and
errors_reprocessed.jsonl alongside the originals in the same run
directory — the original files are never modified or deleted, per the
"never overwrite a previous run" rule.

Usage:
    python scripts/reprocess_extraction.py results/exp00_baseline/run_20260905_183615
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extraction import extract_final_answer, is_correct
from src.evaluation.error_analysis import write_errors_jsonl
from src.evaluation.metrics import compute_metrics
from src.utils.logging import get_logger
from src.utils.reproducibility import write_json

logger = get_logger("reprocess")


def parse_args():
    parser = argparse.ArgumentParser(description="Reprocess a run's extraction without rerunning inference")
    parser.add_argument("run_dir", type=str, help="Path to the run directory (contains generations.jsonl)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    input_path = run_dir / "generations.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"No generations.jsonl found in {run_dir}")

    with open(input_path) as f:
        old_records = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(old_records)} records from {input_path}")

    new_records = []
    changed_predictions = 0
    for r in old_records:
        # Derive hit_max_new_tokens from the record's own saved generation
        # config, since older records predate that field being logged
        # directly on the record.
        max_new_tokens = r["generation_config"]["max_new_tokens"]
        hit_cap = r["response_tokens"] >= max_new_tokens

        extraction = extract_final_answer(r["model_response"], hit_cap)
        correct = is_correct(extraction["extracted_answer"], r["reference_answer"])

        if extraction["extracted_answer"] != r.get("extracted_answer"):
            changed_predictions += 1

        new_record = dict(r)  # preserve every original field
        new_record["raw_extracted_answer"] = extraction["raw_extracted_answer"]
        new_record["extraction_method"] = extraction["extraction_method"]
        new_record["extracted_answer"] = extraction["extracted_answer"]
        new_record["termination_status"] = extraction["termination_status"]
        new_record["hit_max_new_tokens"] = hit_cap
        new_record["is_correct"] = correct
        new_records.append(new_record)

    output_gen_path = run_dir / "generations_reprocessed.jsonl"
    with open(output_gen_path, "w") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {output_gen_path}")

    metrics = compute_metrics(new_records)
    write_json(metrics.to_dict(), run_dir / "metrics_reprocessed.json")
    logger.info(f"Wrote {run_dir / 'metrics_reprocessed.json'}")

    n_errors = write_errors_jsonl(new_records, run_dir / "errors_reprocessed.jsonl")
    logger.info(f"Wrote {run_dir / 'errors_reprocessed.jsonl'} ({n_errors} errors)")

    logger.info(f"Predictions changed by reprocessing: {changed_predictions}/{len(old_records)}")
    logger.info(f"Accuracy — original vs reprocessed: check metrics.json vs metrics_reprocessed.json")
    logger.info(f"Reprocessed accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total_examples})")
    logger.info(f"Termination status breakdown: {metrics.termination_status_counts}")


if __name__ == "__main__":
    main()
