"""
Stage 3 finalization — merge the main teacher-generation file and the
retry/manual-fix pool into ONE clean, deduplicated, ground-truth-verified
pool ready for Stage 5 Arms A/C/D derivation.

Merge rule per example_id (retry pool always wins when present, since it's
strictly newer information — either a successful automated retry, or a
from-scratch manual derivation that was independently checked against
reference_answer before being accepted):
    1. If example_id is in traces_retried.jsonl AND that row is usable
       (is_correct == True, i.e. recovered — not a disputed exclusion),
       use that row's teacher_trace.
    2. Else if example_id's original traces_n2000.jsonl row already had
       is_correct == True (never needed retrying), use that.
    3. Else (only in the disputed-exclusion set, or genuinely never
       resolved) — excluded from the final pool. Every exclusion is logged
       with its reason so nothing silently disappears unexplained.

Output: results/exp03_teacher/teacher_pool_final.jsonl (one row per
example_id, verified-correct only) plus a summary printed to stdout.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.answer_extraction import is_correct as answers_match
from src.utils.logging import get_logger

logger = get_logger("build_final_teacher_pool")

MAIN_FILE = Path("results/exp03_teacher/traces_n2000.jsonl")
RETRY_FILE = Path("results/exp03_teacher/traces_retried.jsonl")
OUT_FILE = Path("results/exp03_teacher/teacher_pool_final.jsonl")


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["example_id"]] = row
    return out


def main():
    main_rows = load_jsonl(MAIN_FILE)
    retry_rows = load_jsonl(RETRY_FILE)

    logger.info(f"Main file: {len(main_rows)} rows. Retry pool: {len(retry_rows)} rows.")

    final_pool = {}
    excluded_disputed = []
    excluded_unresolved = []
    used_from_retry = 0
    used_from_main = 0

    for example_id, row in main_rows.items():
        retry_row = retry_rows.get(example_id)

        if retry_row is not None:
            if retry_row.get("exclude_from_training"):
                excluded_disputed.append(example_id)
                continue
            if retry_row.get("is_correct"):
                # Sanity check: verify answer still matches reference before trusting.
                if answers_match(retry_row.get("extracted_answer"), retry_row.get("reference_answer")):
                    final_pool[example_id] = retry_row
                    used_from_retry += 1
                    continue
            # Retry row exists but isn't usable (still wrong, not yet disputed) — unresolved.
            excluded_unresolved.append(example_id)
            continue

        # No retry row at all — either never needed retrying, or not yet processed.
        if row.get("is_correct") and answers_match(row.get("extracted_answer"), row.get("reference_answer")):
            final_pool[example_id] = row
            used_from_main += 1
        else:
            excluded_unresolved.append(example_id)

    with open(OUT_FILE, "w") as f:
        for example_id in sorted(final_pool):
            f.write(json.dumps(final_pool[example_id]) + "\n")

    logger.info(f"Final pool: {len(final_pool)} examples -> {OUT_FILE}")
    logger.info(f"  from main (never needed retry): {used_from_main}")
    logger.info(f"  from retry pool (recovered): {used_from_retry}")
    logger.info(f"Excluded, confirmed disputed (dataset bug): {len(excluded_disputed)}")
    logger.info(f"Excluded, still unresolved: {len(excluded_unresolved)} -> {excluded_unresolved}")
    logger.info(
        f"Coverage: {len(final_pool)}/{len(main_rows)} "
        f"({100*len(final_pool)/len(main_rows):.1f}%) of generated examples usable for training"
    )


if __name__ == "__main__":
    main()
