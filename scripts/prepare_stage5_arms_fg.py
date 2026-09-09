"""
Stage 5, Arms F (answer-only) and G (random-shortened control) — the two
remaining reasoning-format variants from the plan's 7-arm matrix.

Also re-filters Arm B against the 38 confirmed-disputed example_ids from
Stage 3's manual-fix pipeline (results/exp03_teacher/traces_retried.jsonl)
for consistency with Arms A/C/D/E, which were all individually verified
against ground truth. Arm B was originally derived straight from GSM8K's
own reference solutions with no independent check, so it could otherwise
silently include some of the same 38 bad labels the other arms exclude.

- Arm F (answer-only): "The answer is <N>." — no reasoning at all. The
  plan's weakest-content baseline.
- Arm G (random-shortened control, CRITICAL): take Arm A's full teacher
  trace and randomly drop whole sentences until it's roughly as short as
  Arm B's concise version of the same example — same length reduction as
  a deliberate-content arm, but achieved by RANDOM removal instead. This
  is the control that lets you tell "shorter reasoning wins" apart from
  "this specific content matters" (the plan's own stated reason for
  including it). Matched against Arm B specifically because B has full
  7473-example coverage (unlike E, which has gaps), so every Arm A
  example has a length target to match against.

Usage:
    python scripts/prepare_stage5_arms_fg.py
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger("prepare_stage5_arms_fg")

TEACHER_POOL = Path("results/exp03_teacher/teacher_pool_final.jsonl")
RETRY_POOL = Path("results/exp03_teacher/traces_retried.jsonl")
ARM_A = Path("results/exp05_arms_acd/arm_a_full_n7435.jsonl")
ARM_B_IN = Path("results/exp05_arms_be/arm_b_concise_n7473.jsonl")
OUT_ACD_DIR = Path("results/exp05_arms_acd")
OUT_BE_DIR = Path("results/exp05_arms_be")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_sentences(text: str) -> list[str]:
    # Split on blank lines first (paragraph-level), then sentences within —
    # keeps whole markdown sections/equations intact rather than mid-formula cuts.
    chunks = re.split(r"(\n\s*\n)", text)
    return [c for c in chunks if c.strip()]


def random_shorten(text: str, target_len: int, rng: random.Random) -> str:
    """Randomly drop whole chunks (paragraphs/sections) until <= target_len
    chars, preserving order of what remains (not reordering)."""
    chunks = split_sentences(text)
    content_chunks = [c for c in chunks if c.strip() and not c.isspace()]
    if not content_chunks:
        return text
    indices = list(range(len(content_chunks)))
    rng.shuffle(indices)
    keep = set(range(len(content_chunks)))
    current_len = sum(len(c) for c in content_chunks)
    for idx in indices:
        if current_len <= target_len or len(keep) <= 1:
            break
        keep.discard(idx)
        current_len -= len(content_chunks[idx])
    return "".join(c for i, c in enumerate(content_chunks) if i in keep)


def main():
    random.seed(42)
    rng = random.Random(42)

    disputed_ids = {
        r["example_id"] for r in load_jsonl(RETRY_POOL) if r.get("exclude_from_training")
    }
    logger.info(f"Confirmed-disputed example_ids to exclude everywhere: {len(disputed_ids)}")

    # --- Fix Arm B: filter against disputed_ids for consistency ---
    arm_b_rows = load_jsonl(ARM_B_IN)
    arm_b_filtered = [r for r in arm_b_rows if r["example_id"] not in disputed_ids]
    out_b = OUT_BE_DIR / f"arm_b_concise_n{len(arm_b_filtered)}_filtered.jsonl"
    with open(out_b, "w") as f:
        for r in arm_b_filtered:
            f.write(json.dumps(r) + "\n")
    logger.info(
        f"Arm B refiltered: {len(arm_b_filtered)}/{len(arm_b_rows)} "
        f"({len(arm_b_rows) - len(arm_b_filtered)} disputed rows removed) -> {out_b}"
    )

    # --- Arm F: answer-only, full 7473 coverage minus disputed ---
    teacher_pool = {r["example_id"]: r for r in load_jsonl(TEACHER_POOL)}
    arm_f_rows = []
    for r in arm_b_filtered:  # arm_b_filtered already covers all non-disputed example_ids
        arm_f_rows.append(
            {
                "example_id": r["example_id"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "reasoning_trace": f"The answer is {r['reference_answer']}.",
                "arm": "F_answer_only",
                "source": "reference_answer_only",
            }
        )
    out_f = OUT_ACD_DIR / f"arm_f_answer_only_n{len(arm_f_rows)}.jsonl"
    with open(out_f, "w") as f:
        for r in arm_f_rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Arm F (answer-only): {len(arm_f_rows)} -> {out_f}")

    # --- Arm G: random-shortened control, matched against Arm B's length per example ---
    arm_a_rows = {r["example_id"]: r for r in load_jsonl(ARM_A)}
    arm_b_lengths = {r["example_id"]: len(r["reasoning_trace"]) for r in arm_b_filtered}

    arm_g_rows = []
    skipped_no_b_target = 0
    for example_id, a_row in arm_a_rows.items():
        if example_id in disputed_ids:
            continue
        target_len = arm_b_lengths.get(example_id)
        if target_len is None:
            skipped_no_b_target += 1
            continue
        shortened = random_shorten(a_row["reasoning_trace"], target_len, rng)
        arm_g_rows.append(
            {
                "example_id": example_id,
                "question": a_row["question"],
                "reference_answer": a_row["reference_answer"],
                "reasoning_trace": shortened,
                "arm": "G_random_shortened_control",
                "source": "teacher_trace_randomly_truncated",
                "target_len_matched_to": "arm_b",
            }
        )
    out_g = OUT_ACD_DIR / f"arm_g_random_control_n{len(arm_g_rows)}.jsonl"
    with open(out_g, "w") as f:
        for r in arm_g_rows:
            f.write(json.dumps(r) + "\n")
    logger.info(
        f"Arm G (random-shortened control): {len(arm_g_rows)} "
        f"({skipped_no_b_target} skipped, no Arm B length target) -> {out_g}"
    )


if __name__ == "__main__":
    main()
