"""
Stage 5, Arms B (concise) and E (symbolic) — derived directly from GSM8K's
own train-split reference solutions, per the Stage 2.5 side-investigation
decision (docs/research_log.md, Finding 7 / research-plan.md Stage 3):
only 0.8% of reference solutions contain reflection/verification language,
so they're already "concise" and have nothing to strip for Arms A/C/D —
but they're a free, zero-cost, zero-contamination source for Arms B and E.

Arm B (concise reasoning): reference solution's prose, with the inline
calculator annotations ("<<48/2=24>>") removed — same logic, same steps,
just without the calculator-tool artifact that a real training target
shouldn't contain.

Arm E (symbolic reasoning): only the equations pulled out of those same
calculator annotations, chained with "; ", prose dropped entirely.

Both arms are verified against the dataset's own ground-truth final
answer before being kept (never trust a derived trace blindly, same rule
applied to teacher traces in Stage 3).

Usage:
    python scripts/prepare_stage5_arms_be.py --n-examples 50
    python scripts/prepare_stage5_arms_be.py --n-examples 7473 --out-dir results/exp05_arms_be
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import extract_reference_answer
from src.evaluation.answer_extraction import is_correct
from src.utils.logging import get_logger
from datasets import load_dataset

logger = get_logger("prepare_stage5_arms_be")

_CALC_ANNOTATION = re.compile(r"<<([^>]+)>>")


def build_arm_b(reference_solution: str) -> str:
    """Concise: strip calculator annotations, keep the prose steps as-is."""
    text = _CALC_ANNOTATION.sub("", reference_solution)
    # Collapse the blank the annotation removal can leave before its result number.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def build_arm_e(reference_solution: str) -> str | None:
    """Symbolic: only the chained equations, no prose. None if a step has no equation."""
    equations = _CALC_ANNOTATION.findall(reference_solution)
    if not equations:
        return None
    return "; ".join(equations)


def extract_final_line_answer(text: str) -> str | None:
    m = re.search(r"####\s*([\-\$]?[\d,]*\.?\d+%?)", text)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-dir", type=str, default="results/exp05_arms_be")
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    n = min(args.n_examples, len(dataset))
    logger.info(f"Deriving Arms B/E from {n}/{len(dataset)} {args.split} examples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_b_records, arm_e_records = [], []
    arm_e_skipped_no_equation = 0
    arm_b_verify_failed = 0
    arm_e_verify_failed = 0

    for idx in range(n):
        row = dataset[idx]
        reference_solution = row["answer"]
        reference_answer = extract_reference_answer(reference_solution)

        arm_b_text = build_arm_b(reference_solution)
        arm_b_answer = extract_final_line_answer(arm_b_text)
        if is_correct(arm_b_answer, reference_answer):
            arm_b_records.append(
                {
                    "example_id": idx,
                    "question": row["question"],
                    "reference_answer": reference_answer,
                    "reasoning_trace": arm_b_text,
                    "arm": "B_concise",
                    "source": "gsm8k_reference_stripped",
                }
            )
        else:
            arm_b_verify_failed += 1
            logger.warning(f"[Arm B] example {idx}: verify failed ({arm_b_answer!r} != {reference_answer!r})")

        arm_e_text = build_arm_e(reference_solution)
        if arm_e_text is None:
            arm_e_skipped_no_equation += 1
            continue
        # The equation chain's last equation's result should equal the reference answer.
        last_result = arm_e_text.split(";")[-1].strip().split("=")[-1].strip()
        if is_correct(last_result, reference_answer):
            arm_e_records.append(
                {
                    "example_id": idx,
                    "question": row["question"],
                    "reference_answer": reference_answer,
                    "reasoning_trace": arm_e_text + f". Answer: {reference_answer}.",
                    "arm": "E_symbolic",
                    "source": "gsm8k_reference_equations_only",
                }
            )
        else:
            arm_e_verify_failed += 1
            logger.warning(f"[Arm E] example {idx}: verify failed ({last_result!r} != {reference_answer!r})")

    arm_b_path = out_dir / f"arm_b_concise_n{n}.jsonl"
    arm_e_path = out_dir / f"arm_e_symbolic_n{n}.jsonl"
    with open(arm_b_path, "w") as f:
        for r in arm_b_records:
            f.write(json.dumps(r) + "\n")
    with open(arm_e_path, "w") as f:
        for r in arm_e_records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Arm B (concise): {len(arm_b_records)}/{n} kept, {arm_b_verify_failed} verify-failed -> {arm_b_path}")
    logger.info(
        f"Arm E (symbolic): {len(arm_e_records)}/{n} kept, "
        f"{arm_e_skipped_no_equation} skipped (no equations), {arm_e_verify_failed} verify-failed -> {arm_e_path}"
    )

    print("\n--- Sample Arm B (concise) ---")
    print(json.dumps(arm_b_records[0], indent=2))
    print("\n--- Sample Arm E (symbolic) ---")
    print(json.dumps(arm_e_records[0], indent=2))


if __name__ == "__main__":
    main()
