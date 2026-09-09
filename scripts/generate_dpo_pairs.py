"""
DPO preference-pair generation.

For each GSM8K TRAIN question that has a teacher reasoning trace (from
Arm A), run the base Qwen3-0.6B-Base model on that same question. Where
the base model's own greedy attempt is WRONG, save a preference pair:
    chosen   = Arm A's teacher trace (known-correct, verified elsewhere)
    rejected = the base model's own natural wrong attempt

Using the model's own wrong attempts (rather than an arbitrary bad
example) is deliberate — DPO learns most effectively from rejecting
outputs the model would actually produce on its own, not synthetic
negatives it was never going to generate anyway.

Batched generation (see src/inference/generate.py's generate_batch,
same left-padding technique as self_consistency_batched.py's B3).

Usage:
    python scripts/generate_dpo_pairs.py --out results/exp09_dpo/pairs.jsonl --batch-size 16
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from src.data.gsm8k import build_completion_prompt, extract_reference_answer  # noqa: E402
from src.evaluation.answer_extraction import extract_final_answer, is_correct  # noqa: E402
from src.inference.generate import Generator  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_dpo_pairs")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chosen-source", default="results/exp05_arms_acd/arm_a_full_n7435.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on train examples (debugging)")
    args = ap.parse_args()

    logger.info(f"Loading chosen-side teacher traces from {args.chosen_source}")
    chosen_by_id = {}
    with open(args.chosen_source) as f:
        for line in f:
            row = json.loads(line)
            chosen_by_id[row["example_id"]] = row

    logger.info("Loading GSM8K train split")
    train = load_dataset("openai/gsm8k", "main", split="train")
    questions = []
    for idx, row in enumerate(train):
        if idx not in chosen_by_id:
            continue  # only questions with a verified teacher trace can form a pair
        questions.append({
            "example_id": idx,
            "question": row["question"],
            "reference_answer": extract_reference_answer(row["answer"]),
        })
    if args.limit:
        questions = questions[: args.limit]
    logger.info(f"{len(questions)} train questions have a matching teacher trace")

    logger.info(f"Loading base model: {MODEL_NAME}")
    generator = Generator(
        model_name=MODEL_NAME,
        dtype="float16",
        device="cuda",
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_correct = 0
    with open(out_path, "w") as out_f:
        for batch_start in range(0, len(questions), args.batch_size):
            batch = questions[batch_start : batch_start + args.batch_size]
            prompts = [build_completion_prompt(q["question"]) for q in batch]
            results = generator.generate_batch(prompts)

            for q, result in zip(batch, results):
                extraction = extract_final_answer(result.response_text, result.hit_max_new_tokens)
                correct = is_correct(extraction["extracted_answer"], q["reference_answer"])
                if correct:
                    n_correct += 1
                    continue  # only wrong attempts are useful as "rejected"

                chosen = chosen_by_id[q["example_id"]]
                record = {
                    "example_id": q["example_id"],
                    "prompt": build_completion_prompt(q["question"]),
                    "chosen": " " + chosen["reasoning_trace"].strip(),
                    "rejected": " " + result.response_text.strip(),
                    "reference_answer": q["reference_answer"],
                }
                out_f.write(json.dumps(record) + "\n")
                n_pairs += 1

            out_f.flush()
            done = batch_start + len(batch)
            logger.info(f"[{done}/{len(questions)}] pairs so far: {n_pairs}, base-model correct so far: {n_correct}")

    logger.info(f"Done. {n_pairs} preference pairs written to {out_path} "
                f"(base model was correct on {n_correct}/{len(questions)} = {n_correct/len(questions):.1%})")


if __name__ == "__main__":
    main()
