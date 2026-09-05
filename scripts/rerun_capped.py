"""
Rerun specific examples from an existing run under a higher token budget
than the original — targeted at examples whose termination_status shows
they were truncated (by default, capped_no_answer: hit the cap with no
coherent answer at all).

Does NOT touch the original run's files. Writes a new file inside the
same run directory: rerun_<status>_maxtok<N>.jsonl.

This exists because Experiment 0 deliberately kept max_new_tokens=512
based on evidence that raising it wouldn't change most outcomes (see
docs/research_log.md, Finding 3) — but that evidence was from a 100-
example sample. This script lets us re-check specific truncated examples
from the full run under a larger budget without rerunning the whole
1319-example set, and without ever silently assuming what a higher
budget would have produced.

Usage:
    python scripts/rerun_capped.py results/exp00_baseline/run_XXXXXXXX_XXXXXX \
        --status capped_no_answer --max-new-tokens 2048
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.gsm8k import build_completion_prompt, load_gsm8k_test, to_examples
from src.evaluation.answer_extraction import extract_final_answer, is_correct
from src.inference.generate import Generator
from src.utils.logging import get_logger

logger = get_logger("rerun_capped")


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun truncated examples from a run under a higher token budget")
    parser.add_argument("run_dir", type=str, help="Path to the source run directory")
    parser.add_argument(
        "--status",
        type=str,
        default="capped_no_answer",
        help="termination_status to target for rerun (default: capped_no_answer)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="New max_new_tokens for the rerun")
    parser.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="Which JSONL in run_dir to read termination_status from (default: prefers "
        "generations_reprocessed.jsonl, falls back to generations.jsonl)",
    )
    return parser.parse_args()


def load_source_records(run_dir: Path, source_file: str | None) -> list[dict]:
    if source_file:
        path = run_dir / source_file
    else:
        reprocessed = run_dir / "generations_reprocessed.jsonl"
        path = reprocessed if reprocessed.exists() else run_dir / "generations.jsonl"
    with open(path) as f:
        records = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(records)} source records from {path}")
    return records


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())

    source_records = load_source_records(run_dir, args.source_file)
    targeted = [r for r in source_records if r.get("termination_status") == args.status]
    logger.info(f"Targeting {len(targeted)} examples with termination_status='{args.status}'")

    if not targeted:
        logger.info("Nothing to rerun.")
        return

    dataset = load_gsm8k_test(config["dataset_name"], config["dataset_config"])
    examples = {ex.example_id: ex for ex in to_examples(dataset)}

    logger.info(f"Loading model: {config['model_name']} (max_new_tokens={args.max_new_tokens})")
    generator = Generator(
        model_name=config["model_name"],
        dtype=config["dtype"],
        device=config["device"],
        max_new_tokens=args.max_new_tokens,
        do_sample=config["do_sample"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        seed=config["seed"],
    )

    output_path = run_dir / f"rerun_{args.status}_maxtok{args.max_new_tokens}.jsonl"
    now_answered = 0
    now_correct = 0

    with open(output_path, "w") as out_file:
        for r in targeted:
            ex = examples[r["example_id"]]
            prompt = build_completion_prompt(ex.question)
            result = generator.generate(prompt)
            extraction = extract_final_answer(result.response_text, result.hit_max_new_tokens)
            correct = is_correct(extraction["extracted_answer"], ex.reference_answer)

            if extraction["extracted_answer"] is not None:
                now_answered += 1
            if correct:
                now_correct += 1

            record = {
                "example_id": ex.example_id,
                "question": ex.question,
                "reference_answer": ex.reference_answer,
                "original_termination_status": r.get("termination_status"),
                "original_response_tokens": r.get("response_tokens"),
                "rerun_max_new_tokens": args.max_new_tokens,
                "rerun_model_response": result.response_text,
                "rerun_extraction_method": extraction["extraction_method"],
                "rerun_extracted_answer": extraction["extracted_answer"],
                "rerun_termination_status": extraction["termination_status"],
                "rerun_is_correct": correct,
                "rerun_response_tokens": result.response_tokens,
                "rerun_hit_max_new_tokens": result.hit_max_new_tokens,
                "rerun_generation_time": result.generation_time,
                "timestamp": datetime.now().isoformat(),
            }
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            logger.info(
                f"[{ex.example_id}] {r.get('termination_status')} -> {extraction['termination_status']} "
                f"({result.response_tokens} tok, correct={correct})"
            )

    logger.info(f"Reran {len(targeted)} examples -> {output_path}")
    logger.info(
        f"Now produced SOME coherent answer: {now_answered}/{len(targeted)} "
        f"(were 0/{len(targeted)} under the original cap, by definition of '{args.status}')"
    )
    logger.info(f"Now correct: {now_correct}/{len(targeted)}")


if __name__ == "__main__":
    main()
