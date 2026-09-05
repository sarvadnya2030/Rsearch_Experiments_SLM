"""
Experiment 0 runner: Qwen3-0.6B-Base natural-reasoning baseline on GSM8K test.

Usage:
    python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 10
    python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 100
    python scripts/run_baseline.py --config configs/exp00_baseline.yaml            # full test set

    # Resume an interrupted run (e.g. killed by a reboot) without
    # redoing already-completed examples:
    python scripts/run_baseline.py --resume-dir results/exp00_baseline/run_YYYYMMDD_HHMMSS

Never overwrites a previous run — each fresh invocation creates a new
results/exp00_baseline/run_YYYYMMDD_HHMMSS/ directory. --resume-dir is
the one exception: it deliberately continues writing into an existing
run's generations.jsonl (append, not truncate) so already-completed
examples and their sunk generation time aren't discarded.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import build_completion_prompt, load_gsm8k_test, to_examples
from src.evaluation.answer_extraction import extract_final_answer, is_correct
from src.evaluation.error_analysis import write_errors_jsonl
from src.evaluation.metrics import compute_metrics
from src.inference.generate import Generator
from src.utils.logging import get_logger
from src.utils.reproducibility import capture_environment, set_seed, write_json

logger = get_logger("exp00")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment 0 baseline")
    parser.add_argument("--config", type=str, default="configs/exp00_baseline.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (smoke/partial runs)")
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume an existing run directory, skipping already-completed example_ids "
        "(reads its config.yaml; --config/--limit are ignored when this is set)",
    )
    return parser.parse_args()


def load_completed_example_ids(run_dir: Path) -> set[int]:
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        return set()
    with open(generations_path) as f:
        return {json.loads(line)["example_id"] for line in f if line.strip()}


def main():
    args = parse_args()

    if args.resume_dir:
        run_dir = Path(args.resume_dir)
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        timestamp = run_dir.name.replace("run_", "")
        completed_ids = load_completed_example_ids(run_dir)
        logger.info(f"Resuming run: {run_dir} ({len(completed_ids)} examples already completed)")
    else:
        config_path = Path(args.config)
        config = yaml.safe_load(config_path.read_text())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config["output_directory"]) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Run directory: {run_dir}")
        # Persist the exact config used for this run.
        (run_dir / "config.yaml").write_text(yaml.dump(config))
        completed_ids = set()

    set_seed(config["seed"])

    logger.info(f"Loading GSM8K ({config['dataset_name']}/{config['dataset_config']}, split={config['split']})")
    dataset = load_gsm8k_test(config["dataset_name"], config["dataset_config"])
    examples = to_examples(dataset)
    if args.resume_dir is None and args.limit is not None:
        examples = examples[: args.limit]
    if completed_ids:
        examples = [ex for ex in examples if ex.example_id not in completed_ids]
        logger.info(f"Skipping {len(completed_ids)} already-completed examples; {len(examples)} remain")
    logger.info(f"Running on {len(examples)} examples")

    if not examples:
        logger.info("Nothing left to run — all examples already completed.")
        return

    logger.info(f"Loading model: {config['model_name']}")
    generator = Generator(
        model_name=config["model_name"],
        dtype=config["dtype"],
        device=config["device"],
        max_new_tokens=config["max_new_tokens"],
        do_sample=config["do_sample"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        seed=config["seed"],
    )

    env = capture_environment(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        generation_config={
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": config["do_sample"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
        },
        seed=config["seed"],
    )
    write_json(env, run_dir / "environment.json")

    generations_path = run_dir / "generations.jsonl"
    records = []
    file_mode = "a" if completed_ids else "w"

    with open(generations_path, file_mode) as gen_file:
        for ex in examples:
            prompt = build_completion_prompt(ex.question)
            result = generator.generate(prompt)

            extraction = extract_final_answer(result.response_text, result.hit_max_new_tokens)
            correct = is_correct(extraction["extracted_answer"], ex.reference_answer)

            record = {
                "example_id": ex.example_id,
                "question": ex.question,
                "reference_solution": ex.reference_solution,
                "reference_answer": ex.reference_answer,
                "model_response": result.response_text,
                "raw_extracted_answer": extraction["raw_extracted_answer"],
                "extraction_method": extraction["extraction_method"],
                "extracted_answer": extraction["extracted_answer"],
                "termination_status": extraction["termination_status"],
                "is_correct": correct,
                "prompt_tokens": result.prompt_tokens,
                "response_tokens": result.response_tokens,
                "total_tokens": result.total_tokens,
                "hit_max_new_tokens": result.hit_max_new_tokens,
                "generation_time": result.generation_time,
                "tokens_per_second": result.tokens_per_second,
                "model_name": config["model_name"],
                "generation_config": {
                    "max_new_tokens": config["max_new_tokens"],
                    "do_sample": config["do_sample"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                },
                "seed": config["seed"],
                "timestamp": datetime.now().isoformat(),
                "git_commit": env["git_commit"],
            }
            records.append(record)
            gen_file.write(json.dumps(record) + "\n")
            gen_file.flush()

            status = "OK " if correct else "ERR"
            logger.info(
                f"[{ex.example_id + 1}/{len(examples)}] {status} "
                f"pred={extraction['extracted_answer']} ref={ex.reference_answer} "
                f"[{extraction['termination_status']}] "
                f"({result.response_tokens} tok, {result.tokens_per_second:.1f} tok/s)"
            )

    # Metrics/errors must cover ALL completed examples, not just the ones
    # generated in this invocation — matters when resuming, since earlier
    # examples from before an interruption are only on disk, not in
    # `records` (which holds only what this process itself generated).
    if completed_ids:
        with open(generations_path) as f:
            all_records = [json.loads(line) for line in f]
    else:
        all_records = records

    metrics = compute_metrics(all_records)
    write_json(metrics.to_dict(), run_dir / "metrics.json")

    n_errors = write_errors_jsonl(all_records, run_dir / "errors.jsonl")

    readme = f"""# Experiment 0 baseline run — {timestamp}

Model: {config['model_name']}
Dataset: {config['dataset_name']}/{config['dataset_config']} ({config['split']} split)
Examples: {len(all_records)}
Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total_examples})
Errors saved: {n_errors}

See config.yaml, environment.json, generations.jsonl, metrics.json, errors.jsonl in this directory.
"""
    (run_dir / "README.md").write_text(readme)

    logger.info(f"Accuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total_examples})")
    logger.info(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
