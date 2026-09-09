"""
Stage 5 — orchestrates the full reasoning-format ablation: trains one
Qwen3-0.6B-Base model per (arm, seed) via train_sft.py, then evaluates
that exact checkpoint on the GSM8K test split by reusing run_baseline.py
unmodified (just pointing model_name at the checkpoint dir) — so results
are directly comparable to the Stage 2 baseline (52.16%, 95% CI
[49.46%, 54.86%], n=1319).

Sequential, not parallel (one GPU). Resumable: skips any (arm, seed)
pair whose run_config.json + eval metrics.json already exist, so a
reboot or interruption only costs the in-flight run, not the whole sweep
— same principle as the Stage 3 teacher-generation --resume mechanism.

Usage:
    python scripts/run_stage5_sweep.py --seeds 42
    python scripts/run_stage5_sweep.py --seeds 42 123 7 --eval-limit 100
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_stage5_sweep")

ARMS = {
    "A": "results/exp05_arms_acd/arm_a_full_n7435.jsonl",
    "B": "results/exp05_arms_be/arm_b_concise_n7435_filtered.jsonl",
    "C": "results/exp05_arms_acd/arm_c_no_verification_n7435.jsonl",
    "D": "results/exp05_arms_acd/arm_d_no_reflection_n7435.jsonl",
    "E": "results/exp05_arms_be/arm_e_symbolic_n6973_filtered.jsonl",
    "F": "results/exp05_arms_acd/arm_f_answer_only_n7435.jsonl",
    "G": "results/exp05_arms_acd/arm_g_random_control_n7435.jsonl",
}

SFT_OUT_ROOT = Path("results/exp05_sft")
BASE_EVAL_CONFIG = Path("configs/exp00_baseline.yaml")


def run_name(arm: str, seed: int) -> str:
    return f"arm_{arm}_seed{seed}"


def already_done(out_dir: Path) -> bool:
    return (out_dir / "run_config.json").exists() and (out_dir / "eval" / "metrics.json").exists()


def train_one(arm: str, data_path: str, seed: int, out_dir: Path, epochs: int, eval_limit: int) -> dict:
    if not (out_dir / "run_config.json").exists():
        logger.info(f"=== TRAINING arm={arm} seed={seed} -> {out_dir} ===")
        cmd = [
            sys.executable, "scripts/train_sft.py",
            "--arm", arm,
            "--data", data_path,
            "--seed", str(seed),
            "--out-dir", str(out_dir),
            "--epochs", str(epochs),
            "--per-device-batch-size", "2",
            "--grad-accum-steps", "8",
            "--max-length", "768",
            "--logging-steps", "10",
        ]
        result = subprocess.run(cmd, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **_env()})
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for arm={arm} seed={seed} (exit {result.returncode})")
    else:
        logger.info(f"Training already done for arm={arm} seed={seed}, skipping")

    eval_dir = out_dir / "eval"
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        logger.info(f"=== EVALUATING arm={arm} seed={seed} on {eval_limit} test examples ===")
        eval_config = yaml.safe_load(BASE_EVAL_CONFIG.read_text())
        eval_config["model_name"] = str(out_dir.resolve())
        eval_config["output_directory"] = str(eval_dir)
        eval_config["seed"] = seed
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_config_path = eval_dir / "eval_config.yaml"
        eval_config_path.write_text(yaml.dump(eval_config))

        cmd = [
            sys.executable, "scripts/run_baseline.py",
            "--config", str(eval_config_path),
            "--limit", str(eval_limit),
        ]
        result = subprocess.run(cmd, env=_env())
        if result.returncode != 0:
            raise RuntimeError(f"Eval failed for arm={arm} seed={seed} (exit {result.returncode})")

        # run_baseline.py writes into a run_YYYYMMDD_HHMMSS subdir under
        # output_directory (its own convention) — find it and copy its
        # metrics.json up to the flat path this script expects.
        run_subdirs = sorted(eval_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not run_subdirs:
            raise RuntimeError(f"run_baseline.py produced no run_* subdir under {eval_dir}")
        latest_run = run_subdirs[-1]
        metrics_path.write_text((latest_run / "metrics.json").read_text())
    else:
        logger.info(f"Eval already done for arm={arm} seed={seed}, skipping")

    return json.loads(metrics_path.read_text())


def _env():
    import os
    return dict(os.environ)


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()), help="Subset of arms to run (default: all 7)")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-limit", type=int, default=100, help="GSM8K test examples per eval (100 ~ Stage 2's first-look precedent, ±~5% CI)")
    ap.add_argument("--summary-out", default="results/exp05_sft/sweep_summary.jsonl")
    args = ap.parse_args()

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_start = datetime.now(timezone.utc)
    all_results = []

    for arm in args.arms:
        data_path = ARMS[arm]
        for seed in args.seeds:
            out_dir = SFT_OUT_ROOT / run_name(arm, seed)
            metrics = train_one(arm, data_path, seed, out_dir, args.epochs, args.eval_limit)
            correct, total = metrics["correct"], metrics["total_examples"]
            ci_low, ci_high = wilson_ci(correct, total)
            row = {
                "arm": arm,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "correct": correct,
                "total": total,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "out_dir": str(out_dir),
            }
            all_results.append(row)
            with open(summary_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(f"arm={arm} seed={seed}: {metrics['accuracy']:.4f} ({correct}/{total}) 95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    sweep_end = datetime.now(timezone.utc)
    logger.info(f"Sweep complete: {len(all_results)} (arm, seed) runs in {(sweep_end - sweep_start).total_seconds()/3600:.2f} hours")
    logger.info(f"Full results: {summary_path}")


if __name__ == "__main__":
    main()
