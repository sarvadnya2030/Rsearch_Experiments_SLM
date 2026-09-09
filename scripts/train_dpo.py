"""
Stage 9 (pulled forward) — DPO training on the preference pairs from
generate_dpo_pairs.py (chosen = Arm A teacher trace, rejected = base
model's own wrong attempt on the same GSM8K train question).

Two modes, selected by --init-model:
  1. "raw DPO"    --init-model Qwen/Qwen3-0.6B-Base
       DPO straight from the untrained base model.
  2. "SFT+DPO"    --init-model results/exp05_sft/arm_A_seed42
       The standard modern recipe: DPO refining an already-SFT'd
       checkpoint, rather than starting from scratch.

Reuses ThermalSafetyCallback/MetricsLoggingCallback/gpu_temp_c/
set_all_seeds from train_sft.py rather than duplicating them, and logs
every parameter (git commit, library versions, full DPOConfig.to_dict(),
model_config.to_dict()) the same way train_sft.py does, for full
research reproducibility.

Usage:
    python scripts/train_dpo.py --init-model Qwen/Qwen3-0.6B-Base \
        --data results/exp09_dpo/pairs.jsonl --seed 42 \
        --out-dir results/exp09_dpo/raw_dpo_seed42

    python scripts/train_dpo.py --init-model results/exp05_sft/arm_A_seed42 \
        --data results/exp09_dpo/pairs.jsonl --seed 42 \
        --out-dir results/exp09_dpo/sft_plus_dpo_seed42
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import _git_commit  # noqa: E402
from train_sft import MetricsLoggingCallback, ThermalSafetyCallback, gpu_temp_c, set_all_seeds  # noqa: E402

logger = get_logger("train_dpo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", required=True,
                     help="HF repo id for raw DPO, or a local checkpoint dir for SFT+DPO")
    ap.add_argument("--data", required=True, help="Path to the preference-pairs JSONL")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-7, help="DPO conventionally uses a much lower LR than SFT")
    ap.add_argument("--beta", type=float, default=0.1, help="KL-penalty strength vs. the frozen reference model")
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-prompt-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on training pairs (debugging)")
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    run_start = datetime.now(timezone.utc)

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"init_model={args.init_model}, seed={args.seed}: {len(rows)} preference pairs from {args.data}")

    dataset = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows
    ])

    tokenizer = AutoTokenizer.from_pretrained(args.init_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO-specific deviation from train_sft.py's fp32+fp16 precision: DPO's
    # loss is computed from log-probabilities that can be large in magnitude
    # (observed logps as low as -500+ in a smoke test), which overflowed
    # fp16's limited range into grad_norm=nan/inf. bf16 has a much wider
    # dynamic range at the same bit width and is the standard fix for this
    # exact failure mode in DPO training; this A5000 (Ampere) supports it
    # natively. SFT arms don't hit this because cross-entropy loss doesn't
    # operate on log-probabilities of this magnitude.
    model = AutoModelForCausalLM.from_pretrained(args.init_model, dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # trl 0.27.0 assumes this internal bookkeeping dict exists (used only to
    # suppress duplicate warnings); transformers==5.3.0's PreTrainedModel no
    # longer initializes it. Harmless to set manually — a version-gap patch,
    # not a training-behavior change.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"
    run_config_path = out_dir / "run_config.json"
    full_config_path = out_dir / "full_config.json"

    import transformers
    import trl as trl_module

    mode = "sft_plus_dpo" if Path(args.init_model).exists() else "raw_dpo"
    run_meta = {
        "mode": mode,
        "init_model": args.init_model,
        "seed": args.seed,
        "data_file": args.data,
        "n_pairs": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.per_device_batch_size * args.grad_accum_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "precision": "fp32_master_amp_bf16",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "trl_version": trl_module.__version__,
        "cuda_version": torch.version.cuda,
        "git_commit": _git_commit(),
        "start_time": run_start.isoformat(),
    }

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer auto-creates a frozen reference copy of the initial weights
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[ThermalSafetyCallback(), MetricsLoggingCallback(metrics_path, run_meta)],
    )

    # Every param, including library defaults left untouched (adam betas/eps,
    # loss_type, label_smoothing, etc.) — same discipline as train_sft.py's
    # full_config.json, written before training starts so it survives a crash.
    full_config_path.write_text(json.dumps({
        "run_meta": run_meta,
        "dpo_config": dpo_config.to_dict(),
        "model_config": model.config.to_dict(),
    }, indent=2, default=str))

    train_result = trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    run_end = datetime.now(timezone.utc)
    run_meta["end_time"] = run_end.isoformat()
    run_meta["duration_seconds"] = (run_end - run_start).total_seconds()
    run_meta["final_train_loss"] = train_result.metrics.get("train_loss")
    run_meta["train_runtime_seconds"] = train_result.metrics.get("train_runtime")
    run_meta["train_samples_per_second"] = train_result.metrics.get("train_samples_per_second")
    with open(run_config_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    full_config = json.loads(full_config_path.read_text())
    full_config["run_meta"] = run_meta
    full_config_path.write_text(json.dumps(full_config, indent=2, default=str))

    logger.info(f"Saved DPO ({mode}) checkpoint + run_config.json + full_config.json + train_metrics.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
