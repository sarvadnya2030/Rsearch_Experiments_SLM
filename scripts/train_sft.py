"""
Stage 5 — SFT training for one reasoning-format arm, one seed.

Full fine-tuning (NOT LoRA/QLoRA) of Qwen3-0.6B-Base on a single arm's
data. Full fine-tuning specifically because Stage 9 (training method
ablation: SFT vs LoRA vs QLoRA vs DPO vs RLVR) is a separate, later
question — using LoRA now would confound "does this format help" with
"does this training method help," which the plan's staging is designed
to keep apart.

Prompt format matches every other script in this repo: plain completion
style ("Question: ...\nAnswer: ..."), NOT apply_chat_template — the base
model was never instruction-tuned to follow its inherited chat template
(established in Stage 2).

Loss is computed ONLY on the answer tokens (prompt tokens masked to -100)
— standard SFT practice, so the model isn't asked to predict the
question it's given.

Usage:
    python scripts/train_sft.py --arm A --data results/exp05_arms_acd/arm_a_full_n7435.jsonl \
        --seed 42 --out-dir results/exp05_sft/arm_A_seed42
"""

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import _git_commit  # noqa: E402

logger = get_logger("train_sft")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


def gpu_temp_c() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip())
    except Exception:
        return None


class ThermalSafetyCallback(TrainerCallback):
    """Same pattern as self_consistency_batched.py's cooldown check — pause
    training above pause_above_c, resume once below resume_below_c. Training
    sweeps here run for hours unattended on the same RTX 2070 that hit 87C
    with active throttling during Stage 2.5's sustained inference runs."""

    def __init__(self, pause_above_c: int = 80, resume_below_c: int = 65, poll_seconds: int = 15):
        self.pause_above_c = pause_above_c
        self.resume_below_c = resume_below_c
        self.poll_seconds = poll_seconds

    def on_step_end(self, args, state, control, **kwargs):
        temp = gpu_temp_c()
        if temp is None or temp < self.pause_above_c:
            return
        print(f"GPU at {temp}C >= {self.pause_above_c}C, pausing training until it drops below {self.resume_below_c}C...")
        while True:
            time.sleep(self.poll_seconds)
            temp = gpu_temp_c()
            if temp is not None and temp < self.resume_below_c:
                print(f"GPU cooled to {temp}C, resuming training.")
                break


class MetricsLoggingCallback(TrainerCallback):
    """Append every on_log event (loss, learning_rate, grad_norm, epoch —
    whatever Trainer's own logging surfaces) to a JSONL file, one line per
    log event, plus wall-clock time and GPU temp/memory at that moment.
    This is the raw data for loss-curve plots and cross-run comparison —
    report_to=[] disables wandb/tensorboard, so without this callback
    nothing but the final summary line would be recorded anywhere."""

    def __init__(self, metrics_path: Path, run_meta: dict):
        self.metrics_path = metrics_path
        self.run_meta = run_meta
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w") as f:
            f.write(json.dumps({"event": "run_start", **run_meta}) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {
            "event": "log",
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_temp_c": gpu_temp_c(),
            **logs,
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"event": "run_end", "total_steps": state.global_step}) + "\n")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReasoningSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 1024):
        self.examples = []
        for row in rows:
            prompt = f"Question: {row['question']}\nAnswer:"
            target = " " + row["reasoning_trace"].strip() + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            if len(input_ids) > max_length:
                # Truncate from the left of the target overflow, keep the full prompt —
                # losing the tail of an overlong trace is better than losing the question.
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="Arm label, e.g. A, B, C, D, E, F, G (for logging/output naming only)")
    ap.add_argument("--data", required=True, help="Path to the arm's JSONL file")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on training examples (debugging)")
    ap.add_argument("--logging-steps", type=int, default=5, help="Optimizer steps between metric log points (for loss-curve resolution)")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    run_start = datetime.now(timezone.utc)

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"Arm {args.arm}, seed {args.seed}: {len(rows)} training examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Standard full-fine-tuning setup: fp32 master weights + TrainingArguments
    # (fp16=True) mixed-precision autocast + plain fp32 AdamW. This is the
    # numerically clean version — no 8-bit-quantized optimizer states, no
    # fp16-only weights. Only used on hardware with enough VRAM (16GB A5000);
    # the earlier 8-bit-optimizer/fp16-weights variant existed purely to fit
    # this same training onto an 8GB RTX 2070 and traded some numerical
    # precision for that. Kept as a comment here, not a config flag, because
    # every arm in one ablation sweep should use the SAME setup — mixing
    # precision/optimizer choices across arms would confound the format
    # comparison with a training-method difference.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = False  # KV cache is a generation-time thing; wastes memory during training
    model.gradient_checkpointing_enable()

    dataset = ReasoningSFTDataset(rows, tokenizer, max_length=args.max_length)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers

    run_meta = {
        "arm": args.arm,
        "seed": args.seed,
        "data_file": args.data,
        "n_examples": len(rows),
        "model_name": MODEL_NAME,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.per_device_batch_size * args.grad_accum_steps,
        "max_length": args.max_length,
        "optimizer": "adamw_torch",
        "precision": "fp32_master_amp_fp16",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "git_commit": _git_commit(),
        "start_time": run_start.isoformat(),
    }
    metrics_path = out_dir / "train_metrics.jsonl"
    run_config_path = out_dir / "run_config.json"
    full_config_path = out_dir / "full_config.json"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        fp16=True,  # standard AMP mixed precision on top of fp32 master weights
        logging_steps=args.logging_steps,
        save_strategy="no",  # only save the final model, not per-step checkpoints
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        optim="adamw_torch",  # plain fp32 AdamW — no 8-bit quantization of optimizer states
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[ThermalSafetyCallback(), MetricsLoggingCallback(metrics_path, run_meta)],
    )

    # Every param, not just the ones passed on the CLI — TrainingArguments.to_dict()
    # and model.config.to_dict() include library defaults (weight decay, LR scheduler
    # type, warmup ratio, optimizer betas/epsilon, etc.) that would otherwise be silently
    # unlogged. Written before training starts so it's captured even if a run crashes mid-way.
    full_config_path.write_text(json.dumps({
        "run_meta": run_meta,
        "training_arguments": training_args.to_dict(),
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

    logger.info(f"Saved fine-tuned model + run_config.json + full_config.json + train_metrics.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
