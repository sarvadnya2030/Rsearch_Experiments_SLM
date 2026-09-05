# Experiment 0 — Baseline Reasoning Observation

## Objective

Measure how Qwen3-0.6B-Base naturally reasons on GSM8K, with no training or fine-tuning applied. This is the reference point every later post-training intervention (SFT, LoRA/QLoRA, DPO, GRPO, ...) will be compared against.

## Pipeline

```
GSM8K question (test split)
  -> completion-style prompt ("Question: ...\nAnswer:")
  -> Qwen3-0.6B-Base.generate() [greedy, fp16, CUDA]
  -> generated response (raw text preserved in full)
  -> extract_predicted_answer() (deterministic regex, no LLM judge)
  -> compare against reference "#### N" answer
  -> per-example JSONL record (raw response + full metadata)
  -> aggregate metrics.json + errors.jsonl
```

## Why completion-style prompting, not chat template

Qwen3-0.6B-Base's tokenizer_config.json ships a full Qwen3 chat_template (im_start/im_end tags, tool-calling support), inherited from the Qwen3 family's tooling — but this Base checkpoint was never instruction-tuned to follow it. Using `apply_chat_template` here would prompt the model in a format it has no training signal for. Instead we use a minimal zero-shot completion prompt, consistent with standard base-model GSM8K evaluation methodology in the literature.

## Key config decisions (see configs/exp00_baseline.yaml)

- `do_sample: false` (greedy) — deterministic, reproducible baseline; no sampling randomness confounding the "natural behavior" observation.
- `dtype: float16` — model's native dtype is bfloat16 (per its config.json), but fp16 is used for broader RTX 2070 kernel compatibility and to fit comfortably in the currently-available VRAM.
- `max_new_tokens: 512` — generous for multi-step GSM8K reasoning, smaller than the model's own generation_config default (2048) to keep smoke/partial runs fast.
- `batch_size: 1` — conservative starting point given ~2.6GB VRAM was free at last check (Ollama was holding ~5.5GB of the 8GB card). Raise only after confirming headroom.

## Commands

```bash
# 1. Smoke test — 10 examples
python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 10

# 2. Partial run — 100 examples
python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 100

# 3. Full GSM8K test set — 1319 examples
python scripts/run_baseline.py --config configs/exp00_baseline.yaml
```

Each run creates a new `results/exp00_baseline/run_YYYYMMDD_HHMMSS/` directory — previous runs are never overwritten.

## What NOT to conclude from this experiment

- Response-length buckets in metrics.json are descriptive only — do not infer that longer/shorter responses *cause* correctness or incorrectness.
- `errors.jsonl` entries are all tagged `error_category: "unclassified"` on purpose — failure taxonomy will be designed after a human manually inspects a sample of real failures, not auto-assigned.
