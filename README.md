# SLM Reasoning Research

## Research objective

How do different post-training methods and parameter-update mechanisms (SFT, LoRA/QLoRA, DPO, GRPO/RLVR, process supervision, ...) change reasoning behavior in small language models? This is a learning-first research codebase: the goal is to understand the mathematics, algorithms, and implementation behind each technique, not to chase benchmark leaderboard numbers.

## Learning philosophy

For every major technique, we maintain two implementations:

1. **Educational** (`from_scratch/`) — minimal, line-by-line-readable code, built to make the math in `theory/` concrete.
2. **Production** (`src/`, using Transformers/PEFT/TRL) — what we actually run experiments with.

We implement the mathematical prerequisites (`theory/01`–`06`) and Experiment 0 first, and only build forward from there once the baseline is understood.

## Current stage

**Experiment 0 — baseline reasoning observation.** No training happens yet. We are purely observing how Qwen3-0.6B-Base reasons on GSM8K out of the box, to establish the reference point for every future intervention.

## Current hardware

- Primary: NVIDIA RTX 2070, 8GB VRAM. **Note:** at last check, Ollama was holding ~5.5GB of this card, leaving only ~2.6GB free — stop other GPU processes before running full evaluation if you hit OOM.
- Secondary (future): NVIDIA A500, 16GB VRAM — for larger experiments later, not used by Experiment 0.

## Future teacher

A GPT-OSS 20B API will eventually be used as a teacher/reasoning generator for later experiments (e.g. distillation, preference data generation). Not used in Experiment 0.

## Current experiment

Qwen3-0.6B-Base + GSM8K test set (1319 examples), greedy decoding, completion-style prompting (not chat template — see `experiments/exp00_baseline/README.md` for why).

## Repository layout

```
theory/           mathematical foundations (01_linear_algebra ... 06_language_modeling populated;
                   07_sft ... 13_reasoning_and_verification reserved for later)
from_scratch/      educational implementations (placeholders for now)
src/               production code: data loading, inference, evaluation
scripts/           experiment entry points
configs/           experiment configuration (YAML)
experiments/       per-experiment objective/pipeline docs
results/           per-run outputs, never overwritten (gitignored)
tests/             unit tests for evaluation logic
```

## Setup

```bash
pip install -r requirements.txt
```

## Commands

```bash
# Run unit tests
pytest tests/ -v

# Smoke test — 10 examples
python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 10

# Partial run — 100 examples
python scripts/run_baseline.py --config configs/exp00_baseline.yaml --limit 100

# Full GSM8K test set — 1319 examples (only after inspecting the smoke/partial results)
python scripts/run_baseline.py --config configs/exp00_baseline.yaml
```

## Explicitly out of scope right now

SFT, full fine-tuning, LoRA/QLoRA, DPO, GRPO/RLVR, reward models, process supervision, teacher distillation — all deferred until Experiment 0's baseline traces have been manually reviewed.
