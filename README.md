# SLM Reasoning Research

**What this is:** a hypothesis-driven research project on what kind of reasoning supervision actually helps a small language model (SLM) learn to reason. Every experiment is designed and logged as prediction → hypothesis → result → interpretation, not just "run script, report number." Full findings, including a mistake we caught and fixed, are in [`docs/research_log.md`](docs/research_log.md).

**What this isn't:** a leaderboard-chasing project, and not a roadmap — this README describes what's built and measured so far, not what's planned next.

Currently: Experiment 0 — observing Qwen3-0.6B-Base's untrained baseline behavior on GSM8K (no training yet).

## Results so far

| Run | Examples | Accuracy | Notes |
|---|---|---|---|
| Smoke test (before fix) | 10 | 0% | Extraction bug — see log |
| Smoke test (after fix) | 10 | 40% | |
| Baseline run | 100 | 53% | 4 distinct failure types identified from manual review |
| Full test set | **1319** | **52.16%** (95% CI: 49.5–54.9%) | Anchor baseline for Qwen3-0.6B-Base zero-shot on GSM8K |

See [`docs/research_log.md`](docs/research_log.md) for the actual findings behind these numbers — including a real bug we caught by reading raw traces instead of trusting an aggregate metric, and the four qualitatively distinct reasoning-failure types found in manual review.

## Learning philosophy

For every major technique, we maintain two implementations:

1. **Educational** (`from_scratch/`) — minimal, line-by-line-readable code, built to make the math in `theory/` concrete.
2. **Production** (`src/`, using Transformers/PEFT/TRL/vLLM) — what we actually run experiments with.

Math and architecture prerequisites (`theory/01`–`06`, `14`) are written before the code that depends on them, not after.

## Hardware

- Primary: NVIDIA RTX 2070, 8GB VRAM.
- Secondary: NVIDIA A500, 16GB VRAM — for larger-scale experiments, not yet used.
- Teacher: GPT-OSS 20B API — not yet used.

## Repository layout

```
theory/            01-06 core math/architecture, 14 inference-serving foundations (populated)
                   07-13 (SFT/LoRA/quantization/DPO/RL/GRPO/verification) reserved until their stage
from_scratch/      educational implementations (placeholders until their stage starts)
src/               production code: data loading, inference, evaluation
scripts/           experiment entry points
configs/           experiment configuration (YAML)
experiments/       per-experiment objective/pipeline docs
docs/              public research log — mirrors key findings (source of truth for narrative)
results/           per-run raw outputs, never overwritten (gitignored — not committed)
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

# Full GSM8K test set — 1319 examples (only after inspecting smoke/partial results)
python scripts/run_baseline.py --config configs/exp00_baseline.yaml
```

## Not implemented yet

SFT, full fine-tuning, LoRA/QLoRA, DPO, GRPO/RLVR, reward models, process supervision, teacher distillation, and inference-serving benchmarks beyond `theory/14`.
