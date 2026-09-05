# SLM Reasoning Research

**What this is:** a hypothesis-driven research project on what kind of reasoning supervision actually helps a small language model (SLM) learn to reason — and, in parallel, a hands-on study of how to serve LLMs efficiently in production. Full findings, including a mistake we caught and fixed, are in [`docs/research_log.md`](docs/research_log.md).

**What this isn't:** a leaderboard-chasing project. Every experiment here is designed and logged as prediction → hypothesis → result → interpretation, not just "run script, report number." Post-training method comparisons (SFT/LoRA/QLoRA/DPO/GRPO) are deliberately deferred until we know *what content* to train on — optimizing the wrong axis first would waste the comparison.

## Two tracks

**Track A — Reasoning research.** Central question: does the *style* of reasoning supervision (full chain-of-thought, concise, symbolic, verification-stripped, reflection-stripped, answer-only) change what a small model actually learns — controlled against a randomized-length baseline so "shorter is better" can't masquerade as the real effect. Currently at: Experiment 0, observing Qwen3-0.6B-Base's untrained baseline behavior on GSM8K.

**Track B — Inference engineering.** Benchmarking the same model across serving backends (HF Transformers, vLLM, quantized GGUF/llama.cpp) for throughput/latency/memory, reusing Track A's own evaluation code so every backend is also checked for accuracy parity — not benchmarked on speed alone. Currently at: planning stage, foundations written.

## Results so far

| Run | Examples | Accuracy | Notes |
|---|---|---|---|
| Smoke test (before fix) | 10 | 0% | Extraction bug — see log |
| Smoke test (after fix) | 10 | 40% | |
| Baseline run | 100 | **53%** | 4 distinct failure types identified from manual review |
| Full test set | 1319 | pending | ~6.7h at current per-example rate; Track B batching work targets this |

See [`docs/research_log.md`](docs/research_log.md) for the actual findings behind these numbers — including a real bug we caught by reading raw traces instead of trusting an aggregate metric, and the four qualitatively distinct reasoning-failure types found in manual review.

## Learning philosophy

For every major technique, we maintain two implementations:

1. **Educational** (`from_scratch/`) — minimal, line-by-line-readable code, built to make the math in `theory/` concrete.
2. **Production** (`src/`, using Transformers/PEFT/TRL/vLLM) — what we actually run experiments with.

Math and architecture prerequisites (`theory/01`–`06`, `14`) are written before the code that depends on them, not after.

## Hardware

- Primary: NVIDIA RTX 2070, 8GB VRAM (Track A + B1-B3 experiments).
- Secondary: NVIDIA A500, 16GB VRAM — reserved for larger-scale experiments (model-size scaling, Stage 7).
- Teacher: GPT-OSS 20B API, for generating reasoning traces in Track A Stage 3+ (verified against ground truth before use, never trusted blindly).

## Repository layout

```
theory/            01-06 core math/architecture, 14 inference-serving foundations (populated)
                   07-13 (SFT/LoRA/quantization/DPO/RL/GRPO/verification) reserved until their stage
from_scratch/      educational implementations (placeholders until their stage starts)
src/               production code: data loading, inference, evaluation
scripts/           experiment entry points
configs/           experiment configuration (YAML)
experiments/       per-experiment objective/pipeline docs (Track A: exp00_*, Track B: expB*_*)
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

## Explicitly out of scope right now

SFT, full fine-tuning, LoRA/QLoRA, DPO, GRPO/RLVR, reward models, process supervision, teacher distillation, and all Track B serving backends beyond planning — deferred until Track A's Experiment 0 traces are fully reviewed and a failure taxonomy is locked.
