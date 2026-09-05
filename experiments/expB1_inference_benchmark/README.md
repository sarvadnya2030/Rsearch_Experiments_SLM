# Experiment B1 — Inference Serving Benchmark (Track B)

Parallel track to `experiments/exp00_baseline` (Track A). See `theory/14_inference_serving.md` for the prerequisite concepts this experiment exercises.

## Objective

Characterize the throughput / latency / memory / accuracy tradeoff of serving Qwen3-0.6B-Base across multiple inference backends, on the same hardware (RTX 2070, 8GB) and the same evaluation set (a GSM8K subset), reusing Track A's exact evaluation code (`src/evaluation/answer_extraction.py`, `metrics.py`) so every backend is checked for accuracy parity, not benchmarked on speed alone.

## Backends to compare

1. **HF Transformers baseline** (batch_size=1) — already measured via Track A's Experiment 0 runs: 25.65 tok/s avg, 18.2s/example avg on the 100-example run.
2. **HF Transformers, real batching** — `src/inference/generate.py` currently only implements batch_size=1 despite the config field existing; needs padding + attention_mask handling to actually batch.
3. **vLLM** — PagedAttention + continuous batching.
4. **Quantized GGUF via llama.cpp/Ollama** — INT4/INT8 quantized weights.

## Metrics (per backend, per config)

- Time-to-first-token (TTFT)
- Inter-token latency (ITL) / tok/s per request
- Aggregate throughput (tok/s across all concurrent requests, where applicable)
- Peak GPU memory used
- GSM8K accuracy on the same fixed subset (must match Track A's extraction/scoring logic exactly — no separate scoring implementation)

## Status

Planning stage — not yet implemented. Foundations (`theory/14_inference_serving.md`) written 2026-09-05. Implementation order: B2 (already have data) -> B3 (real batching on existing HF backend) -> B4 (vLLM) -> B5 (GGUF/llama.cpp) -> B6 (comparison write-up).

## Why this is a separate experiment from exp00_baseline

Track A (exp00_baseline) asks "how does the model reason." Track B asks "how do you serve it efficiently." Keeping them separate keeps each experiment's config/results honest about what's actually being varied — Track A varies nothing about serving (always HF, batch_size=1, fp16) specifically so its accuracy numbers aren't confounded by serving-backend differences.
