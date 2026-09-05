# 14 — Inference Serving (Track B foundations)

Everything in `theory/01-06` explains a single forward pass. Production serving is a different problem: many requests arriving over time, sharing a GPU, each wanting low latency *and* the fleet wanting high throughput. This file is the prerequisite reference for Track B (see `research-plan.md` Track B in the project's Obsidian vault) — benchmarking HF Transformers vs vLLM vs quantized llama.cpp serving of Qwen3-0.6B.

## KV-cache

During autoregressive generation, computing attention for token `t` needs the `K` and `V` vectors of every prior token (`05_transformers.md`). Recomputing those from scratch at every new token would be `O(seq_len²)` work per generated token. Instead, `K`/`V` for each past token are computed once and cached; each new step only computes `K`/`V` for the *new* token and appends it. This is what `use_cache=True` (Qwen3's default, see its `config.json`) does under the hood in `model.generate()`.

Cost: the KV-cache itself consumes GPU memory proportional to `2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_element` (the `2` is for K and V). For Qwen3-0.6B with GQA (`num_key_value_heads=8`, not 16 — see `05_transformers.md`), the KV-cache is already ~2x smaller than it would be without GQA, at fixed batch size and sequence length. This is the memory that limits how many concurrent requests (or how large a batch) an 8GB card can hold.

## Static vs continuous (dynamic) batching

- **Static batching**: group a fixed batch of requests, run them together until *all* finish, discard padding. Whichever request in the batch generates the longest response holds up every other request's GPU slot (padding wastes compute/memory) — this is what a naive `model.generate(**batched_inputs)` call does.
- **Continuous (dynamic) batching**: as soon as any request in a running batch finishes, immediately splice in a new waiting request to take its slot, without waiting for the whole batch to drain. This is the core serving-throughput idea behind vLLM and similar systems — GPU utilization stays high because slots are never idle waiting for the slowest request in the group.

## PagedAttention (vLLM's core idea)

Naive KV-cache allocation reserves a contiguous memory block sized for the *maximum possible* sequence length per request, even if the actual output is much shorter — wasteful, and it fragments memory so batch sizes must be conservative. PagedAttention (vLLM) borrows the OS-paging idea: KV-cache is allocated in fixed-size non-contiguous "pages," referenced via a per-sequence page table, so memory is only used for tokens that actually exist. This lets more concurrent sequences fit in the same VRAM, directly increasing achievable batch size / throughput without changing model quality.

## Quantization for serving (distinct from training-time quantization)

`theory/09_quantization.md` (training-time, e.g. QLoRA) will cover quantizing a frozen base model so a LoRA adapter can be trained on top of it in low memory. Serving-time quantization has a different goal — pure inference speed/memory, no training involved:
- **GGUF**: a file format (used by llama.cpp/Ollama) storing quantized weights (common levels: Q4_K_M, Q8_0, etc. — the number is roughly bits per weight) plus metadata, optimized for fast CPU/GPU-hybrid loading.
- **AWQ / GPTQ**: post-training quantization *algorithms* that choose quantization scales more carefully than naive rounding (e.g. AWQ protects the small subset of "salient" weight channels that matter disproportionately for output quality), typically better accuracy-per-bit than naive round-to-nearest at the same bit width.

The tradeoff to measure empirically in Track B: lower bit-width -> smaller memory footprint and higher throughput, but some accuracy loss — and that loss must be measured on our actual task (GSM8K exact-match), not assumed from a generic benchmark, since sensitivity varies by task.

## Speculative decoding (intuition only, not implemented in Track B initially)

Generate `k` candidate future tokens cheaply with a small "draft" model, then verify all `k` in a single forward pass of the real (target) model — accept the longest correct prefix, reject and resample the rest. Because verifying `k` tokens in one parallel forward pass is much cheaper than generating them one at a time autoregressively, this can meaningfully speed up generation when the draft model's guesses are often right, at zero cost to output quality (the target model's distribution is what's ultimately sampled from). Relevant here since Qwen3-0.6B is itself small enough to plausibly serve as a *draft* model for a larger Qwen3 checkpoint in some future setup — noted for later, not benchmarked in Track B's first pass.

## Throughput vs. latency — the metrics that actually matter

- **Time-to-first-token (TTFT)**: latency until the first output token appears — dominated by the prompt's prefill pass (one forward pass over the whole prompt, computing all its KV-cache entries at once).
- **Inter-token latency (ITL)** / time-per-output-token: latency for each subsequent token — one incremental forward step using the cached K/V.
- **Aggregate throughput (tok/s)**: total tokens produced across all concurrent requests, per second — what continuous batching and PagedAttention are optimizing for. Note this is a *fleet* metric, distinct from any single request's latency; a system can have high aggregate throughput while an individual request still waits behind others in the queue.

Track B's benchmark (B2-B6 in `research-plan.md`) measures all three, not just tok/s alone, because a serving choice that maximizes throughput at the cost of per-request latency is a real tradeoff, not a strict improvement.

## Connection to our research

Track B reuses Track A's exact evaluation code (`src/evaluation/answer_extraction.py`, `metrics.py`) to check that a faster/quantized backend still produces GSM8K-correct answers at (approximately) the same rate as the fp16 HF baseline — speed without an accuracy-parity check is not a valid comparison.
