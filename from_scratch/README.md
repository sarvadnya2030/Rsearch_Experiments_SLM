# from_scratch/ — educational implementations

Purpose: for every major technique we use in this research, maintain a minimal, line-by-line-readable implementation *alongside* the production library implementation (Transformers/PEFT/TRL). The educational version is never meant to be fast or complete — only to make the math in `theory/` concrete and inspectable.

## Learning progression

Implement these in order, each only once we're about to actually use the corresponding technique in an experiment (not ahead of need):

1. **`attention.py`** — manual Q/K/V projection, `QK^T / sqrt(d_k)`, causal mask, softmax, weighted sum over V. Validate against `torch.nn.functional.scaled_dot_product_attention` on random inputs (outputs should match to float precision).
2. **`language_model.py`** — a tiny GPT-style decoder (embeddings → a few manual transformer blocks → LM head) trained on a toy corpus, to make autoregressive next-token prediction concrete end-to-end.
3. **`training.py`** — manual training loop: forward pass, cross-entropy loss, `.backward()`, a hand-rolled SGD step, gradient descent update rule made explicit (no optimizer abstraction).
4. **`lora.py`** — manual low-rank adapter: freeze a linear layer's weight, add `B @ A` (rank `r`) computed and added at the forward pass, train only `A`/`B`. Validate against PEFT's `LoraConfig` applied to the same layer.
5. **`quantization.py`** — manual weight quantization (e.g. int8 symmetric quantization of a linear layer's weights) and comparison of output error against the full-precision layer.
6. **`dpo.py`** — manual DPO loss computation from a pair of (chosen, rejected) log-probabilities and a reference model's log-probabilities, matching the closed-form DPO loss.
7. **`grpo.py`** — manual group-relative advantage computation and policy gradient update over a small group of sampled completions with a verifiable reward.

## Status

All files below are placeholders (docstring only, no implementation) as of Experiment 0. They will be filled in sequentially, only when the corresponding experiment needs them — per project rules, we do not implement SFT/LoRA/DPO/GRPO yet.
