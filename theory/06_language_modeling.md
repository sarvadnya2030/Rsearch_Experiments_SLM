# 06 — Language Modeling

How the pieces from files 01–05 combine into "predict the next token," and why Experiment 0 is built the way it is.

## Autoregressive factorization

The joint probability of a token sequence factors into a product of conditionals, by the chain rule of probability (not calculus chain rule — same name, different concept):
```
P(x_1, ..., x_T) = Π_t P(x_t | x_<t)
```
A language model never directly models the joint distribution over a whole sequence — it only ever needs to model `P(x_t | x_<t)`, one token at a time, conditioned on everything before it. Generation is just repeatedly sampling (or arg-maxing) from this conditional and feeding the result back in as the new "before."

## Next-token prediction

Concretely, at each position `t`, the model produces a distribution over the vocabulary and we ask: how much probability mass did it put on the actual next token in the training data (during training) or which token do we emit (during inference)? This single objective — next-token prediction — is the entirety of how models like Qwen3 are pretrained, no separate "reasoning" objective is used at pretraining time. Any reasoning ability observed in Experiment 0 is an emergent property of this objective plus scale and data, not something explicitly trained in for a base model.

## Teacher forcing

During *training*, when computing the loss at position `t`, the model is fed the true previous tokens `x_<t` from the dataset (not its own possibly-wrong previous predictions). This decouples the per-position loss computation — every position's loss can be computed in one parallel forward pass rather than sequentially. Not relevant to Experiment 0 (no training there), but essential vocabulary for every training experiment after it, and it explains the well-known train/inference mismatch ("exposure bias"): at inference the model must condition on its *own* generated tokens, which may contain errors teacher forcing never let it see during training.

## Causal LM loss

```
L_CE = -Σ_t log P_θ(y_t | x, y_<t)
```
Sum (or mean) of negative log-probabilities the model assigns to the actual next token, at every position. This is cross-entropy (`04_neural_networks.md`) applied per-token and summed across the sequence. `θ` denotes all model parameters. Again: not computed in Experiment 0, since there are no gradients or training there — but this is exactly the loss SFT (Experiment 1) will minimize, just restricted to computing the loss only over the assistant's response tokens, not the prompt.

## Connection to our research — why Experiment 0 looks the way it does

Experiment 0 does a **pure forward-pass generation loop**: feed a GSM8K question as a prompt, autoregressively sample/argmax tokens using the conditional `P(x_t | x_<t)` above until an EOS token or max length, then extract and check the final numeric answer. No loss is computed, no gradients, no backward pass — we are purely observing what the pretraining objective (next-token prediction over internet + curated text) produced, in terms of emergent multi-step arithmetic reasoning. This baseline is the reference point against which every future post-training intervention (SFT, DPO, GRPO, etc.) will be measured.

Since Qwen3-0.6B-Base's tokenizer_config carries a chat template but the model itself was never instruction-tuned, Experiment 0 prompts it with **plain completion-style text** (e.g. `"Question: ...\nAnswer:"`), not `apply_chat_template`. This matches how base models are conventionally evaluated in the literature (e.g. the original GSM8K/GPT-3 few-shot evaluation setup) and avoids feeding the model a format it was never trained to interpret.
