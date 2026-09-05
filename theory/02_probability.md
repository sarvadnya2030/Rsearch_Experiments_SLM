# 02 — Probability

A language model *is* a probability distribution over token sequences. Everything here explains what that means precisely.

## Probability distributions

A discrete distribution over a vocabulary `V` assigns each token `v` a probability `p(v) ≥ 0` with `Σ_v p(v) = 1`. The model's final layer produces a distribution over the ~152k tokens in Qwen3's vocabulary for "what comes next."

## Conditional probability

`P(A|B) = P(A,B) / P(B)` — probability of `A` given that `B` already happened. Every next-token prediction is a conditional probability: `P(x_t | x_<t)`, "probability of this token given everything before it."

## Expectation

`E[X] = Σ_x x · P(x)`. The average outcome weighted by probability. Training losses are expectations over a dataset: `E_{(x,y)~D}[loss(x,y)]` — we can't compute the true expectation (infinite/unknown data distribution), so we approximate it with a batch average (Monte Carlo estimate).

## Likelihood

Given a fixed model with parameters `θ` and observed data `x`, the *likelihood* is `P(x | θ)` — viewed as a function of `θ`, not `x`. Training = finding `θ` that makes the observed data as likely as possible ("maximum likelihood estimation").

## Log-likelihood

`log P(x | θ)`. We work in log-space for two reasons: (1) products of many small probabilities underflow to 0 in floating point, but sums of logs don't; (2) `log` turns the product of per-token probabilities in an autoregressive model into a sum, which is what makes the loss decompose per-token (see `06_language_modeling.md`).

## Entropy

`H(p) = -Σ_v p(v) log p(v)`. Measures uncertainty in a distribution. Uniform distribution over `V` tokens → maximum entropy `log|V|`. A distribution that puts all mass on one token → entropy 0. Relevant to reasoning research: a model's per-token entropy during generation is a cheap proxy for "how uncertain/confident was it here" — useful later for analyzing *where* reasoning goes wrong.

## KL divergence

`KL(p‖q) = Σ_v p(v) log(p(v)/q(v))`. Measures how different distribution `q` is from a reference `p`. Always `≥ 0`, and `0` only when `p = q`. Not symmetric: `KL(p‖q) ≠ KL(q‖p)`.

This becomes central once we reach DPO/RL (not implemented yet): DPO's loss is derived from a KL-constrained reward maximization — the policy is kept close (low KL) to a reference model while being pushed toward preferred outputs. GRPO similarly uses a KL penalty term to prevent the policy from drifting too far from the reference during RL updates.

## Sampling

Given a distribution over next tokens, "sampling" means drawing a token according to those probabilities rather than always taking the max (greedy decoding). Common controls:
- **Temperature** `T`: rescale logits before softmax as `logits / T`. `T < 1` sharpens the distribution (more greedy-like, less diverse); `T > 1` flattens it (more random). `T → 0` recovers greedy argmax.
- **Top-p (nucleus) sampling**: keep the smallest set of tokens whose cumulative probability exceeds `p`, renormalize, sample from that truncated set. Prevents sampling from the long unreliable tail of low-probability tokens.
- **do_sample=False**: pure greedy decoding, always pick `argmax P(x_t | x_<t)`. Deterministic — this is what Experiment 0 uses, for reproducibility.

## Connection to our research

- Experiment 0 uses `do_sample=False` (greedy) specifically so results are reproducible run-to-run — no sampling randomness confounding our observation of "natural" reasoning behavior.
- Cross-entropy loss (`06_language_modeling.md`, `04_neural_networks.md`) is literally negative log-likelihood.
- KL divergence reappears verbatim in the math of DPO and GRPO (theory files 10, 12 — not written yet).
