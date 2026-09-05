# 03 — Calculus & Gradients

How a model actually learns — not used in Experiment 0 (no training happens there), but foundational for every experiment after it.

## Derivatives

`f'(x) = df/dx` — how much `f` changes per unit change in `x`, at a point. For a loss function `L(θ)`, the derivative tells you which direction increases or decreases the loss.

## Partial derivatives

For a multivariate function `L(θ_1, ..., θ_n)`, the partial derivative `∂L/∂θ_i` holds all other parameters fixed and asks how `L` changes with `θ_i` alone. A neural network has millions to billions of parameters — Qwen3-0.6B has ~0.6B — so training needs the *gradient*, the vector of all partials at once.

## Gradients

`∇_θ L = (∂L/∂θ_1, ..., ∂L/∂θ_n)`. Points in the direction of steepest *increase* of `L`. To minimize loss, step in the *negative* gradient direction.

## Chain rule

`d/dx f(g(x)) = f'(g(x)) · g'(x)`. This is the entire mathematical basis of backpropagation: a neural network is a composition of many functions (layers), so the gradient of the loss with respect to an early layer's weights is a product of the local derivatives of every layer between it and the loss, chained together. In vector/matrix form (Jacobians), the same idea scales to millions of parameters — this is what `loss.backward()` computes automatically in PyTorch (autograd).

## Gradient descent

```
θ_new = θ - η * ∇_θ L
```

`η` (eta) is the learning rate — step size. Too large → overshoot/diverge. Too small → painfully slow convergence. In practice we use variants (Adam, AdamW) that adapt the effective step size per-parameter using running estimates of the gradient's mean and variance, but the core update rule above is the concept every optimizer is built on top of.

## Connection to our research

- Experiment 0 does **no training** — it's pure inference (forward pass only, `torch.inference_mode()`), so no gradients are computed. This file is here because Experiment 1+ (SFT) will need it immediately.
- Backpropagation through a transformer means the chain rule is applied through attention, softmax, layernorm, and the MLP block — each has its own local derivative that autograd handles for us, but understanding roughly what's happening is the point of the `from_scratch/` code (not written yet).
- LoRA training only computes gradients for the small `A`, `B` matrices, not the frozen base weights — dramatically cutting the memory needed to store gradients and optimizer state, which is why it's runs on 8GB VRAM.
