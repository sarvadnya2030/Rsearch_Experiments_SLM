# 04 — Neural Networks

The building blocks that transformers are made of.

## Affine transformations

`y = Wx + b` — a linear map plus a bias offset. Every `nn.Linear` layer in a transformer (the Q/K/V projections, the MLP up/down projections, the output head) is an affine transformation. `W` and `b` are the learned parameters.

## Activation functions

Without a nonlinearity between affine layers, stacking them would collapse into one big affine transformation (composition of linear maps is linear) — depth would buy nothing. Activations break that.

Qwen3 uses **SiLU** (`hidden_act: "silu"` in its config), also called Swish:
```
SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
```
Smooth, non-monotonic near 0, unlike ReLU (`max(0, x)`) which has a hard corner. Qwen3's MLP block uses a **gated** variant (SwiGLU-style): `down_proj(SiLU(gate_proj(x)) * up_proj(x))` — two parallel projections, one gates the other elementwise before the down-projection.

## Softmax

Converts a vector of arbitrary real-valued scores ("logits") into a valid probability distribution:
```
softmax(z)_i = e^{z_i} / Σ_j e^{z_j}
```
Used in two places central to our work: (1) turning attention scores into attention *weights* that sum to 1 across the keys being attended to, and (2) turning the final layer's logits (one score per vocabulary token, size 151,936 for Qwen3) into `P(next token)`.

Numerically, softmax is always computed as `softmax(z - max(z))` in practice — subtracting the max before exponentiating prevents overflow, without changing the result (softmax is shift-invariant).

## Cross entropy

For a true label `y` and predicted distribution `p`:
```
CE(y, p) = -log p(y)
```
i.e., negative log-probability the model assigned to the *correct* answer. Minimizing cross-entropy is exactly maximizing log-likelihood (`02_probability.md`). This is the loss used for every next-token prediction in language model training (`06_language_modeling.md`) — not used in Experiment 0 since there's no training, but essential vocabulary for reading the loss curves in every experiment after.

## Backpropagation

The application of the chain rule (`03_calculus_gradients.md`) across an entire network, computed efficiently backward from the loss to the inputs, reusing intermediate results (this is why it's "back"-prop and not naively recomputing each partial derivative from scratch). PyTorch's autograd builds a computation graph during the forward pass and walks it backward on `.backward()`. Not exercised in Experiment 0 (inference only) but this is the mechanism every subsequent experiment trains through.

## Connection to our research

- Qwen3-0.6B-Base architecture facts relevant here: `hidden_size=1024`, `intermediate_size=3072` (the MLP's hidden expansion — 3x, smaller than the classic 4x ratio, likely because of the gated SwiGLU design needing two projections instead of one), 28 transformer layers, SiLU-gated MLP, RMSNorm (see `05_transformers.md`) instead of LayerNorm.
- In Experiment 0, only the forward pass matters: embeddings → 28 transformer blocks → final norm → LM head → softmax → sample/argmax next token. No backward pass, no cross-entropy loss computed against labels, because we're observing generation, not training.
