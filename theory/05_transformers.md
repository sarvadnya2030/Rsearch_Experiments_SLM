# 05 — Transformers

The architecture of Qwen3-0.6B-Base, mapped to its actual config values.

## Embeddings

Each token id (integer in `[0, vocab_size)`) is looked up in an embedding matrix `E ∈ R^(vocab_size × hidden_size)` to get its vector representation. Qwen3-0.6B: `vocab_size=151936`, `hidden_size=1024` → embedding table is `151936 × 1024 ≈ 155M` parameters. `tie_word_embeddings: true` in its config means the same matrix is reused (transposed) as the final LM head that projects hidden states back to vocabulary logits — halves the parameter count spent on input/output projections, common for smaller models.

## Q/K/V projections

```
Q = X W_Q      K = X W_K      V = X W_V
```
`X ∈ R^(seq_len × hidden_size)` is the sequence of token representations at this layer. `W_Q, W_K, W_V` are learned projection matrices. Intuition: `Q` (query) is "what am I looking for," `K` (key) is "what do I offer," `V` (value) is "what do I actually contribute if attended to."

Qwen3 uses **Grouped Query Attention (GQA)**: `num_attention_heads=16` but `num_key_value_heads=8` — queries have 16 heads, but keys/values only 8, with each KV head shared by 2 query heads. This roughly halves the KV-cache memory during generation (a big deal for our 8GB VRAM) at a small quality cost versus full multi-head attention. `head_dim=128` is set explicitly (not derived as `hidden_size/num_heads=64`), a deliberate architecture choice in Qwen3.

## Scaled dot-product attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```
- `QK^T`: dot product of every query against every key → raw similarity scores, shape `(seq_len, seq_len)`.
- `/ sqrt(d_k)`: scaling factor (`d_k = head_dim = 128` here). Without it, dot products of high-dimensional vectors grow large in magnitude, pushing softmax into a saturated regime with near-zero gradients. Dividing by `sqrt(d_k)` keeps the variance of the scores roughly constant regardless of dimension.
- `softmax(...)`: converts scores into a distribution over "how much attention to pay to each position."
- `... V`: weighted sum of value vectors according to those attention weights.

## Causal masking

For autoregressive generation, token `t` must not see tokens `t+1, t+2, ...` (it hasn't "happened" yet, and letting the model see the future during training would make the task trivial and useless). Implemented by setting the upper-triangular part of the `QK^T` score matrix to `-∞` before the softmax, so those positions get exactly 0 probability. Qwen3-0.6B has `sliding_window: null` — full causal attention over the whole context, no windowing, up to `max_position_embeddings=32768`.

## Multi-head attention

Instead of one attention computation over the full `hidden_size`, split into `num_attention_heads` parallel smaller attention computations, each on a `head_dim`-sized slice, then concatenate and project back down. Lets different heads specialize (some heads track syntax, others long-range dependencies, etc. — empirically observed, not designed-in).

## Residual connections

Each sublayer's output is added back to its input: `x = x + Sublayer(x)`, rather than `x = Sublayer(x)`. Critical for training stability in deep networks (28 layers here) — it gives gradients a direct path backward through the addition, unaffected by whatever the sublayer's own gradient looks like, which mitigates vanishing gradients.

## Normalization

Qwen3 uses **RMSNorm** (`rms_norm_eps: 1e-06`), not the classic LayerNorm. RMSNorm rescales each vector by its root-mean-square rather than centering (subtracting mean) *and* rescaling like LayerNorm does:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g
```
`g` is a learned per-dimension gain. Cheaper than LayerNorm (no mean subtraction) and works about as well in practice for transformers at this scale.

## MLP blocks

Per-layer feed-forward network applied identically (position-wise) to each token's hidden state independently. Qwen3's is a gated SwiGLU-style MLP (see `04_neural_networks.md`): expands `hidden_size=1024 → intermediate_size=3072` via two parallel projections, applies SiLU gating, projects back down to 1024.

## Positional encoding — RoPE

Attention itself has no notion of token order — `QK^T` is permutation-invariant to how tokens are arranged unless position information is injected. Qwen3 uses **RoPE** (Rotary Position Embedding, `rope_theta: 1000000`): instead of adding a positional vector to the embedding, RoPE *rotates* each Q/K vector's dimensions by an angle proportional to its position, in pairs of dimensions treated as 2D planes. The key property: the dot product `q_m · k_n` after rotation depends only on the *relative* distance `m - n`, not absolute positions — this generalizes better to longer contexts than learned absolute position embeddings, and is why Qwen3 supports a 32,768-token context.

## Connection to our research

Every architectural choice above (GQA, RoPE, RMSNorm, SwiGLU, tied embeddings) is standard for the current generation of efficient small models — Qwen3-0.6B is representative of the class of models this research studies. Understanding these mechanics matters most once we start asking *why* a model reasons the way it does, or start modifying components for research (e.g. attention-pattern analysis, later intervention experiments in theory file 13).
