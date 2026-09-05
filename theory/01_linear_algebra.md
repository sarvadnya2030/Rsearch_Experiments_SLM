# 01 — Linear Algebra

Prerequisite math for everything downstream: attention is matrix multiplication, LoRA is low-rank matrix decomposition, embeddings are vectors.

## Vectors

A vector `x ∈ R^d` is a list of `d` numbers. In our models, every token becomes a vector: `hidden_size = 1024` for Qwen3-0.6B, so each token's hidden state lives in `R^1024`.

## Matrices

A matrix `W ∈ R^(m×n)` maps vectors from `R^n` to `R^m` via `y = Wx`. Linear layers in a transformer are exactly this: `nn.Linear(in_features, out_features)` stores `W ∈ R^(out×in)`.

## Dot product

`a · b = Σ_i a_i b_i`. Measures alignment between two vectors — large when they point the same direction, ~0 when orthogonal, negative when opposed. This is the core primitive of attention: how much does query `q` "match" key `k`? Answer: `q · k`.

## Matrix multiplication

`(AB)_{ij} = Σ_k A_{ik} B_{kj}`. Shapes must chain: `(m×k) @ (k×n) = (m×n)`. Every attention score matrix, every feed-forward layer, every embedding lookup reduces to this operation. Shape-tracking is the single most useful debugging habit for transformer code — get in the habit of writing `# (batch, seq, hidden)` next to every tensor.

## Transpose

`(A^T)_{ij} = A_{ji}`. Used constantly in attention: `Q @ K^T` needs `K^T` so that the shared dimension (`d_k`, the key dimension) lines up for the dot product between every query and every key.

## Norms

- L2 norm: `‖x‖_2 = sqrt(Σ x_i^2)` — the length of a vector. Used in RMSNorm (see `04_neural_networks.md`), gradient clipping, and weight-decay regularization.
- L1 norm: `‖x‖_1 = Σ |x_i|` — used in some sparsity-inducing penalties.

## Rank

The rank of a matrix is the number of linearly independent rows/columns — how much genuinely new information the matrix carries, versus dimensions that are redundant combinations of others. A full-rank `d×d` matrix has rank `d`; a rank-`r` matrix with `r < d` can be written as a product of a `d×r` and an `r×d` matrix — this is the entire mathematical basis for LoRA (see `08_lora.md`, not implemented yet): weight *updates* during fine-tuning are empirically low-rank, so instead of learning a full `ΔW ∈ R^(d×d)`, LoRA learns `ΔW = BA` where `B ∈ R^(d×r)`, `A ∈ R^(r×d)`, `r ≪ d`.

## Low-rank matrices

A rank-`r` decomposition `M = BA` stores `d·r + r·d = 2dr` numbers instead of `d²`. For `d = 1024, r = 8`: `16,384` vs `1,048,576` — a 64x reduction in trainable parameters. This is *why* LoRA is cheap enough to run fine-tuning on an 8GB GPU.

## SVD (intuition only)

Any matrix `A = UΣV^T`, where `Σ` is diagonal with non-negative singular values sorted descending. The singular values tell you how much "energy" each rank-1 component of the matrix carries — truncating to the top `r` singular values gives the *best possible* rank-`r` approximation of `A` (Eckart–Young theorem). Intuition to carry forward: this is the theoretical justification for why low-rank approximations of weight updates (LoRA) don't lose much — most of a weight matrix's "important" behavior concentrates in a few dominant directions.

## Connection to our research

- Attention scores: `QK^T` — matrix multiply + transpose.
- Multi-head attention: splitting `hidden_size` into `num_heads` chunks is literally reshaping one big matmul into several smaller parallel ones (Qwen3-0.6B: `hidden_size=1024`, `num_attention_heads=16` → head_dim would be 64, but Qwen3 configures `head_dim=128` explicitly and uses GQA with `num_key_value_heads=8`).
- LoRA (future): rank-`r` decomposition of weight updates.
- Quantization (future): relies on understanding what precision a matrix's values actually need.
