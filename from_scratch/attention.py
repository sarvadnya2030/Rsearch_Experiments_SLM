"""
Educational scaled dot-product attention — placeholder.

Will implement, from raw tensor ops (no nn.MultiheadAttention):
    Q = X @ W_Q ; K = X @ W_K ; V = X @ W_V
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    scores = scores.masked_fill(causal_mask, -inf)
    weights = softmax(scores, dim=-1)
    out = weights @ V

To be validated against torch.nn.functional.scaled_dot_product_attention
on random inputs before being trusted for any downstream use.

See theory/05_transformers.md for the math this implements.
Not implemented yet — placeholder only, per project scope for Experiment 0.
"""
