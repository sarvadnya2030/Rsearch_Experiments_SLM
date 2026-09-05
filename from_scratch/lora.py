"""
Educational LoRA (Low-Rank Adaptation) — placeholder.

Will implement: wrap a frozen nn.Linear, add a trainable low-rank update
delta_W = B @ A (B: d x r, A: r x d, r << d) computed at forward time and
added to the frozen layer's output, per theory/01_linear_algebra.md's
low-rank decomposition and theory/08_lora.md (not yet written).
Validate against peft.LoraConfig applied to the same layer.

Not implemented yet — LoRA is explicitly out of scope until after
Experiment 0 per project restrictions.
"""
