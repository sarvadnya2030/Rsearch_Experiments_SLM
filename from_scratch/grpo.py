"""
Educational GRPO (Group Relative Policy Optimization) — placeholder.

Will implement: sample a group of completions per prompt, score each with
a verifiable reward (e.g. GSM8K answer correctness), compute
group-relative advantages (reward - group mean, divided by group std),
and a clipped policy-gradient update with a KL penalty to a reference
model, matching theory/12_grpo.md (not yet written).

Not implemented yet — GRPO/RLVR is explicitly out of scope until after
Experiment 0 per project restrictions.
"""
