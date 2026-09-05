"""
Educational DPO (Direct Preference Optimization) loss — placeholder.

Will implement the closed-form DPO loss from (chosen, rejected) log-probs
under the policy and a frozen reference model:
    L_DPO = -log sigmoid( beta * [ (logpi_c - logref_c) - (logpi_r - logref_r) ] )
matching theory/10_dpo.md (not yet written) and reusing the KL-divergence
intuition from theory/02_probability.md.

Not implemented yet — DPO is explicitly out of scope until after
Experiment 0 per project restrictions.
"""
