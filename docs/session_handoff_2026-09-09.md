# Session Handoff — 2026-09-09

Full context dump of this session's work, for a cold-started Claude Code
session (e.g. on the ZBook) to pick up without re-deriving anything.
Complements `docs/research_log.md` (public findings mirror) and the
Obsidian vault's `project-log.md`/`research-plan.md` — this file is the
narrative of *this specific session*, including decisions and their
reasoning, not just the polished findings.

## What happened this session, in order

1. **Resumed Stage 3 teacher generation** (GPT-OSS-20B traces for the
   full 7473-example GSM8K train split) — it had been running overnight
   via `scripts/generate_teacher_traces.py --resume`. Restarted it once
   after the process died silently (caught because the progress monitor's
   ticks flatlined at 0/hr).

2. **Built an automated retry + escalating manual-fix pipeline**
   (`scripts/retry_wrong_teacher_traces.py`): every first-pass API
   failure gets 2 fresh attempts per pass, parallelized 8-way, retried
   again on every subsequent pass. Once an example's cumulative attempts
   hit 3, it's flagged `needs_manual_fix` and solved from scratch (by
   Claude, via dispatched subagents, one per newly-flagged batch) instead
   of burning more API budget — never copying the dataset's own
   `reference_solution` (contamination check).

3. **Found and independently re-verified 38 genuine GSM8K
   reference-solution bugs.** When a from-scratch derivation couldn't
   honestly reach `reference_answer`, it was marked disputed
   (`exclude_from_training: true`) rather than forced. A separate
   fresh-agent audit re-derived all disputed cases from scratch (not
   trusting the original disputing agent's write-up) and confirmed
   30/33 as objective errors, 3 as merely uncertain/ambiguous, 0 as
   wrongly excluded. The list grew to 38 as generation finished the
   last ~700 examples. Full list and examples are in
   `docs/research_log.md`'s "Finding 10" entry.

4. **Teacher generation completed**: all 7473 examples generated.
   One example (id 3148) was silently skipped after 3 API failures by
   the original script (a real gap — caught by diffing generated
   example_ids against 0-7472) and solved manually.

5. **Built all 7 Stage 5 reasoning-format arms at final scale**:
   - `scripts/build_final_teacher_pool.py` — merges the main teacher
     file + retry pool into one deduplicated, verified pool (7435/7473
     usable) before deriving arms from it.
   - `scripts/prepare_stage5_arms_acd.py` — Arms A (full, 7435), C
     (no-verification, 7426), D (no-reflection, 7418).
   - `scripts/prepare_stage5_arms_be.py` — Arms B (concise) and E
     (symbolic), derived from GSM8K's own reference solutions (no
     teacher needed).
   - `scripts/prepare_stage5_arms_fg.py` — Arm F (answer-only) and Arm
     G (random-shortened control, length-matched to Arm B per example).
     **Also fixed a consistency bug**: Arms B and E were originally
     derived straight from GSM8K's reference solutions with no
     independent verification, so they silently carried some of the 38
     bad labels. Refiltered both against the same 38-id exclusion list.

   Final arm file sizes (all in `results/exp05_arms_acd/` or
   `results/exp05_arms_be/`):
   - A (full): `arm_a_full_n7435.jsonl` — 7435
   - B (concise): `arm_b_concise_n7435_filtered.jsonl` — 7435
   - C (no verification): `arm_c_no_verification_n7435.jsonl` — 7426 usable
   - D (no reflection): `arm_d_no_reflection_n7435.jsonl` — 7418 usable
   - E (symbolic): `arm_e_symbolic_n6973_filtered.jsonl` — 6973
   - F (answer-only): `arm_f_answer_only_n7435.jsonl` — 7435
   - G (random control): `arm_g_random_control_n7435.jsonl` — 7435

6. **Built the Stage 5 SFT training pipeline**:
   - `scripts/train_sft.py` — full fine-tuning (NOT LoRA/QLoRA —
     deliberately, so training method stays constant across arms;
     that's Stage 9's separate question) of `Qwen/Qwen3-0.6B-Base`.
     Completion-style prompt (`"Question: ...\nAnswer: ..."`, matching
     every other script — the base model was never instruction-tuned).
     Loss masked to answer tokens only. Every step's loss/grad-norm/LR/
     GPU-temp logged to `train_metrics.jsonl`; every hyperparameter
     saved to `run_config.json` — built specifically because the user
     wants everything recorded for later plots/graphs.
   - `scripts/run_stage5_sweep.py` — orchestrates train+eval across all
     (arm, seed) pairs sequentially, resumable (skips pairs whose
     `run_config.json` + eval `metrics.json` already exist). Reuses
     `scripts/run_baseline.py` UNMODIFIED for eval — just points
     `model_name` in a generated config at the checkpoint dir, so
     results are directly comparable to the Stage 2 baseline
     (52.16% accuracy, 95% CI [49.46%, 54.86%], n=1319). Eval limit
     defaults to 100 test examples per run (Stage 2's first-look
     precedent, ±~5% CI) — full 1319 reserved for whichever arm wins.

7. **Hardware saga — this is important context**:
   - Local RTX 2070 (8GB) hit repeated OOMs during setup: fp32 model +
     plain AdamW doesn't fit (~9.6GB needed just for params+grad+Adam
     states on this 0.6B model, whose 151936-token vocab lm_head
     dominates memory). Worked around with fp16 weights (not fp32
     master + AMP) + bitsandbytes 8-bit AdamW + gradient checkpointing +
     `use_cache=False`. That combo trained successfully but the 2070
     then thermal-cycled constantly (hit 80°C every ~5 steps, pausing
     to cool via a `ThermalSafetyCallback` reusing
     `self_consistency_batched.py`'s pattern) — Arm A alone was
     projected to take ~7+ hours at that throttled rate.
   - User has a second machine — a ZBook laptop, Ubuntu, same WiFi,
     **RTX A5000 Laptop GPU, 16GB VRAM** (initially misidentified as an
     "A500" with assumed 4GB — actually confirmed via `nvidia-smi` as
     A5000/16GB). Much more headroom and presumably better sustained
     cooling.
   - Set up SSH access (172.20.10.3, user `shripad`, host `govinda`) —
     had to add this desktop's public key to the ZBook's
     `authorized_keys` since password auth doesn't work through
     automated tool calls.
   - Pushed the full repo (`~/slm-reasoning-research/`) and this
     project's Obsidian vault notes
     (`~/Documents/ObsidianVault/SLM-Reasoning-Research/`) via `rsync`.
   - Installed deps there (`pip3 install --user torch==2.5.1
     transformers==5.3.0 datasets==4.6.0 accelerate==1.12.0
     bitsandbytes pyyaml numpy huggingface_hub`) — hit one real bug:
     an old system Pillow (9.0.1, missing `PIL.Image.Resampling`) was
     shadowing the need for a newer one; fixed with
     `pip3 install --user --upgrade pillow`.
   - **Current plan (mid-execution as of this handoff)**: move ALL
     training to the ZBook (abandoning the partial, thermal-throttled
     2070 run of Arm A — only ~16% done, not worth preserving) since it
     has far more VRAM and — per the user's explicit direction — should
     use the numerically cleanest full-fine-tuning setup rather than
     the 2070's memory-saving compromises, since "we want everything
     best for research." That means: **fp32 master weights + AMP fp16
     autocast + plain `adamw_torch` (not the 8-bit optimizer)** — just
     changed in `train_sft.py` on the desktop, needs re-syncing to the
     ZBook before restarting the sweep there.
   - Killed all training processes on both machines as of this
     handoff, about to restart the full 7-arm (`--arms A B C D E F G`)
     sweep on the ZBook alone with the updated fp32/AMP/adamw_torch
     config and larger batch sizes (headroom allows it) — resuming
     from a clean state.

## Key decisions and their reasoning (don't re-litigate these)

- **Full fine-tuning, not LoRA, for Stage 5.** Training method is a
  separate later question (Stage 9); mixing it in now would confound
  "does this reasoning format help" with "does this training method
  help."
- **Same hyperparameters across all 7 arms, regardless of which
  machine trains them.** Otherwise a training-setup difference
  confounds the format comparison.
- **Disputed/excluded example set (38 ids) applies identically to every
  arm.** Arms B and E needed retrofitting for this — already done.
- **Eval reuses `run_baseline.py` unmodified.** Don't write a parallel
  eval implementation — any drift from the Stage 2 baseline's exact
  scoring logic would make the new numbers incomparable.
- **Multi-seed requirement (2-3 seeds per arm) is still pending** — the
  current sweep in flight is seed=42 only, a first-pass signal. The
  plan requires confidence intervals, not point estimates, before
  treating any arm's win as a real finding.

## What to do next (if resuming this cold)

1. Check `results/exp05_sft/sweep_log_seed42*.txt` (or wherever the
   latest sweep log lands) for progress.
2. Once all 7 arms finish for seed 42, look at
   `results/exp05_sft/sweep_summary.jsonl` for the first-pass accuracy
   comparison (with Wilson 95% CIs already computed).
3. Decide whether to extend to seeds 2-3 before treating any ordering
   as a real finding (per the plan's own stated methodology
   requirement).
4. Update `docs/research_log.md` and the Obsidian `project-log.md` with
   the ablation results once available — same practice used throughout
   this project.
