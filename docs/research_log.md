# Research Log

Public mirror of project findings. Working notes live in a private Obsidian vault during active development; this file is updated after each milestone so the methodology and findings are visible without requesting access to anything private.

Every entry follows the project's experiment discipline: **prediction -> hypothesis -> result -> interpretation.**

---

## 2026-09-05 — Repo scaffolded; Experiment 0 baseline (Track A)

**What we did:** Built the initial repo structure (`theory/01-06`, `from_scratch/` placeholders, `src/`, `scripts/run_baseline.py`, tests). Ran Qwen3-0.6B-Base on GSM8K's test split with zero training — pure observation of baseline reasoning behavior, no sampling (greedy decoding) for reproducibility.

**Environment:** Python 3.10.12, torch 2.5.1+cu121, transformers 5.3.0, RTX 2070 8GB VRAM. Confirmed `transformers==5.3.0` deprecated `torch_dtype=` in favor of `dtype=` in `from_pretrained` — adapted accordingly rather than guessing at the API.

**Model/data facts confirmed (not assumed) via HF Hub / `datasets`:**
- `Qwen/Qwen3-0.6B-Base`: Qwen3ForCausalLM, hidden_size=1024, 28 layers, GQA (16 query heads / 8 KV heads), head_dim=128, RoPE, tied embeddings, vocab 151,936. Its tokenizer ships a full Qwen3 chat_template despite being a non-instruction-tuned base checkpoint — we deliberately prompt with plain completion-style text (`"Question: ...\nAnswer:"`), not `apply_chat_template`.
- GSM8K (`openai/gsm8k`): two configs, `main` and `socratic`, both 7473 train / 1319 test rows, same underlying questions/answers — `socratic` phrases each reasoning step as a self-posed question, a candidate second SFT supervision format for later.

### Finding 1 — An extraction bug masqueraded as a model failure

**Prediction going in:** the smoke test (10 examples) would show some baseline accuracy, likely modest for a 0.6B base model.

**What happened:** first run scored 0/10. Reading the raw traces (not just the aggregate number) showed the model was often reasoning correctly and stating the right answer — then, because it's a true base model with no learned stop-on-answer behavior, continuing to generate and hallucinating a brand-new, unrelated Q&A pair afterward. The answer extractor took the *last* "the answer is N" match in the text, which was landing on the model's self-invented follow-up question's answer, not the actual question's.

**Fix:** truncate the extraction search at the first sign of a new, unrelated question (`\nQuestion:`, `\n[Question]`, `\nQ:`) before pattern-matching, while still preserving the complete raw response in the saved record (never lose data, only change what we search).

**Result after fix:** 4/10 correct on the same 10 examples — a plausible zero-shot number for a 0.6B base model on GSM8K.

**Interpretation:** the lesson here generalizes beyond this one bug — an evaluation pipeline's correctness has to be verified against raw traces before trusting its aggregate metric, especially for a model whose failure modes (not stopping) interact with how you measure it (taking the last match). This is now a permanent regression test (`tests/test_answer_extraction.py::test_ignores_hallucinated_followup_*`).

### Finding 2 — 100-example run: 53% accuracy, four distinct real failure types

**Result:** 53/100 correct, 25.65 tok/s avg, 18.2s/example avg.

**Manual review of a sample of errors** (not LLM-judged; human-read against each reference solution) surfaced four qualitatively distinct failure types, not just "arithmetic mistakes":
1. **Missed/dropped a necessary constraint** — e.g. ignored "3 cups *per chicken*," used the given numbers in the wrong relationship instead of computing the true total first.
2. **Misinterpreted problem semantics** — e.g. read "every second glass costs 60%" as "the second half of the glasses are discounted" instead of alternating glasses.
3. **Misapplied a percentage operation** — e.g. treated "reduced by 30%" as "equals 30% of," rather than "subtract 30% of" from the base quantity.
4. **Failed to recognize a multi-step accumulation/break-even structure** — correctly computed one year's profit, never iterated to find the actual break-even year.

Plus two behavioral patterns from the smoke test: **degenerate repetition loops** (re-derives the same intermediate arithmetic step endlessly, never reaches an answer) and **post-answer hallucination** (the Finding 1 behavior — answers correctly, then invents an unrelated new question, which still costs generation budget even after the extraction fix).

**Length-cap observation (descriptive only, not causal):** 70% of incorrect answers hit the 512-token generation cap, vs. 55% of correct answers. A real but modest gap — hitting the cap is common regardless of correctness, mostly explained by the rambling/hallucination behavior above rather than the model needing more room to actually solve the problem.

**Interpretation:** these four failure types (plus the two behavioral patterns) give an evidence-based basis for later reasoning-format experiments, rather than guessing in advance what a small model's failures look like.

### Finding 3 — Is the 512-token cap truncating genuine reasoning, or just rambling?

**Prediction:** since 512 median-length responses were hitting the generation cap, raising `max_new_tokens` might reveal reasoning that's currently being cut off before it converges.

**What we checked:** of the 62/100 capped responses, 57 already contained an explicit "the answer is N" phrase somewhere within the 512 tokens — the model had already answered, then kept rambling (harmless, already handled by Finding 1's fix). Only 5 had no answer phrase at all. Reading those 5 directly: all five are degenerate, non-convergent (repeated identical arithmetic sentences, or an ever-diverging numeric drift) — none show signs of being *about to* reach an answer with more room.

**Result:** raising the token cap would not have changed any of these 100 outcomes. The failure mode is the model not stopping, not the model needing more space to think.

**Fix applied anyway (unrelated to the cap size):** the extraction pipeline was silently treating a bare trailing number from those 5 degenerate cases as if it were a real prediction (e.g. extracting "10" from a repetition loop that never stated an answer). Added `extraction_method` (`phrase` vs `fallback` vs `none`) and a `termination_status` label (`stopped_with_answer` / `stopped_no_answer` / `capped_with_answer` / `capped_no_answer`) to every record. A fallback match is now only trusted when the model stopped on its own; a fallback match from a response that was truncated at the cap is treated as `capped_no_answer` — no coherent answer — rather than a wrong guess. Reprocessing the existing 100-example run changed 5 predictions (accuracy unchanged at 53/100, since those 5 were already wrong either way) and produced a clean termination breakdown: 38 `stopped_with_answer`, 57 `capped_with_answer`, 5 `capped_no_answer`.

**Decision:** keep `max_new_tokens=512` for the full 1319-example run. `hit_max_new_tokens` and `termination_status` are now logged per-example, so if the full run's distribution looks different from this 100-example sample, that will be visible in the data rather than assumed.

### Finding 4 — Full 1319-example run confirms the 100-example sample

**Prediction:** the 100-example sample (53% accuracy, 57%/38%/5% termination split) should be a reasonable predictor of the full test set, within its ±5% confidence interval.

**Result:** 688/1319 correct = **52.16% accuracy** (95% CI: 49.46%–54.86%). Termination status: `capped_with_answer` 785 (59.5%), `stopped_with_answer` 487 (36.9%), `capped_no_answer` 47 (3.6%), zero `stopped_no_answer`. Avg 17.77s/example, 26.84 tok/s.

**Interpretation:** the full run lands squarely inside the 100-example sample's confidence interval, and the termination-status proportions match closely (57%/38%/5% predicted vs. 59.5%/36.9%/3.6% actual). The smaller sample generalized well — this is now the anchor baseline number (52.16% ± 2.7%) for Qwen3-0.6B-Base zero-shot on GSM8K, to compare every later post-training stage against. The 47 `capped_no_answer` cases are candidates for a targeted rerun under a higher token budget (`scripts/rerun_capped.py`) to confirm at full scale that a larger cap wouldn't help — not yet run.

**Status:** Experiment 0's baseline-observation objective is complete. Remaining before moving to the next stage: manually review a larger sample of the 631 total errors to solidify the failure taxonomy (currently based on a handful of manually-reviewed examples), and run the calibration/entropy check.

## 2026-09-06 — Failure taxonomy locked (LLM-judged, full 631 errors); calibration check; Stage 2.5 self-consistency baseline; Track B first batching win

**What we did:** Closed out Stage 2 (failure taxonomy + calibration check), then ran Stage 2.5 (self-consistency baseline) both locally and on a Colab T4, discovering and fixing a real GPU-batching gap in the process.

### Finding 5 — Full-scale failure taxonomy via GPT-OSS-20B as judge, with saved reasoning traces

**Method:** rather than hand-classifying all 631 errors from the full 1319-example run, used GPT-OSS-20B (NVIDIA-hosted API — the same model planned as Stage 3's teacher) as a judge: given the question, reference solution, and the model's raw response, classify into one of the 4 failure types from Finding 2 plus new candidates, with a one-sentence explanation citing the actual numbers. Verified the judge's calls by hand against a 30-example random sample first — agreement was strong, and the judge's calls were often more precise than the manual pass (e.g. correctly separating "reused the wrong time span" from a generic "misread semantics" label).

**Taxonomy expanded from 4 to 11 categories** after the manual sample surfaced patterns the original 4 didn't cover: `arithmetic_slip` (correct setup, wrong final number — a fundamentally different failure than a reasoning error), `unit_conversion_error` (feet/inches, dollars/cents), `rate_segment_conflation` (merges two rate brackets that must stay separate, e.g. weekday/weekend), `quantity_type_confusion` (treats a count as a dollar amount), and `no_sanity_check` (reaches an implausible result — e.g. a negative price — without flagging it). The first classification pass silently folded `unit_conversion_error` and `no_sanity_check` into broader categories; fixed by explicitly instructing the judge to check for those two narrower categories first, before falling back to broad ones.

**Result across all 631 errors** (631/631 classified, 0 unparseable after retrying 3 with a larger token budget — their reasoning traces had consumed the whole 2048-token budget before reaching the final JSON):

| Category | Count | % |
|---|---|---|
| misread_semantics | 350 | 55.5% |
| missed_constraint | 63 | 10.0% |
| arithmetic_slip | 63 | 10.0% |
| misapplied_percentage | 61 | 9.7% |
| degenerate_loop | 25 | 4.0% |
| accumulation_failure | 22 | 3.5% |
| unit_conversion_error | 15 | 2.4% |
| no_sanity_check | 12 | 1.9% |
| other | 11 | 1.7% |
| rate_segment_conflation | 5 | 0.8% |
| quantity_type_confusion | 4 | 0.6% |

**Interpretation:** over half of all errors (55.5%) are misread-semantics failures — the model's core weakness is translating word-problem structure into the right sequence of operations, not arithmetic itself (pure `arithmetic_slip` is only 10%). This argues that Stage 5's reasoning-format supervision should target problem comprehension, not just calculation practice. Full reasoning traces for every classification saved to `results/exp00_baseline/error_taxonomy_full.jsonl` — flagged as a reusable seed for Stage 5.5's verifier (the judge's per-error explanations are effectively step-level "here's what went wrong" labels, close to PRM training data) and as an early positive signal on GPT-OSS-20B's reasoning quality ahead of using it as Stage 3's teacher.

### Finding 6 — Calibration check: the model's own confidence barely predicts correctness, and completely fails to flag implausible answers

**Method:** teacher-forced the model's own saved raw text back through itself in one forward pass per example (no new generation needed — logits at position *i* depend only on tokens before it, so replaying the model's own output reproduces the exact distribution it used to pick each token), then measured entropy at the token(s) covering the extracted final-answer number. Scored 1162/1319 examples (157 skipped — no extractable phrase-matched answer span).

**Result:** median entropy for incorrect answers (0.0043) is ~2x that of correct answers (0.0022); mean is ~5x higher (0.0282 vs 0.0058), driven by a heavy tail. Formal separation: **AUC = 0.672** (better than chance, far from reliable) for "does answer-token entropy predict correctness." 93% of correct answers are near-zero entropy vs. 70% of incorrect ones — most wrong answers are still produced with full confidence.

**Counterintuitive finding:** the 12 `no_sanity_check` examples (negative dollar amounts, etc.) have *lower* mean entropy (0.021) than other error types (0.028) — the model isn't hesitating at all when it produces something absurd; if anything it's more confident there.

**Interpretation:** greedy-decode confidence alone cannot catch obviously-wrong outputs — this is a point in favor of both Stage 2.5 (external agreement via voting, rather than internal confidence) and Stage 5.5 (an explicit trained verifier), since neither depends on the base model's own (weak) self-assessment.

### Finding 7 — Self-consistency baseline: real, un-saturated gains up to k=16; and is a teacher even needed?

**Side investigation before Stage 3:** checked whether GSM8K's own reference solutions (already present in the dataset) could substitute for teacher-generated reasoning traces in Stage 4/5. Only **61/7473 (0.8%) of train reference solutions contain any reflection/verification language** ("wait," "let me check," etc.) — they're terse, single-pass, calculator-annotated derivations. **Decision:** derive Stage 5's Arms B (concise) and E (symbolic) directly from the dataset's own solutions — free, no teacher cost, no external contamination risk. Only generate teacher traces for Arms A/C/D, which need reflective/verification content the dataset doesn't have.

**Self-consistency result (majority vote over k temperature-sampled completions, temperature=0.7, no training):**

| k | Local (RTX 2070, n=100) | Colab (Tesla T4, n=100) |
|---|---|---|
| 1 | 40% | 32% |
| 2 | 40% | 32% |
| 4 | 47% | 41% |
| 8 | 60% | 50% |
| 16 | — | 61% |

Every matching k's 95% Wilson confidence interval overlaps between the two runs (e.g. k=8: local [50.2,69.1] vs. colab [40.4,59.6]) despite identical nominal config (temperature=0.7, seed=42) — the ~8-10 point gap is sampling noise at n=100, not a real hardware/software discrepancy. This is exactly why the plan requires confidence intervals rather than point estimates for comparative claims (see Stage 5's methodology requirement).

**Interpretation:** accuracy climbs from ~32-40% (k=1, temperature-sampled — expected to be below the 52.16% greedy baseline, since greedy is a stronger single-shot strategy) to ~50-61% (k=8-16), and has **not saturated by k=16**. Per the plan's own decision criterion — "if majority voting alone closes most of the gap, a verifier's added value is small" — the still-growing gap means **Stage 5.5's verifier remains worth pursuing**, not redundant.

### Track B — First real batching implementation (B3), plus a GPU thermal-safety lesson

**Problem:** the original self-consistency pilot processed one question at a time (`num_return_sequences=k` for a single prompt), leaving most of the GPU's parallel capacity idle — 24s/question at k=8 on the RTX 2070.

**Fix:** batched multiple *different* questions together (`self_consistency_batched.py`) — tokenize B questions with left-padding, generate k samples for each in one forward-pass batch (batch dim = B×k). Result: **6.9s/question at B=6, k=8 — a 3.5x speedup**, verified to produce correctly-grouped, uncorrupted per-question outputs. Found the GPU's real memory ceiling empirically (B=7 questions × k=8 = 56 sequences: 5.61GB peak, safe; B=8 × k=8 = 64 sequences: OOM) rather than guessing.

**Real hardware lesson:** after hours of sustained inference (full baseline run, full entropy check, several self-consistency attempts), the RTX 2070 hit **87°C with active software thermal throttling** (`clocks_event_reasons.active = 0x20`), causing unpredictable 10-30x slowdowns that looked like hangs/bugs but were actually the GPU protecting itself. Added a self-regulating cooldown check to `self_consistency_batched.py` (pauses between batches above 80°C, resumes below 65°C) so runs are safe unattended rather than needing manual monitoring.

**Colab/Kaggle portability:** wrote a fully self-contained notebook script (`scripts/colab_self_consistency.py`) — installs its own dependencies, downloads model+dataset fresh, and auto-adapts its batch size on OOM (halves the question sub-batch and retries recursively, never touching k) rather than needing hand-tuned constants per GPU. Confirmed working on a Tesla T4 (15GB) after one OOM at the initial 16×16 batch guess corrected itself down automatically.

**Status:** Stage 2 (taxonomy + calibration) and Stage 2.5 (self-consistency baseline) both complete. Next: Stage 3 (GPT-OSS-20B teacher generation), scoped to only the reflection/verification-heavy reasoning-format arms per Finding 7's dataset-reuse decision. `research-plan.md` updated accordingly.

## 2026-09-07 — STaR hint-rationalization fix (few-shot); saved-trace hallucination bug found and fixed

**Started Stage 3 (teacher generation, GPT-OSS-20B, local, API-bound) and Stage 4 Arm 2 (STaR self-improvement, Qwen3-0.6B, Colab GPU) in parallel** — no resource conflict since one is API-bound and one is local-GPU-bound. Teacher generation hung once on an unresponsive API call that outlasted its 90s read timeout (client-side timeout doesn't cover a connect-level stall); fixed with a separate 15s connect timeout, `--resume` support, and unbuffered output (the previous buffered stdout made a live process look frozen when it wasn't, and briefly the reverse — wasted real debugging time before finding the true cause). Also found and fixed a real dataset-selection gap by checking public HF datasets before generating anything: `HAD653/gsm8k-cot-120b` (7269 rows, 100% verified correct, but only 1.2% reflection content) is a good free source for Stage 5's Arm B — checked and adopted; no existing dataset had enough reflection+verification content for Arms A/C/D, confirming teacher generation was still necessary there.

### Finding 8 — Zero-shot hints fail on a non-instruction-tuned base model; few-shot fixes it

STaR's hint-rationalization step (give the model the correct answer, ask it to derive reasoning toward it) initially used a zero-shot embedded instruction. Piloted on 10 examples: 0/3 zero-shot hint attempts succeeded. Reading the actual failures showed why: one case ignored the hint entirely and produced an unrelated wrong derivation; another **faked arithmetic to force-land on the given number** — "...he writes 144 x 52 = 7488 pages... however the final answer is 624, which is 7488 - 624 = 6864 - 624 = 624 pages" — nonsense math dressed up as a derivation. Root cause: Qwen3-0.6B-Base has no instruction tuning, so a zero-shot embedded instruction isn't reliably followed (consistent with every other base-model behavior observed all project: no learned stop-on-answer, no reliable format-following).

**Fix:** switched to a few-shot hint prompt (3 original, non-GSM8K toy problems demonstrating clean forward derivation landing on a stated answer). Re-tested the same two failing examples: the fake-arithmetic case now produces genuine derivation reaching the correct answer; the other case still fails, but genuinely (a legitimately hard two-part problem), not by faking — correctly discarded rather than padded with garbage. Re-ran the 10-example pilot: yield improved from 80% to 90%, with the recovered example's saved trace confirmed clean by manual inspection.

### Finding 9 — Full-scale STaR run: real yield holds, but a bigger saved-data bug was hiding underneath

Scaled to n=2000 on Colab (Tesla T4), using the few-shot-fixed pipeline, batched (question-batching, not just per-question sampling) with an OOM-adaptive batch size (self-adapts via recursive halving instead of a hand-tuned constant) and Google Drive checkpointing (writes directly to a mounted Drive path with resume-on-reconnect, added specifically because Colab sessions disconnect and this run was left overnight).

**Result:** 1725/2000 kept (86.2%) — 1552 own-attempt (77.6%), 173 hint-rationalized (8.6%), 275 discarded (13.8%), in 12149s (6.07s/example). Yield matches the pilot's ~85-90% closely — no degradation at 200x scale.

**But spot-checking `hint_rationalized` traces for the fake-arithmetic pattern (the thing Finding 8 fixed) surfaced a DIFFERENT, bigger problem**: some saved traces contained a full second, fabricated Q&A pair appended after the real, correct answer — e.g. after correctly answering "The answer is 6," the model kept generating and hallucinated an entirely unrelated geometry problem in the exact format of the few-shot template, ending with a nonsensical "The answer is π." This is the same post-answer-hallucination behavior from Stage 2's Finding 1 (base model never learned to stop after answering) — but here it slipped into the *saved training data*, not just an evaluation artifact. **Scoring was never wrong** (extraction already truncates before this content, so `is_correct` was computed correctly the whole time) — the bug was that the *raw, untruncated* generation got saved to `star_trace`, not the clean truncated version used for scoring.

**Scanned the full n=2000 output:** 88.1% of `own_attempt` traces (1368/1552) and 11.6% of `hint_rationalized` traces (20/173) contained this — far more widespread than the 2-in-8 hit rate from the initial random sample suggested. Checked teacher traces (GPT-OSS-20B, Stage 3) for the same issue: 0/1010 affected — expected, since that's a properly instruction-tuned model with correct stop behavior, unlike the raw base model STaR uses.

**Fix, two parts:**
1. Wrote `scripts/clean_star_traces.py` to post-process the existing n=2000 output: truncates every `star_trace` at the first hallucinated-continuation marker, then re-verifies the truncated trace still reaches the correct answer (it always should, since truncation only removes content *after* the answer). Result: 1388/1725 rows needed truncation, 0 broke re-verification, all 1725 rows kept. Output: `results/exp04_star/star_n2000_cleaned.jsonl` — this is the training-ready file, not the raw download.
2. Fixed the root cause in both `star_self_improvement.py` and `colab_star_self_improvement.py` so future runs save the truncated trace directly — no post-processing step needed going forward.

**Interpretation:** this is the same lesson as Stage 2's Finding 1, relearned in a new context — an evaluation-time fix (truncating for scoring) does not automatically propagate to every place raw model output gets saved and reused. Any pipeline that both scores AND stores a base model's raw generation needs the truncation applied at the storage boundary, not just the scoring boundary. Worth checking for this same class of bug in any future pipeline that saves base-model output for later reuse.

**Status:** STaR data now clean and training-ready (1725 examples). Teacher generation continuing overnight (2026-09-06/07) past its original n=2000 target to cover the full 7473-example train split, using the `--resume` mechanism, orchestrated to auto-continue once the first target is hit.

## 2026-09-07/09 — Full-scale teacher generation complete; Stage 5 Arms A-E built at full scale; 38 GSM8K reference-solution bugs found and excluded

**Teacher generation (Stage 3) ran to completion over ~2 days** (started 2026-09-06 overnight, finished 2026-09-09), producing traces for the full 7473-example GSM8K train split via `scripts/generate_teacher_traces.py`. First-pass accuracy: 94.8% (a normal, expected rate for a 20B instruction-tuned model on GSM8K, not a concern). One example (id 3148) hit 3 consecutive API failures during generation and was silently skipped by the original script (a real gap — worth hardening `generate_teacher_traces.py` to log and retry API failures at generation time, not just downstream) — caught by comparing generated example_ids against the full 0-7472 range, then solved manually and appended.

**Automated retry + escalating manual-fix pipeline (new, `scripts/retry_wrong_teacher_traces.py`):** every first-pass failure gets up to 2 fresh API attempts per pass (parallelized across 8 workers via `ThreadPoolExecutor` — cut a ~4.5hr sequential retry pass down to ~20 minutes), re-attempted on every subsequent pass rather than given up on after one try. Once an example's *cumulative* attempts across passes hits 3, it's flagged `needs_manual_fix` and handed to a from-scratch derivation (by Claude, not the API) instead of burning further budget — the derivation must independently reach a verified answer, never copy the dataset's own `reference_solution` (which would trip the same contamination check used elsewhere in this pipeline). Of ~430 total first-pass failures across the full run, all were eventually resolved: the large majority recovered (API retry or manual derivation reaching `reference_answer`), and 38 correctly identified as genuine dataset bugs (below) rather than forced to match a wrong label.

**Finding 10 — 38 confirmed GSM8K reference-solution errors, independently re-verified.** When a from-scratch derivation couldn't reach `reference_answer` even after careful double-checking, it was marked disputed rather than forced — this surfaced 38 examples where GSM8K's own reference solution is objectively wrong: self-contradictions (own arithmetic computes one number, `####` states another — e.g. id 356: derivation gives 64, tag says 66; id 7182: derivation gives 10, tag says 4), arithmetic slips (id 167: `192×100` mislabeled 1920 instead of 19200), dropped/misread quantifiers (id 4943 drops "dozen"; id 5936 misreads "each" as "total"; id 4838 uses 4 sections instead of the stated 6), wrong operand substitutions (id 6768: subtraction where the wording requires addition), and at least one case (id 7457) where the reference appears to solve for the wrong person in the problem entirely. **Independently re-verified in a separate pass** (a fresh agent re-derived all 33-then-38 disputed cases from scratch, without seeing the original disputing agent's write-up): 30/33 of the first batch were CONFIRMED objective errors, 3 marked merely UNCERTAIN (defensible under an alternate but non-obvious reading), 0 were found to be wrongly disputed — no exclusions needed reversing. This is a real, citable data-quality finding about GSM8K itself, not just an artifact of this pipeline.

**Quality spot-check (n=25 random sample, seed 42):** average reasoning-quality score 4.6/5 — verification and reflection sections are substantive (real re-derivations, genuine ambiguity resolution), not empty headers. One outlier scored 3 (confident but wrong — the model's own reflection step surfaced the correct reading, then talked itself out of it, id 5378). Separately, 47/7473 (0.64%) had a completely empty `teacher_trace` from an empty API response — all caught and resolved by the same retry/manual pipeline, none slipped into the final training pool unresolved.

**Stage 5 Arms A-E built at final full scale** (`scripts/build_final_teacher_pool.py` merges the main generation file + retry pool into one deduplicated, ground-truth-verified pool before arm derivation):

| Arm | Kept | Source |
|---|---|---|
| A — Full reasoning | 7435/7435 | Teacher (GPT-OSS-20B), verified |
| B — Concise | 7473/7473 | GSM8K reference solutions, calculator annotations stripped |
| C — No verification | 7426/7435 | Teacher, verification section removed |
| D — No reflection | 7418/7435 | Teacher, reflection section removed |
| E — Symbolic | 7001/7473 | GSM8K reference solutions, equations-only chain |

Arm E's higher exclusion rate (472/7473, 6.3%) is mostly structural, not the 38 dataset bugs alone: roughly 8% of GSM8K solutions state their *final* arithmetic step in plain prose without a `<<...>>` calculator annotation, so the equations-only chain is legitimately incomplete for those — correctly filtered by the pre-use verification check (never trust a derived trace blindly), not a bug in the derivation script.

**Status:** Stage 3 (teacher generation) complete. Stage 5's data-prep phase (deriving all 7 planned format variants from real data — F "answer-only" and G "random-shortened control" remain trivial mechanical derivations of what's already here) is now complete for Arms A-E. Next: F/G derivation, then the actual SFT training runs per arm with the plan's required multi-seed, confidence-interval methodology.
