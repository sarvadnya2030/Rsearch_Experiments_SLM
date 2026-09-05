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
