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

**Interpretation / what this sets up:** these four failure types (plus the two behavioral patterns) become the evidence-based lens for Stage 5's reasoning-format ablation — e.g. if "missed constraint" and "misapplied percentage" errors dominate, that's a specific argument for why *explicit, verbose* reasoning traces might matter more than concise ones for this model, which is exactly the kind of hypothesis the ablation study (full CoT / concise / symbolic / no-verification / no-reflection / answer-only / randomized-length control) is designed to test rather than assume.

**Status:** full 1319-example test-set run not yet executed (~6.7 hours at current per-example rate at batch_size=1 — Track B's batching work is directly relevant to speeding this up before committing to it).

---

## Project structure note

This project runs two parallel tracks:
- **Track A — reasoning research**: what kind of reasoning supervision actually helps a small model learn to reason (the entries above). See `research-plan.md` in the project's private working notes for the full staged plan; this file mirrors findings, not the full planning detail.
- **Track B — inference engineering**: benchmarking Qwen3-0.6B serving across HF Transformers, vLLM, and quantized GGUF/llama.cpp backends, reusing Track A's own evaluation code to check accuracy parity alongside speed. See `theory/14_inference_serving.md` and `experiments/expB1_inference_benchmark/README.md`. Not yet started as of this entry.
