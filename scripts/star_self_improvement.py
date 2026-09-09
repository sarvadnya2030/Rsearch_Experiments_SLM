"""Stage 4, Arm 2 — STaR (Self-Taught Reasoner) self-improvement, no teacher.

For each training question:
1. Sample k own-attempt completions at temperature>0 (reusing the batched
   generation approach from self_consistency_batched.py). If ANY sample
   reaches the correct answer, keep the first correct one as-is — proof
   the model already has a path to this answer.
2. If NONE of the k own attempts are correct, fall back to hint-based
   backward rationalization: tell the model the correct answer and ask it
   to derive reasoning that reaches it. Verify the hinted generation
   actually states that answer before keeping it (a hint is not a
   guarantee the model will use it correctly).

Both outcomes get written to the same output file with a `source` field
("own_attempt" or "hint_rationalized") so later analysis can compare how
much of the final training set came from each path.

Usage:
    python scripts/star_self_improvement.py --out results/exp04_star/star_n2000.jsonl \
        --n-examples 2000 --k-own-attempt 4
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.gsm8k import build_completion_prompt, to_examples  # noqa: E402
from src.evaluation.answer_extraction import extract_final_answer, is_correct, _truncate_at_continuation  # noqa: E402

# Few-shot demonstrations of genuine backward rationalization, needed because
# Qwen3-0.6B-Base has no instruction tuning and does not reliably follow a
# zero-shot embedded instruction (piloted: it either ignores the hint entirely,
# or fakes arithmetic to force-land on the given number rather than deriving
# it — e.g. "144 x 52 = 7488... however the answer is 624, which is
# 7488 - 624 = 6864 - 624 = 624"). These demos are original toy problems, NOT
# from GSM8K, to avoid any contamination with the actual train/test data —
# they only exist to show the completion pattern: derive forward using the
# given numbers, arrive cleanly at the stated hint, no reverse-engineering.
_HINT_FEWSHOT_PREFIX = """Question: A farmer has 12 sheep. He buys 5 more sheep. How many sheep does he have now?
(The final answer to this question is 17. Show the step-by-step reasoning that leads to this answer.)
Answer: The farmer starts with 12 sheep. He buys 5 more sheep, so he now has 12 + 5 = 17 sheep. The answer is 17.

Question: A bakery makes 8 trays of muffins with 6 muffins on each tray. They sell 20 muffins. How many muffins are left?
(The final answer to this question is 28. Show the step-by-step reasoning that leads to this answer.)
Answer: The bakery makes 8 trays with 6 muffins each, so they have 8 x 6 = 48 muffins in total. They sell 20 muffins, so they have 48 - 20 = 28 muffins left. The answer is 28.

Question: A store had 90 apples. They sold some apples and have 34 left. How many apples did they sell?
(The final answer to this question is 56. Show the step-by-step reasoning that leads to this answer.)
Answer: The store started with 90 apples and now has 34 left. The number sold is the difference: 90 - 34 = 56. The answer is 56.

"""

HINT_PROMPT_TEMPLATE = (
    _HINT_FEWSHOT_PREFIX +
    "Question: {question}\n"
    "(The final answer to this question is {answer}. Show the step-by-step reasoning "
    "that leads to this answer.)\n"
    "Answer:"
)


def gpu_temp_c():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip())
    except Exception:
        return None


def wait_for_cooldown(pause_above_c: int, resume_below_c: int, poll_seconds: int = 15):
    temp = gpu_temp_c()
    if temp is None or temp < pause_above_c:
        return
    print(f"GPU at {temp}C >= {pause_above_c}C, pausing until below {resume_below_c}C...")
    while True:
        time.sleep(poll_seconds)
        temp = gpu_temp_c()
        if temp is None or temp < resume_below_c:
            print(f"resuming (temp={temp}C)")
            return


@torch.no_grad()
def _generate_raw(model, tokenizer, device, prompts, k, temperature, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    padded_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )
    texts = tokenizer.batch_decode(outputs[:, padded_len:], skip_special_tokens=True)
    return [texts[i * k : (i + 1) * k] for i in range(len(prompts))]


def generate_adaptive(model, tokenizer, device, prompts, k, temperature, max_new_tokens):
    """Same OOM-halving self-adaptation as self_consistency_batched.py."""
    try:
        return _generate_raw(model, tokenizer, device, prompts, k, temperature, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        left = generate_adaptive(model, tokenizer, device, prompts[:mid], k, temperature, max_new_tokens)
        right = generate_adaptive(model, tokenizer, device, prompts[mid:], k, temperature, max_new_tokens)
        return left + right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--question-batch-size", type=int, default=6)
    ap.add_argument("--k-own-attempt", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pause-above-c", type=int, default=80)
    ap.add_argument("--resume-below-c", type=int, default=65)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = to_examples(ds)[args.offset : args.offset + args.n_examples]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_own, n_hint_ok, n_hint_failed, n_total = 0, 0, 0, 0

    with open(args.out, "w") as out_f:
        for batch_start in range(0, len(examples), args.question_batch_size):
            wait_for_cooldown(args.pause_above_c, args.resume_below_c)
            batch = examples[batch_start : batch_start + args.question_batch_size]

            # Phase 1: own-attempt sampling
            own_prompts = [build_completion_prompt(ex.question) for ex in batch]
            own_grouped = generate_adaptive(model, tokenizer, device, own_prompts, args.k_own_attempt, args.temperature, args.max_new_tokens)

            needs_hint = []
            for ex, responses in zip(batch, own_grouped):
                kept = None
                for r in responses:
                    res = extract_final_answer(r, hit_max_new_tokens=False)
                    if is_correct(res["extracted_answer"], ex.reference_answer):
                        kept = r
                        break
                if kept is not None:
                    # Truncate at the first sign of a hallucinated continuation
                    # (base model rambling into a new, unrelated question after
                    # already answering — same behavior as Stage 2's Finding 1).
                    # Scoring already accounted for this via extract_final_answer,
                    # but the SAVED trace must be the clean version too, or the
                    # student model would be trained to imitate the rambling.
                    record = {
                        "example_id": ex.example_id,
                        "question": ex.question,
                        "reference_answer": ex.reference_answer,
                        "star_trace": _truncate_at_continuation(kept).rstrip(),
                        "source": "own_attempt",
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_own += 1
                else:
                    needs_hint.append(ex)

            # Phase 2: hint-based backward rationalization for the rest
            if needs_hint:
                hint_prompts = [HINT_PROMPT_TEMPLATE.format(question=ex.question, answer=ex.reference_answer) for ex in needs_hint]
                hint_grouped = generate_adaptive(model, tokenizer, device, hint_prompts, 1, args.temperature, args.max_new_tokens)
                for ex, responses in zip(needs_hint, hint_grouped):
                    r = responses[0]
                    res = extract_final_answer(r, hit_max_new_tokens=False)
                    if is_correct(res["extracted_answer"], ex.reference_answer):
                        record = {
                            "example_id": ex.example_id,
                            "question": ex.question,
                            "reference_answer": ex.reference_answer,
                            "star_trace": _truncate_at_continuation(r).rstrip(),
                            "source": "hint_rationalized",
                        }
                        out_f.write(json.dumps(record) + "\n")
                        n_hint_ok += 1
                    else:
                        n_hint_failed += 1

            out_f.flush()
            n_total += len(batch)
            print(f"[{n_total}/{len(examples)}] own={n_own} hint_ok={n_hint_ok} hint_failed={n_hint_failed}")

    kept_total = n_own + n_hint_ok
    print(f"\ndone: {kept_total}/{n_total} kept ({kept_total/n_total:.1%}) — "
          f"{n_own} from own attempts, {n_hint_ok} via hint rationalization, "
          f"{n_hint_failed} failed even with the hint (discarded)")


if __name__ == "__main__":
    main()
