"""Stage 2.5 self-consistency baseline, batched across QUESTIONS (Track B's
first real batching implementation — B3), not just across samples of one
question. Tokenizes a batch of B distinct questions with left-padding,
generates k samples per question in the same forward-pass batch (batch dim
= B*k), and evaluates majority-vote accuracy at k=1,2,4,8.

Usage:
    python scripts/self_consistency_batched.py --out results/exp02_5_self_consistency/n100_batched.jsonl \
        --n-examples 100 --question-batch-size 5 --k-max 8
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpu_temp_c() -> int | None:
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
    print(f"GPU at {temp}C >= {pause_above_c}C, pausing until it drops below {resume_below_c}C...")
    while True:
        time.sleep(poll_seconds)
        temp = gpu_temp_c()
        if temp is None or temp < resume_below_c:
            print(f"resuming (temp={temp}C)")
            return
        print(f"  still hot: {temp}C")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.gsm8k import load_gsm8k_test, to_examples, build_completion_prompt  # noqa: E402
from src.evaluation.answer_extraction import extract_final_answer, is_correct  # noqa: E402


@torch.no_grad()
def generate_batch(model, tokenizer, device, prompts, k, temperature, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    padded_prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )
    response_ids = outputs[:, padded_prompt_len:]
    texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    # HF repeat_interleaves inputs for num_return_sequences: rows [0..k-1] -> prompt 0, etc.
    grouped = [texts[i * k : (i + 1) * k] for i in range(len(prompts))]
    return grouped


def majority_vote(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--question-batch-size", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pause-above-c", type=int, default=80, help="pause between batches if GPU hits this temp")
    ap.add_argument("--resume-below-c", type=int, default=65, help="resume once GPU cools below this temp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation on a causal LM

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()

    examples = to_examples(load_gsm8k_test())[: args.n_examples]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_example = []
    t_start = time.time()
    with open(args.out, "w") as out_f:
        for batch_start in range(0, len(examples), args.question_batch_size):
            wait_for_cooldown(args.pause_above_c, args.resume_below_c)
            batch = examples[batch_start : batch_start + args.question_batch_size]
            prompts = [build_completion_prompt(ex.question) for ex in batch]

            t0 = time.time()
            grouped_texts = generate_batch(model, tokenizer, device, prompts, args.k_max, args.temperature, args.max_new_tokens)
            dt = time.time() - t0

            for ex, responses in zip(batch, grouped_texts):
                extracted = [extract_final_answer(r, hit_max_new_tokens=False)["extracted_answer"] for r in responses]
                record = {
                    "example_id": ex.example_id,
                    "reference_answer": ex.reference_answer,
                    "extracted_answers": extracted,
                    "responses": responses,
                }
                out_f.write(json.dumps(record) + "\n")
                per_example.append(record)

            out_f.flush()
            elapsed = time.time() - t_start
            print(f"batch [{batch_start}:{batch_start+len(batch)}] took {dt:.1f}s ({dt/len(batch):.1f}s/question) | total elapsed {elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"\ndone: {len(per_example)} examples in {total_time:.1f}s ({total_time/len(per_example):.2f}s/example)")

    for k in [1, 2, 4, min(8, args.k_max)]:
        if k > args.k_max:
            continue
        correct = sum(
            1 for rec in per_example
            if is_correct(majority_vote(rec["extracted_answers"][:k]), rec["reference_answer"])
        )
        print(f"k={k}: majority-vote accuracy = {correct}/{len(per_example)} = {correct/len(per_example):.2%}")


if __name__ == "__main__":
    main()
