"""Stage 3 — teacher trace generation via GPT-OSS-20B (NVIDIA API).

Scope (per the 2026-09-06 dataset-reuse decision, see research-plan.md):
GSM8K's own reference solutions already cover the "concise/symbolic" style
content needed for Stage 5's Arms B/E — only 0.8% contain any reflection/
verification language. So this script generates ONE teacher trace per
problem that explicitly includes BOTH verification and reflection content,
from which Arms A (full), C (verification-stripped), and D (reflection-
stripped) can all be derived later by stripping specific sentences — not
three separate teacher generations per problem.

Every trace's final answer is checked against ground truth and only
matching traces are kept (never trust blindly). A cheap text-similarity
check against the dataset's own reference solution flags likely memorized
recital rather than genuine derivation (GSM8K is old and widely scraped).

Usage:
    python scripts/generate_teacher_traces.py --split train --n-examples 20 \
        --out results/exp03_teacher/traces_n20.jsonl
"""
import argparse
import difflib
import json
import os
import sys

import httpx
import time

from datasets import load_dataset
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.gsm8k import extract_reference_answer, to_examples  # noqa: E402
from src.evaluation.answer_extraction import extract_predicted_answer_with_method, is_correct  # noqa: E402

SYSTEM_PROMPT = """You are solving a grade-school math word problem. Work through it step by \
step, showing every calculation explicitly. Your solution MUST include BOTH of the following, \
naturally worked into your derivation (not as separate labeled sections):

1. At least one VERIFICATION moment — after computing an intermediate or final result, \
explicitly check it (e.g. "Let me verify: ... that checks out" or "double-checking this \
against the total given: ...").
2. At least one REFLECTION moment — a point where you reconsider or double back on your \
own approach (e.g. "Wait, let me reconsider whether I've accounted for..." or "Actually, \
I need to re-examine that step because...").

Solve the problem independently and honestly — do not assume you have seen this exact \
problem before; derive the answer from the numbers given. End your response with exactly \
the phrase "The answer is <number>." with the final numeric answer.
"""


def build_user_prompt(question: str) -> str:
    return f"Question: {question}"


def contamination_similarity(generated: str, reference_solution: str) -> float:
    """Rough text-similarity ratio (0-1) between the generated trace and the
    dataset's own reference solution. High similarity is a signal of
    memorized recital rather than independent derivation — a spot-check
    heuristic, not a hard filter."""
    return difflib.SequenceMatcher(None, generated.lower(), reference_solution.lower()).ratio()


def generate_one(client: OpenAI, question: str) -> dict:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ],
        temperature=0.7,
        top_p=1,
        max_tokens=2048,
        stream=False,
    )
    msg = completion.choices[0].message
    return {
        "content": msg.content or "",
        "reasoning_trace": getattr(msg, "reasoning_content", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0, help="start index into the split (for resuming/expanding)")
    ap.add_argument("--resume", action="store_true", help="skip example_ids already present in --out and append")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        for line in open(env_path):
            if line.startswith("NVIDIA_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
    # connect timeout separate from the overall read timeout — a connection
    # that never establishes should fail fast rather than hang past the
    # nominal 90s (seen in practice: a hung request outlasted a 90s-only
    # timeout, likely a connect-level stall the client didn't enforce).
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=httpx.Timeout(90.0, connect=15.0),
    )

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    examples = to_examples(ds)[args.offset : args.offset + args.n_examples]

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        examples = [e for e in examples if e.example_id not in done_ids]
        print(f"resuming: {len(done_ids)} already done, {len(examples)} remaining", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_kept, n_wrong, n_flagged_contam = 0, 0, 0
    with open(args.out, "a" if args.resume else "w") as out_f:
        for i, ex in enumerate(examples):
            for attempt in range(3):
                try:
                    gen = generate_one(client, ex.question)
                    break
                except Exception as e:
                    print(f"  retry {attempt} for example {ex.example_id}: {e}", file=sys.stderr, flush=True)
                    time.sleep(2 ** attempt)
            else:
                print(f"ex={ex.example_id} API_FAILED, skipping", flush=True)
                continue

            extracted_answer, method = extract_predicted_answer_with_method(gen["content"])
            correct = is_correct(extracted_answer, ex.reference_answer)
            similarity = contamination_similarity(gen["content"], ex.reference_solution)
            flagged_contam = similarity > 0.5  # heuristic threshold, spot-check not hard filter

            record = {
                "example_id": ex.example_id,
                "question": ex.question,
                "reference_solution": ex.reference_solution,
                "reference_answer": ex.reference_answer,
                "teacher_trace": gen["content"],
                "teacher_reasoning_internal": gen["reasoning_trace"],
                "extracted_answer": extracted_answer,
                "extraction_method": method,
                "is_correct": correct,
                "contamination_similarity": round(similarity, 3),
                "flagged_possible_contamination": flagged_contam,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            n_kept += int(correct)
            n_wrong += int(not correct)
            n_flagged_contam += int(flagged_contam)
            print(f"[{i+1}/{len(examples)}] ex={ex.example_id} correct={correct} "
                  f"sim={similarity:.2f}{' CONTAM?' if flagged_contam else ''}", flush=True)

    print(f"\ndone: {n_kept} correct / {n_kept+n_wrong} total ({n_kept/(n_kept+n_wrong):.1%} kept), "
          f"{n_flagged_contam} flagged for possible contamination")


if __name__ == "__main__":
    main()
