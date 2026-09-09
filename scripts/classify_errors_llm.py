"""Classify Experiment 0 errors into the Stage 2 failure taxonomy using GPT-OSS-20B
as the classifier, saving its reasoning trace alongside each classification.

Usage:
    python scripts/classify_errors_llm.py --run results/exp00_baseline/run_20260905_194735 \
        --sample-file /path/to/error_sample.jsonl --out results/exp00_baseline/error_taxonomy.jsonl

If --sample-file is omitted, classifies every row in <run>/errors.jsonl.
"""
import argparse
import json
import os
import sys
import time

from openai import OpenAI

TAXONOMY = """\
1. missed_constraint - dropped or ignored a necessary constraint stated in the problem
2. misread_semantics - misinterpreted what the problem is actually asking / how quantities relate
3. misapplied_percentage - percentage or fraction-of operation applied incorrectly (e.g. "reduced by 30%" treated as "equals 30% of")
4. accumulation_failure - failed to recognize a multi-step accumulation or break-even structure
5. arithmetic_slip - the setup/equation is correct but a basic arithmetic operation is computed wrong
6. unit_conversion_error - mixed up units (feet/inches, dollars/cents, minutes/hours, etc.)
7. rate_segment_conflation - merged two segments/brackets that must be computed separately (e.g. two different hourly rates, weekday vs weekend)
8. quantity_type_confusion - treated a count of items as a dollar amount, or vice versa
9. no_sanity_check - reached an implausible result (e.g. negative price, negative count) without noticing
10. degenerate_loop - response gets stuck repeating the same statement/derivation and never reaches a coherent answer
11. domain_knowledge_gap - requires knowledge outside pure arithmetic (e.g. physical/practical facts) that the model lacks
12. other - does not fit any category above; explain briefly
"""

SYSTEM_PROMPT = f"""You are classifying failure modes of a small language model (Qwen3-0.6B-Base) on GSM8K math word problems. \
Given a question, the reference solution, and the model's full (raw, unedited) response, pick exactly one category \
from this taxonomy that best explains why the model's extracted answer was wrong:

{TAXONOMY}

Before picking a broad category like misread_semantics or arithmetic_slip, explicitly check these two narrower categories first and prefer them when they apply:
- unit_conversion_error: does the error involve mixing up units (feet vs inches, dollars vs cents, minutes vs hours)? If yes, use this category even if the model also misread the problem.
- no_sanity_check: did the model's response reach an implausible result (negative price, negative count, absurd magnitude) without flagging or reconsidering it? If yes, use this category, since the interesting failure is the lack of a plausibility check, not just the upstream arithmetic/semantic mistake that produced it.

Respond with STRICT JSON only, no markdown fences, in this exact shape:
{{"category": "<one of the 12 slugs above>", "explanation": "<one sentence, specific to this example, citing the actual numbers involved>"}}
"""


def build_user_prompt(err: dict) -> str:
    return (
        f"Question: {err['question']}\n\n"
        f"Reference solution: {err['reference_solution']}\n"
        f"Reference answer: {err['reference_answer']}\n\n"
        f"Model's raw response: {err['model_response']}\n\n"
        f"Model's extracted answer: {err['extracted_answer']}\n"
    )


def classify_one(client: OpenAI, err: dict) -> dict:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(err)},
        ],
        temperature=0.2,
        top_p=1,
        max_tokens=2048,
        stream=False,
    )
    msg = completion.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content or ""

    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    return {
        "example_id": err["example_id"],
        "reference_answer": err["reference_answer"],
        "extracted_answer": err["extracted_answer"],
        "category": (parsed or {}).get("category", "PARSE_FAILED"),
        "explanation": (parsed or {}).get("explanation", ""),
        "raw_model_output": content,
        "reasoning_trace": reasoning,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing errors.jsonl")
    ap.add_argument("--sample-file", default=None, help="optional subset jsonl to classify instead of full errors.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip example_ids already present in --out and append")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("NVIDIA_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
    if not api_key:
        sys.exit("NVIDIA_API_KEY not found in environment or .env")

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=60.0)

    src_path = args.sample_file or os.path.join(args.run, "errors.jsonl")
    with open(src_path) as f:
        errors = [json.loads(l) for l in f]
    if args.limit:
        errors = errors[: args.limit]

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        errors = [e for e in errors if e["example_id"] not in done_ids]
        print(f"resuming: {len(done_ids)} already done, {len(errors)} remaining", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a" if args.resume else "w") as out_f:
        for i, err in enumerate(errors):
            for attempt in range(3):
                try:
                    result = classify_one(client, err)
                    break
                except Exception as e:
                    print(f"  retry {attempt} for example {err['example_id']}: {e}", file=sys.stderr)
                    time.sleep(2 ** attempt)
            else:
                result = {"example_id": err["example_id"], "category": "API_FAILED", "explanation": "", "raw_model_output": "", "reasoning_trace": None}
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            print(f"[{i+1}/{len(errors)}] ex={result['example_id']} -> {result['category']}")


if __name__ == "__main__":
    main()
