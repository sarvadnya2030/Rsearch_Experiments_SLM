"""Retry specific example_ids from a classify_errors_llm.py output file with a larger
token budget (for cases where reasoning consumed the whole budget before the final JSON),
patching them in place.

Usage:
    python scripts/retry_failed_classifications.py --run results/exp00_baseline/run_20260905_194735 \
        --out results/exp00_baseline/error_taxonomy_full.jsonl
"""
import argparse
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from classify_errors_llm import build_user_prompt, SYSTEM_PROMPT  # noqa: E402


def classify_one(client, err, max_tokens):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(err)},
        ],
        temperature=0.2,
        top_p=1,
        max_tokens=max_tokens,
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        for line in open(env_path):
            if line.startswith("NVIDIA_API_KEY="):
                api_key = line.strip().split("=", 1)[1]

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=90.0)

    with open(args.out) as f:
        rows = [json.loads(l) for l in f]
    failed_ids = {r["example_id"] for r in rows if r["category"] == "PARSE_FAILED"}
    print(f"retrying {len(failed_ids)} failed ids: {sorted(failed_ids)}")

    errors_by_id = {}
    with open(os.path.join(args.run, "errors.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e["example_id"] in failed_ids:
                errors_by_id[e["example_id"]] = e

    for ex_id in failed_ids:
        result = classify_one(client, errors_by_id[ex_id], args.max_tokens)
        for i, r in enumerate(rows):
            if r["example_id"] == ex_id:
                rows[i] = result
                break
        print(f"ex={ex_id} -> {result['category']}")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
