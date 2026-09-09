"""
Stage 3 — fill in the ~5% of teacher traces that failed (is_correct=False)
instead of just discarding them, so Arms A/C/D can reach 100% usable
coverage of the examples generated so far.

Runs independently of generate_teacher_traces.py's main --resume process
(separate output file, separate API calls) — safe to run concurrently.

For each wrong example, retries generation up to --max-attempts times
(same teacher prompt, temperature=0.7 so each attempt genuinely differs,
not a repeat of the same failure) and keeps the FIRST attempt whose
extracted answer matches ground truth. Never fabricates or forces a
match — an example that fails every attempt this pass is written back
with is_correct=False, recovered_by_retry=False, and a running
total_attempts_tried count.

With --resume, only examples already RECOVERED (recovered_by_retry=True)
in --out are treated as done and skipped. Examples still wrong after a
previous pass are retried again — the output file is rewritten each run
keyed by example_id (latest attempt wins), so repeated 15-min passes
keep hammering on the same still-failing examples across the whole
teacher-generation run instead of giving up on them after one pass.

Usage:
    python scripts/retry_wrong_teacher_traces.py \
        --teacher-file results/exp03_teacher/traces_n2000.jsonl \
        --out results/exp03_teacher/traces_retried.jsonl \
        --max-attempts 5 --resume
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.generate_teacher_traces import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    contamination_similarity,
)
from src.evaluation.answer_extraction import extract_predicted_answer_with_method, is_correct  # noqa: E402


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
    return {"content": msg.content or "", "reasoning_trace": getattr(msg, "reasoning_content", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-attempts", type=int, default=2, help="API attempts per pass for each still-wrong example")
    ap.add_argument(
        "--max-total-attempts",
        type=int,
        default=3,
        help="once an example's cumulative attempts across passes reaches this, flag needs_manual_fix and stop auto-retrying it",
    )
    ap.add_argument("--resume", action="store_true", help="skip example_ids already present in --out")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls (I/O-bound, safe to parallelize)")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        for line in open(env_path):
            if line.startswith("NVIDIA_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=httpx.Timeout(90.0, connect=15.0),
    )

    with open(args.teacher_file) as f:
        all_rows = [json.loads(line) for line in f]
    wrong_rows = [r for r in all_rows if not r.get("is_correct")]
    print(f"Loaded {len(all_rows)} teacher rows, {len(wrong_rows)} wrong -> retrying", flush=True)

    # example_id -> most recent output record (recovered or still-wrong).
    out_records: dict[int, dict] = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    out_records[rec["example_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue

    already_recovered_ids = {eid for eid, rec in out_records.items() if rec.get("recovered_by_retry")}
    # Examples that have already burned through MAX_TOTAL_AUTO_ATTEMPTS across
    # passes are handed off for manual fixing rather than retried forever —
    # skip them here too so the API budget isn't wasted on known-stuck cases.
    needs_manual_ids = {eid for eid, rec in out_records.items() if rec.get("needs_manual_fix")}
    if args.resume:
        skip_ids = already_recovered_ids | needs_manual_ids
        wrong_rows = [r for r in wrong_rows if r["example_id"] not in skip_ids]
        print(
            f"resuming: {len(already_recovered_ids)} already recovered, "
            f"{len(needs_manual_ids)} flagged needs_manual_fix (both skipped), "
            f"{len(wrong_rows)} still need retrying",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_recovered = 0
    n_still_wrong = 0
    n_done = 0
    write_lock = threading.Lock()

    def process_row(r: dict) -> tuple[dict, bool]:
        question = r["question"]
        reference_answer = r["reference_answer"]
        reference_solution = r["reference_solution"]
        prior_attempts = out_records.get(r["example_id"], {}).get("total_attempts_tried", 0)

        recovered = None
        for attempt in range(args.max_attempts):
            try:
                gen = generate_one(client, question)
            except Exception as e:
                print(f"  API error ex={r['example_id']} attempt={attempt}: {e}", file=sys.stderr, flush=True)
                time.sleep(2**attempt)
                continue

            extracted_answer, method = extract_predicted_answer_with_method(gen["content"])
            if is_correct(extracted_answer, reference_answer):
                similarity = contamination_similarity(gen["content"], reference_solution)
                recovered = {
                    "example_id": r["example_id"],
                    "question": question,
                    "reference_solution": reference_solution,
                    "reference_answer": reference_answer,
                    "teacher_trace": gen["content"],
                    "teacher_reasoning_internal": gen["reasoning_trace"],
                    "extracted_answer": extracted_answer,
                    "extraction_method": method,
                    "is_correct": True,
                    "contamination_similarity": round(similarity, 3),
                    "flagged_possible_contamination": similarity > 0.5,
                    "attempts_tried": attempt + 1,
                    "recovered_by_retry": True,
                }
                break

        if recovered:
            recovered["total_attempts_tried"] = prior_attempts + recovered["attempts_tried"]
            return recovered, True

        still_wrong = dict(r)
        still_wrong["attempts_tried"] = args.max_attempts
        still_wrong["total_attempts_tried"] = prior_attempts + args.max_attempts
        still_wrong["recovered_by_retry"] = False
        still_wrong["needs_manual_fix"] = still_wrong["total_attempts_tried"] >= args.max_total_attempts
        return still_wrong, False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, r): r for r in wrong_rows}
        for fut in as_completed(futures):
            r = futures[fut]
            record, was_recovered = fut.result()

            with write_lock:
                out_records[r["example_id"]] = record
                n_done += 1
                if was_recovered:
                    n_recovered += 1
                    print(f"[{n_done}/{len(wrong_rows)}] ex={r['example_id']} RECOVERED after {record['total_attempts_tried']} total attempt(s)", flush=True)
                else:
                    n_still_wrong += 1
                    tag = " -> NEEDS_MANUAL_FIX" if record["needs_manual_fix"] else ""
                    print(f"[{n_done}/{len(wrong_rows)}] ex={r['example_id']} still wrong after {record['total_attempts_tried']} total attempts{tag}", flush=True)

                # Rewrite the whole file each completion so a kill loses at
                # most the in-flight batch, not the accumulated dedup state.
                with open(args.out, "w") as out_f:
                    for rec in out_records.values():
                        out_f.write(json.dumps(rec) + "\n")

    n_needs_manual = sum(1 for rec in out_records.values() if rec.get("needs_manual_fix"))
    print(f"\ndone: {n_recovered} recovered, {n_still_wrong} still wrong this pass -> {args.out}")
    print(
        f"cumulative: {len(already_recovered_ids) + n_recovered}/{len(out_records)} recovered, "
        f"{n_needs_manual} need manual fix"
    )


if __name__ == "__main__":
    main()
