"""
GSM8K loading and prompt construction.

Dataset schema (verified via `datasets.load_dataset("openai/gsm8k", "main")`):
    - split "test": 1319 rows
    - columns: ["question", "answer"]
    - `answer` contains step-by-step reasoning with inline calculator
      annotations like "<<16-3-4=9>>" and ends with "#### <final_answer>"

Experiment 0 uses ONLY the test split, and only for inference. The
reference "answer" field is used solely for extracting the ground-truth
final numeric answer for scoring after generation — it is never placed
into the model's prompt.
"""

from dataclasses import dataclass

from datasets import Dataset, load_dataset


REFERENCE_ANSWER_MARKER = "####"


@dataclass(frozen=True)
class GSM8KExample:
    example_id: int
    question: str
    reference_solution: str
    reference_answer: str


def load_gsm8k_test(dataset_name: str = "openai/gsm8k", dataset_config: str = "main") -> Dataset:
    """Load the GSM8K test split only. Never loads train for this experiment."""
    return load_dataset(dataset_name, dataset_config, split="test")


def extract_reference_answer(raw_answer: str) -> str:
    """Pull the ground-truth final answer out of GSM8K's '#### N' suffix.

    Example:
        "...9 duck eggs...\\n#### 18"  ->  "18"
    """
    if REFERENCE_ANSWER_MARKER not in raw_answer:
        raise ValueError(f"GSM8K answer missing '{REFERENCE_ANSWER_MARKER}' marker: {raw_answer!r}")
    tail = raw_answer.split(REFERENCE_ANSWER_MARKER)[-1]
    return tail.strip().replace(",", "")


def to_examples(dataset: Dataset) -> list[GSM8KExample]:
    examples = []
    for idx, row in enumerate(dataset):
        examples.append(
            GSM8KExample(
                example_id=idx,
                question=row["question"],
                reference_solution=row["answer"],
                reference_answer=extract_reference_answer(row["answer"]),
            )
        )
    return examples


def build_completion_prompt(question: str) -> str:
    """Plain completion-style prompt — NOT a chat template.

    Qwen3-0.6B-Base's tokenizer_config carries a full Qwen3 chat_template
    (inherited from the instruction-tuned family's tooling), but the base
    model itself was never instruction-tuned to follow it. Using
    apply_chat_template on a true base model risks degenerate output
    since it was never trained on that format. Instead we use a minimal
    zero-shot completion prompt, consistent with how base models are
    conventionally evaluated on GSM8K in the literature.
    """
    return f"Question: {question}\nAnswer:"
