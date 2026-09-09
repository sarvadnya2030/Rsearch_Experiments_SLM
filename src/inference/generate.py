"""
Generation loop for Experiment 0: load Qwen3-0.6B-Base, generate a
completion per GSM8K question, time it, and return raw token counts.

VRAM safety: fp16, torch.inference_mode(), model loaded once and reused
across all examples, batch_size configurable (default 1).
"""

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    prompt: str
    response_text: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    hit_max_new_tokens: bool


class Generator:
    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device: str = "cuda",
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 42,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available. "
                "Set device='cpu' in the config, or check your CUDA/driver install."
            )

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # transformers 5.x deprecated `torch_dtype=` in favor of `dtype=`.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Batched version of generate() — left-padding required for correct
        batched generation on a causal LM (same technique as
        self_consistency_batched.py's B3 implementation, which measured a
        3.5x speedup over one-at-a-time generation)."""
        self.tokenizer.padding_side = "left"
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        padded_prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=pad_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        start = time.perf_counter()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start
        # Batch generation is one synchronous GPU call for the whole batch —
        # there's no true per-example wall-clock time, so this amortized
        # value is reported per example instead (still accurate for
        # tokens/second and for comparing batched vs. unbatched throughput).
        per_example_time = elapsed / len(prompts)

        results = []
        for i, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            full_response_ids = output_ids[i][padded_prompt_len:]
            eos_positions = (full_response_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                # Include the eos itself, matching generate()'s single-example
                # slicing (output_ids[0][prompt_tokens:]), which doesn't trim it.
                response_ids = full_response_ids[: eos_positions[0].item() + 1]
                hit_max = False
            else:
                response_ids = full_response_ids
                hit_max = True
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            response_tokens = response_ids.shape[0]
            results.append(GenerationResult(
                prompt=prompt,
                response_text=response_text,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=prompt_tokens + response_tokens,
                generation_time=per_example_time,
                tokens_per_second=(response_tokens / per_example_time) if per_example_time > 0 else 0.0,
                hit_max_new_tokens=hit_max,
            ))
        return results

    @torch.inference_mode()
    def generate(self, prompt: str) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        start = time.perf_counter()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        response_ids = output_ids[0][prompt_tokens:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response_tokens = response_ids.shape[0]
        total_tokens = prompt_tokens + response_tokens

        return GenerationResult(
            prompt=prompt,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            generation_time=elapsed,
            tokens_per_second=(response_tokens / elapsed) if elapsed > 0 else 0.0,
            hit_max_new_tokens=response_tokens >= self.max_new_tokens,
        )
