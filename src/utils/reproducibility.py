"""
Seeding and environment-capture utilities for reproducible runs.
"""

import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def capture_environment(model_name: str, dataset_name: str, generation_config: dict, seed: int) -> dict:
    import transformers

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "generation_config": generation_config,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }


def write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
