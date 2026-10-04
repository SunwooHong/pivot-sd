"""Training recipes that match the reported Pivot-SD runs and ablations.

Loss scopes are the ``IGPivotTrainer`` modes used in the sample-200 jobs.
``lambda_neg`` multiplies the unlikelihood term. ``lambda_neg = 0`` drops
incorrect trajectories from the dataset (positive-only ablations).
Advantage magnitude is not applied to the loss (``--no-advantage-in-loss``).
"""
from __future__ import annotations

from typing import Any, Dict

# Dataset-specific unlikelihood weights from the sample-200 training scripts.
LAMBDA_NEG = {
    "math": 1.0,
    "gsm8k": 5.0,
    "acecode": 0.02,
}

FULL_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

# Shared optimization settings from the sample-200 full-LoRA jobs.
TRAIN_DEFAULTS: Dict[str, Any] = {
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "max_steps": 1000,
    "num_epochs": 1,
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "max_length": 512,
    "acecode_max_length": 768,
}


def _base(dataset: str) -> Dict[str, Any]:
    if dataset not in LAMBDA_NEG:
        raise ValueError(f"dataset must be one of {sorted(LAMBDA_NEG)}, got {dataset!r}")
    max_length = TRAIN_DEFAULTS["acecode_max_length"] if dataset == "acecode" else TRAIN_DEFAULTS["max_length"]
    cfg: Dict[str, Any] = {
        "dataset": dataset,
        "positive_mask_loss": "pivot_only",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": LAMBDA_NEG[dataset],
        "no_advantage_in_loss": True,
        "pivot_block_size": 64,
        "max_length": max_length,
        "code_prompt_style": "kodcode_v3" if dataset == "acecode" else None,
        "acecode_include_tests_in_prompt": dataset == "acecode",
        "selector": "ig_density",
    }
    return cfg


# Keys are the public --method names.
METHODS = {
    # Main result. IG-density states, loss only on the committed pivot token.
    "pivot_sd": {
        "positive_mask_loss": "pivot_only",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": "dataset",
        "selector": "ig_density",
    },
    # Ablation: CE on every supervised masked token in the IG state. No negatives.
    "positive_all": {
        "positive_mask_loss": "uniform",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": 0.0,
        "selector": "ig_density",
    },
    # Ablation: CE only on the pivot. No negatives.
    "positive_pivot": {
        "positive_mask_loss": "pivot_only",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": 0.0,
        "selector": "ig_density",
    },
    # Ablation: states chosen by token entropy, CE only on that token.
    "entropy_pivot": {
        "positive_mask_loss": "pivot_only",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": 0.0,
        "selector": "entropy",
    },
    # Ablation: CE and unlikelihood on every supervised masked token in the IG state.
    # The uniform_uniform jobs did not pass --negative-weight-multiplier, so this is
    # the trainer default 0.5, not the dataset-specific Pivot-SD weight.
    "posneg_all": {
        "positive_mask_loss": "uniform",
        "negative_mask_loss": "uniform",
        "lambda_neg": 0.5,
        "selector": "ig_density",
    },
    # Ablation: random decode steps, CE / unlikelihood on the token committed there.
    "random_step": {
        "positive_mask_loss": "pivot_only",
        "negative_mask_loss": "pivot_only",
        "lambda_neg": "dataset",
        "selector": "random_step",
    },
    # Ablation: IG states, but the supervised index is a random masked token in that block.
    "random_token": {
        "positive_mask_loss": "block_random_pivot_only",
        "negative_mask_loss": "block_random_pivot_only",
        "lambda_neg": "dataset",
        "selector": "ig_density",
    },
}


def resolve_method(name: str, dataset: str) -> Dict[str, Any]:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; choose from {sorted(METHODS)}")
    cfg = _base(dataset)
    spec = METHODS[name]
    cfg["method"] = name
    cfg["selector"] = spec["selector"]
    cfg["positive_mask_loss"] = spec["positive_mask_loss"]
    cfg["negative_mask_loss"] = spec["negative_mask_loss"]
    lam = spec["lambda_neg"]
    cfg["lambda_neg"] = LAMBDA_NEG[dataset] if lam == "dataset" else float(lam)
    return cfg
