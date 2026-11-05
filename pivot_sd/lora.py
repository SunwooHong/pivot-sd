"""LoRA settings for training and evaluation."""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]
FULL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def load_model_for_eval(model_path: str, device):
    """Load a base model, or a Trainer LoRA checkpoint.

    Eval jobs read ``adapter_config.json``, load ``base_model_name_or_path``,
    then attach the adapter with ``PeftModel``.
    """
    import json
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

    def tokenizer_for(path: str):
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:
            if "LLaDAConfig" not in str(exc):
                raise
            return PreTrainedTokenizerFast.from_pretrained(path)

    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_config):
        with open(adapter_config, "r") as handle:
            base_path = json.load(handle)["base_model_name_or_path"]
        from peft import PeftModel

        tokenizer = tokenizer_for(base_path)
        base_model = AutoModel.from_pretrained(
            base_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, model_path).to(device).eval()
        print(f"[eval] loaded LoRA adapter from {model_path}", flush=True)
        return model, tokenizer

    tokenizer = tokenizer_for(model_path)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, tokenizer


def resolve_lora_target_modules(modules: Optional[Sequence[str]]) -> List[str]:
    if modules:
        return [str(m) for m in modules]
    return list(DEFAULT_LORA_TARGET_MODULES)


def add_lora_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        metavar="NAME",
        help="LoRA target module names. The reported runs use q_proj k_proj v_proj o_proj up_proj down_proj gate_proj.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay. The reported runs use 0.1.",
    )
