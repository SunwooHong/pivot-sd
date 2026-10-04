"""Train a Pivot-SD checkpoint, or one of the reported ablations.

Example (main method)::

    python -m pivot_sd.train --method pivot_sd --dataset math \\
        --model-name /path/to/LLaDA-8B-Instruct \\
        --pivot-jsonl pivots_ig.jsonl \\
        --train-json math_train.jsonl \\
        --output-dir runs/math_pivot_sd
"""
from __future__ import annotations

import argparse
import sys

from pivot_sd.methods import FULL_LORA_TARGETS, TRAIN_DEFAULTS, resolve_method
from pivot_sd.train_entry import main as train_main


def build_argv(args: argparse.Namespace) -> list[str]:
    cfg = resolve_method(args.method, args.dataset)
    max_length = args.max_length if args.max_length is not None else cfg["max_length"]
    argv = [
        "pivot_sd.train",
        "--model-name",
        args.model_name,
        "--pivot-jsonl",
        args.pivot_jsonl,
        "--train-json",
        args.train_json,
        "--output-dir",
        args.output_dir,
        "--job-name",
        args.job_name or f"{args.dataset}_{args.method}",
        "--positive-mask-loss",
        cfg["positive_mask_loss"],
        "--negative-mask-loss",
        cfg["negative_mask_loss"],
        "--negative-weight-multiplier",
        str(cfg["lambda_neg"]),
        "--pivot-block-size",
        str(cfg["pivot_block_size"]),
        "--lora-target-modules",
        *FULL_LORA_TARGETS,
        "--weight-decay",
        str(TRAIN_DEFAULTS["weight_decay"]),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--num-epochs",
        str(args.num_epochs),
        "--max-steps",
        str(args.max_steps),
        "--max-length",
        str(max_length),
        "--seed",
        str(args.seed),
        "--save-steps",
        str(args.save_steps),
        "--report-to",
        args.report_to,
    ]
    if cfg["code_prompt_style"]:
        argv.extend(["--code-prompt-style", cfg["code_prompt_style"]])
    if cfg["acecode_include_tests_in_prompt"]:
        argv.append("--acecode-include-tests-in-prompt")
    if args.save_at_steps:
        argv.extend(["--save-at-steps", args.save_at_steps])
    return argv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=(
        "pivot_sd",
        "positive_all",
        "positive_pivot",
        "entropy_pivot",
        "posneg_all",
        "random_step",
        "random_token",
    ))
    p.add_argument("--dataset", required=True, choices=("math", "gsm8k", "acecode"))
    p.add_argument("--model-name", required=True)
    p.add_argument("--pivot-jsonl", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--job-name", default=None)
    p.add_argument("--learning-rate", type=float, default=TRAIN_DEFAULTS["learning_rate"])
    p.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    p.add_argument("--grad-accum-steps", type=int, default=TRAIN_DEFAULTS["grad_accum_steps"])
    p.add_argument("--num-epochs", type=int, default=TRAIN_DEFAULTS["num_epochs"])
    p.add_argument("--max-steps", type=int, default=TRAIN_DEFAULTS["max_steps"])
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-at-steps", default=None, help="Comma-separated global steps, e.g. 500,1000")
    p.add_argument("--report-to", default="none")
    args = p.parse_args()
    sys.argv = build_argv(args)
    train_main()


if __name__ == "__main__":
    main()
