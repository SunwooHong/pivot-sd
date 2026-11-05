"""Evaluate a checkpoint the same way as the reported tables.

Math and GSM8K use greedy ``alg=default`` decoding. Length 256 uses 256 steps;
length 512 uses 512 steps. HumanEval+ and MBPP+ use the kodcode_v3 prompt and
the same decoder.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument(
        "--dataset",
        required=True,
        choices=("math", "gsm8k", "humaneval_plus", "mbpp_plus"),
    )
    p.add_argument("--gen-length", type=int, default=256, choices=(256, 512))
    p.add_argument("--data-path", default=None, help="Test JSON/JSONL for math or gsm8k")
    p.add_argument("--block-length", type=int, default=64)
    p.add_argument("--steps", type=int, default=None, help="Defaults to --gen-length")
    p.add_argument("--backbone", default="llada", choices=("llada", "dream"))
    p.add_argument("--alg", default=None)
    args, unknown = p.parse_known_args()
    steps = args.gen_length if args.steps is None else args.steps
    alg = args.alg or ("maskgit_plus" if args.backbone == "dream" else "default")

    if args.dataset in ("math", "gsm8k"):
        forwarded = [
            "--model_path", args.model_path,
            "--save_dir", args.save_dir,
            "--dataset", args.dataset,
            "--gen_length", str(args.gen_length),
            "--block_length", str(args.block_length),
            "--steps", str(steps),
            "--alg", alg,
            "--backbone", args.backbone,
            "--temperature", "0",
        ]
        if args.data_path:
            forwarded.extend(["--data_path", args.data_path])
        forwarded.extend(unknown)
        sys.argv = ["pivot_sd.evaluate", *forwarded]
        from pivot_sd.evaluate_reasoning import main as eval_main
        eval_main()
        return

    dataset = "humanevalplus" if args.dataset == "humaneval_plus" else "mbppplus"
    forwarded = [
        "--model_path", args.model_path,
        "--save_dir", args.save_dir,
        "--dataset", dataset,
        "--gen_length", str(args.gen_length),
        "--block_length", str(args.block_length),
        "--steps", str(steps),
        "--alg", alg,
        "--backbone", args.backbone,
        "--code_prompt_style", "kodcode_v3",
    ]
    forwarded.extend(unknown)
    sys.argv = ["pivot_sd.evaluate", *forwarded]
    from pivot_sd.evaluate_code import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
