"""Sample denoising trajectories and write stepwise JSON.

Defaults match the runs that produced the sample-200 pivot sets:
``alg=default`` (one committed token per step), temperature 0.9, 4 samples,
length 256, block 64, 256 steps, and stepwise traces on.

Math and GSM8K use the reasoning chat template. AceCode uses the with-tests
code prompt (``kodcode_v3``).
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("math", "gsm8k", "acecode"))
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True, help="Prompt JSONL")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=64)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--alg", default="default", choices=("default", "random", "topk_margin", "entropy"))
    p.add_argument("--slice-start", type=int, default=None)
    p.add_argument("--slice-end", type=int, default=None)
    p.add_argument("--save-steps", action="store_true", default=True)
    p.add_argument("--no-save-steps", action="store_false", dest="save_steps")
    args, unknown = p.parse_known_args()

    forwarded = [
        "--model_path", args.model_path,
        "--dataset", "acecode" if args.dataset == "acecode" else args.dataset,
        "--data_path", args.data_path,
        "--save_dir", args.save_dir,
        "--gen_length", str(args.gen_length),
        "--block_length", str(args.block_length),
        "--steps", str(args.steps),
        "--temperature", str(args.temperature),
        "--num_samples", str(args.num_samples),
        "--alg", args.alg,
    ]
    if args.slice_start is not None:
        forwarded.extend(["--slice_start", str(args.slice_start)])
    if args.slice_end is not None:
        forwarded.extend(["--slice_end", str(args.slice_end)])
    if args.save_steps:
        forwarded.append("--save_steps")
    if args.dataset == "acecode":
        forwarded.extend(["--code_prompt_style", "kodcode_v3"])
    forwarded.extend(unknown)
    sys.argv = ["pivot_sd.generate", *forwarded]
    if args.dataset == "acecode":
        from pivot_sd.generate_code import main as gen_main
    else:
        from pivot_sd.generate_reasoning import main as gen_main
    gen_main()


if __name__ == "__main__":
    main()
