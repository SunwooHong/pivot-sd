"""LoRA training entry point used by ``python -m pivot_sd.train``."""
from __future__ import annotations

import argparse
import inspect
import os
import random
from typing import Callable

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from pivot_sd.trainer import IGPivotDataset, IGPivotTrainer, make_data_collator_with_padding
from pivot_sd.lora import add_lora_training_args, resolve_lora_target_modules
from pivot_sd.save_utils import apply_save_at_steps


def _patch_torch_load_for_trainer_resume() -> Callable[[], None]:
    """Allow Trainer to resume optimizer state on PyTorch 2.6+ (weights_only=True rejects numpy)."""
    orig = torch.load

    def _wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        if "weights_only" not in kwargs and "weights_only" in inspect.signature(orig).parameters:
            kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[method-assign]

    def _restore() -> None:
        torch.load = orig  # type: ignore[method-assign]

    return _restore


def init_seed(seed: int, *, cudnn_deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":409:8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--pivot-jsonl", type=str, required=True)
    p.add_argument("--train-json", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--job-name", type=str, default="pivot_sd")
    p.add_argument("--mask-id", type=int, default=-1, help="Default: tokenizer.mask_token_id")
    p.add_argument("--debugging", action="store_true")
    p.add_argument("--debug-max-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cudnn-deterministic", action="store_true")
    p.add_argument(
        "--positive-mask-loss",
        type=str,
        default="pivot_only",
        choices=("uniform", "pivot_only", "block_random_pivot_only"),
    )
    p.add_argument(
        "--pivot-block-size",
        type=int,
        default=64,
        help="Gen block length for block_random_pivot_only (positive and/or negative; aligned from prompt_len).",
    )
    p.add_argument(
        "--negative-mask-loss",
        type=str,
        default="pivot_only",
        choices=("uniform", "pivot_only", "block_random_pivot_only"),
    )
    p.add_argument("--negative-weight-multiplier", type=float, default=0.5)
    p.add_argument(
        "--keep-negative-advantage-rows",
        action="store_true",
        help="When --negative-weight-multiplier is 0, still load pivot rows with advantage<0 "
        "(they get zero loss; default is to drop them before stepwise/tokenization).",
    )
    p.add_argument(
        "--completion-source",
        type=str,
        default="stepwise",
        choices=("stepwise", "train"),
    )
    p.add_argument(
        "--code-prompt-style",
        type=str,
        default="kodcode_v3",
        help="AceCode prompt. Training uses kodcode_v3 when the row has test_cases.",
    )
    p.add_argument(
        "--acecode-include-tests-in-prompt",
        action="store_true",
        help="AceCode: include the first 3 test cases in the user prompt.",
    )
    p.add_argument(
        "--stepwise-positions",
        type=str,
        default="auto",
        choices=("auto", "absolute", "relative_gen", "file_gen_start"),
    )
    p.add_argument("--report-to", type=str, default="none")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--save-steps", type=int, default=1000, metavar="N")
    p.add_argument(
        "--save-at-steps",
        type=str,
        default=None,
        help="Comma-separated global steps to save checkpoints (overrides --save-steps).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set (>0), stop after this many global optimizer steps (overrides full epoch schedule; good for smoke tests).",
    )
    p.add_argument(
        "--stop-at-steps",
        type=int,
        default=None,
        help="Early-stop at this global step while keeping --max-steps for the LR schedule.",
    )
    p.add_argument(
        "--pivot-dataset-cache",
        type=str,
        default=None,
        help="Pickle file: save/reload validated pivot rows + path_final_text after init scan. "
        "Fingerprint includes pivot/train mtimes, tokenizer name, max_length, mask_id, completion, "
        "stepwise_positions, positive_advantage_only. Env PIVOT_DATASET_CACHE used if this is unset.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="DIR",
        help="Trainer checkpoint directory (e.g. output_dir/job_name/checkpoint-6000). "
        "Restores LoRA weights, optimizer, LR scheduler, RNG, and global_step; keep the same "
        "model, LoRA layout, data, and output_dir/job_name as the original run.",
    )
    add_lora_training_args(p)
    return p.parse_args()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=resolve_lora_target_modules(args.lora_target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    model.config.use_cache = False
    return tokenizer, model


def main() -> None:
    args = parse_args()
    if args.negative_weight_multiplier < 0:
        raise SystemExit("--negative-weight-multiplier must be >= 0")
    resume_ckpt = (args.resume_from_checkpoint or "").strip() or None
    if resume_ckpt:
        resume_ckpt = os.path.abspath(resume_ckpt)
        if not os.path.isdir(resume_ckpt):
            raise SystemExit(f"--resume-from-checkpoint is not a directory: {resume_ckpt}")
        if not os.path.isfile(os.path.join(resume_ckpt, "trainer_state.json")):
            raise SystemExit(
                f"--resume-from-checkpoint must contain trainer_state.json (full Trainer checkpoint): {resume_ckpt}"
            )
    init_seed(args.seed, cudnn_deterministic=not args.no_cudnn_deterministic)
    tokenizer, model = load_model_and_tokenizer(args)
    if args.mask_id < 0:
        if tokenizer.mask_token_id is None:
            raise SystemExit("Tokenizer has no mask_token_id. Pass --mask-id.")
        args.mask_id = int(tokenizer.mask_token_id)

    positive_advantage_only = args.negative_weight_multiplier == 0 and not args.keep_negative_advantage_rows
    init_cache = (args.pivot_dataset_cache or os.environ.get("PIVOT_DATASET_CACHE") or "").strip() or None
    ds = IGPivotDataset(
        args.pivot_jsonl,
        args.train_json,
        tokenizer,
        args.max_length,
        mask_id=args.mask_id,
        use_stepwise_completion=(args.completion_source == "stepwise"),
        stepwise_positions=args.stepwise_positions,
        positive_advantage_only=positive_advantage_only,
        init_cache_path=init_cache,
        code_prompt_style=args.code_prompt_style,
        acecode_include_tests_in_prompt=args.acecode_include_tests_in_prompt,
    )
    print(f"[train] processed pivot training rows: {len(ds)}")
    if len(ds) == 0:
        raise SystemExit("Empty pivot dataset — check pivot-jsonl path and contents.")

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    rt = args.report_to.strip().lower()
    if rt == "none":
        report_to = "none"
    else:
        report_to = [x.strip() for x in args.report_to.split(",") if x.strip()]
        if not report_to:
            report_to = "none"

    save_kw: dict = {}
    trainer_callbacks: list = []
    apply_save_at_steps(
        save_at_steps=args.save_at_steps,
        save_steps=args.save_steps,
        save_kw=save_kw,
        trainer_callbacks=trainer_callbacks,
        stop_at_steps=args.stop_at_steps,
    )

    ta_common = dict(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        bf16=True,
        seed=args.seed,
        report_to=report_to,
        run_name=args.run_name or args.job_name,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        weight_decay=args.weight_decay,
        **save_kw,
    )
    if args.max_steps is not None and args.max_steps > 0:
        ta_common["max_steps"] = int(args.max_steps)
    training_args = TrainingArguments(**ta_common)

    print(
        f"[train] LoRA targets={resolve_lora_target_modules(args.lora_target_modules)} "
        f"weight_decay={args.weight_decay}",
        flush=True,
    )

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=make_data_collator_with_padding(tokenizer),
        tokenizer=tokenizer,
        positive_mask_loss=args.positive_mask_loss,
        negative_mask_loss=args.negative_mask_loss,
        mask_id=args.mask_id,
        negative_weight_multiplier=args.negative_weight_multiplier,
        pivot_block_size=args.pivot_block_size,
        debug_train=args.debugging,
        debug_max_steps=args.debug_max_steps,
    )
    if trainer_callbacks:
        trainer_kw["callbacks"] = trainer_callbacks

    trainer = IGPivotTrainer(**trainer_kw)
    restore_torch_load: Callable[[], None] | None = None
    if resume_ckpt:
        print(f"[train] resuming from checkpoint: {resume_ckpt}")
        restore_torch_load = _patch_torch_load_for_trainer_resume()
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    finally:
        if restore_torch_load is not None:
            restore_torch_load()


if __name__ == "__main__":
    main()
