import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from data_utils import (
    get_acecode_questions,
    get_code_questions,
    get_countdown_questions,
    get_gsm8k_questions,
    get_math_questions,
    get_sudoku_questions,
    set_random_seed,
)
from peft import LoraConfig
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    coding_reward_func,
    countdown_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
)

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from trl import ModelConfig, TrlParser

from wd1.eval.countdown import CTDDataset
from wd1.eval.gsm8k import GSM8KDataset
from wd1.eval.math500 import MATH500Dataset
from wd1.eval.sudoku import SudokuDataset
from wd1.trainers.diffu_grpo_config import DiffuGRPOConfig
from wd1.trainers.diffu_grpo_trainer import DiffuGRPOTrainer
from wd1.trainers.eval_callback import AccuracyEvalCallback
from wd1.trainers.rev_grpo_ref_pol_trainer import RevDiffuRefPolGRPOTrainer
from wd1.trainers.rev_grpo_trainer import RevDiffuGRPOTrainer
from wd1.trainers.rev_grpo_trainer_psr import RevPSRDiffuGRPOTrainer

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

SUB_SAMPLE_MAP = {
    "gsm8k": 250,
    "math": 250,
    "countdown": -1,
    "sudoku": -1,
}


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _resolve_resume_checkpoint(grpo_config) -> Optional[str]:
    """Trainer.train() does not read TrainingArguments.resume_from_checkpoint; pass explicitly."""
    v = getattr(grpo_config, "resume_from_checkpoint", None)
    if v is None or v is False:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("false", "none", "null", "no"):
            return None
        return s
    return str(v)


def _train_eval_callbacks(grpo_config, val_dataset, tokenizer):
    if not grpo_config.use_train_accuracy_callback:
        return []
    return [
        AccuracyEvalCallback(
            val_dataset,
            tokenizer=tokenizer,
            gen_length=grpo_config.max_completion_length,
            temperature=0.0,
            steps=grpo_config.diffusion_steps,
            block_length=grpo_config.block_length,
            batch_size=grpo_config.per_device_eval_batch_size,
        )
    ]


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    val_dataset = None
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif grpo_config.dataset == "code":
        dataset = get_code_questions("train")
        reward_functions = [xmlcount_reward_func, coding_reward_func]
    elif grpo_config.dataset == "acecode":
        dataset = get_acecode_questions("train")
        _ace_style = os.environ.get("CODE_PROMPT_STYLE", "raw_with_tests").strip().lower()
        if _ace_style == "reasoning_answer":
            reward_functions = [
                xmlcount_reward_func,
                soft_format_reward_func,
                coding_reward_func,
            ]
        else:
            reward_functions = [coding_reward_func]
    else:
        raise ValueError(f"Unknown dataset: {grpo_config.dataset}")

    dataset = dataset.shuffle(seed=grpo_config.seed)

    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(
            range(0, len(dataset) - 500)
        )  # Leave last 500 for evaluation
    else:
        train_set = dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if model_config.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModel.from_pretrained(grpo_config.model_path, **load_kwargs).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    val_dataset = None
    if grpo_config.use_train_accuracy_callback:
        val_dataset = DATASET_MAP[grpo_config.dataset](
            tokenizer,
            subsample=SUB_SAMPLE_MAP[grpo_config.dataset],
            num_examples=0,
            add_reasoning=True,  # prefill for all models
        )
    eval_callbacks = _train_eval_callbacks(grpo_config, val_dataset, tokenizer)

    if model_config.lora_target_modules is not None:
        target_modules = list(model_config.lora_target_modules)
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    if is_main_process():
        print("Trainer type is: ", grpo_config.trainer_type)

    # Initialize and run trainer
    if grpo_config.trainer_type == "wll_d1_neg":
        # NSR + PSR + d1 objective
        trainer = RevDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=eval_callbacks,
        )
    elif grpo_config.trainer_type == "wll_d1_pos_only":
        trainer = RevPSRDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=eval_callbacks,
        )
    elif grpo_config.trainer_type == "d1":
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=eval_callbacks,
        )
    elif grpo_config.trainer_type == "wll_d1_neg_ref":
        # add reference policy regularisation for
        trainer = RevDiffuRefPolGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=eval_callbacks,
        )
    else:
        raise ValueError(f"Unknown trainer type: {grpo_config.trainer_type}")

    if is_main_process():
        wandb.init(project=grpo_config.wandb_project, name=grpo_config.run_name)

    resume_ckpt = _resolve_resume_checkpoint(grpo_config)
    if resume_ckpt and is_main_process():
        print(f"[train] resume_from_checkpoint={resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
