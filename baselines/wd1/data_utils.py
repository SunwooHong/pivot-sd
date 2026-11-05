import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from reward_func import extract_hash_answer


@lru_cache(maxsize=1)
def _get_prompt_filter_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _prompt_messages_token_count(messages, tokenizer) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def filter_dataset_drop_long_prompts(
    dataset: Dataset,
    max_tokens: int,
    *,
    dataset_name: str = "dataset",
) -> Dataset:
    """Drop rows whose chat ``prompt`` exceeds ``max_tokens`` (see ``PROMPT_LENGTH_POLICY``)."""
    policy = os.environ.get("PROMPT_LENGTH_POLICY", "drop").strip().lower()
    if policy in ("truncate", "none", "off", "false", "0", "no"):
        return dataset

    model_path = os.environ.get("MODEL_PATH", "GSAI-ML/LLaDA-8B-Instruct")
    tokenizer = _get_prompt_filter_tokenizer(model_path)
    n_before = len(dataset)

    def _keep(example):
        return _prompt_messages_token_count(example["prompt"], tokenizer) <= max_tokens

    filtered = dataset.filter(_keep)
    n_after = len(filtered)
    print(
        f"[{dataset_name}] PROMPT_LENGTH_POLICY=drop: kept {n_after}/{n_before} "
        f"(max_prompt_tokens={max_tokens})",
        flush=True,
    )
    return filtered


def _max_prompt_drop_tokens() -> int:
    return int(os.environ.get("MAX_PROMPT_DROP_TOKENS", "512"))


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_local = os.path.normpath(
        os.path.join(cur_path, "..", "dataset", "gsm8k_train_s2000.jsonl")
    )
    local_path = os.path.expanduser(os.environ.get("GSM8K_TRAIN_JSON", default_local))

    if os.path.isfile(local_path):
        data = load_dataset("json", data_files=local_path, split="train")  # type: ignore[arg-type]
    else:
        import warnings

        warnings.warn(
            f"No local GSM8K file at {local_path!r} (set GSM8K_TRAIN_JSON); using openai/gsm8k main/{split}.",
            stacklevel=2,
        )
        data = load_dataset("openai/gsm8k", "main")[split]

    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return filter_dataset_drop_long_prompts(
        data, _max_prompt_drop_tokens(), dataset_name="gsm8k"
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_local = os.path.normpath(
        os.path.join(cur_path, "..", "dataset", "math_train_s2000.jsonl")
    )
    local_path = os.path.expanduser(os.environ.get("MATH_TRAIN_JSON", default_local))

    if os.path.isfile(local_path):
        data = load_dataset("json", data_files=local_path, split="train")  # type: ignore[arg-type]
    else:
        import warnings

        warnings.warn(
            f"No local math file at {local_path!r} (set MATH_TRAIN_JSON); using ankner/math-500/{split}.",
            stacklevel=2,
        )
        data = load_dataset("ankner/math-500", split=split)  # type: ignore

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return filter_dataset_drop_long_prompts(
        data, _max_prompt_drop_tokens(), dataset_name="math"
    )


def get_code_questions(split="train") -> Dataset:
    """KodCode RL: local JSONL if ``KODCODE_TRAIN_JSON`` (or default under ``../dataset/``) exists, else Hub."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_local = os.path.normpath(
        os.path.join(cur_path, "..", "dataset", "kodcode_train_2k.jsonl")
    )
    local_path = os.path.expanduser(os.environ.get("KODCODE_TRAIN_JSON", default_local))

    if os.path.isfile(local_path):
        data = load_dataset("json", data_files=local_path, split="train")  # type: ignore[arg-type]
    else:
        import warnings

        warnings.warn(
            f"No local KodCode file at {local_path!r} (set KODCODE_TRAIN_JSON); "
            f"using KodCode/KodCode-Light-RL-10K/{split}.",
            stacklevel=2,
        )
        data = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)
        data = data.train_test_split(test_size=0.1, seed=42)["train"]

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{SYSTEM_PROMPT}\n\nYou are a coding expert. You will be given a coding problem to solve. "
                        f"Solve it step by step. \n\n{x['question']}"
                    ),
                }
            ],
            "answer": {"solution": x["solution"], "tests": x["test"]},
        }
    )


ACECODE_RAW_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You must write complete, valid Python code "
    "to solve the user's problem."
)

ACECODE_RAW_USER_FOOTER = (
    "You must output ONLY raw Python code.\n"
    "Do not include any conversational text or explanations before or after the code."
)

REASONING_ANSWER_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step. Ensure you wrap the answer in ```python```

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
```python
Your code here
```
</answer>"""


def build_acecode_chat_messages(
    question: str,
    test_cases=None,
    *,
    code_prompt_style: str = "raw_with_tests",
    include_tests_in_prompt: bool = True,
    num_prompt_tests: int = 3,
) -> list:
    """AceCode chat prompts (``CODE_PROMPT_STYLE`` env in ``get_acecode_questions``)."""
    style = (code_prompt_style or "raw_with_tests").strip().lower()

    if style == "raw_with_tests":
        prompt_tests = list(test_cases)[:num_prompt_tests] if test_cases else []
        tests_str = "\n".join(str(t) for t in prompt_tests)
        user_parts = [
            question,
            "",
            "Your code should pass these tests:",
            "",
            tests_str,
            "",
            ACECODE_RAW_USER_FOOTER,
        ]
        user_text = "\n".join(user_parts)
        return [
            {"role": "system", "content": ACECODE_RAW_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    if style == "reasoning_answer":
        user_text = question
        if include_tests_in_prompt and test_cases:
            prompt_tests = list(test_cases)[:num_prompt_tests]
            tests_str = "\n".join(str(t) for t in prompt_tests)
            user_text = f"{question}\n\nYour code should pass these tests:\n\n{tests_str}"
        return [
            {"role": "system", "content": REASONING_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    raise ValueError(
        f"Unknown code_prompt_style={code_prompt_style!r}; use 'raw_with_tests' or 'reasoning_answer'."
    )


def _acecode_test_cases_to_harness(test_cases) -> str:
    """Join AceCode assert lines into a single script for ``coding_reward_func``."""
    if test_cases is None:
        return ""
    if isinstance(test_cases, str):
        return test_cases.strip()
    lines = []
    for tc in test_cases:
        line = (tc if isinstance(tc, str) else str(tc)).strip()
        if not line:
            continue
        if not line.startswith("assert"):
            line = f"assert {line}"
        lines.append(line)
    return "\n".join(lines)


def get_acecode_questions(split="train") -> Dataset:
    """AceCode RL: local JSONL via ``ACECODE_TRAIN_JSON`` (default: dataset/acecode_train_s2000.jsonl)."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_local = os.path.normpath(
        os.path.join(cur_path, "..", "dataset", "acecode_train_s2000.jsonl")
    )
    local_path = os.path.expanduser(os.environ.get("ACECODE_TRAIN_JSON", default_local))

    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"AceCode training file not found: {local_path!r}. Set ACECODE_TRAIN_JSON."
        )

    data = load_dataset("json", data_files=local_path, split="train")  # type: ignore[arg-type]

    code_prompt_style = os.environ.get("CODE_PROMPT_STYLE", "raw_with_tests")
    num_prompt_tests = int(os.environ.get("ACECODE_PROMPT_NUM_TESTS", "3"))

    if code_prompt_style.strip().lower() == "raw_with_tests":
        include_tests = True
    else:
        run_variant = os.environ.get("ACECODE_RUN_VARIANT", "no_tests").strip().lower()
        include_tests = os.environ.get("ACECODE_INCLUDE_TESTS_IN_PROMPT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if run_variant == "with_tests":
            include_tests = True
        elif run_variant == "no_tests":
            include_tests = False

    def _map_row(x):
        return {
            "prompt": build_acecode_chat_messages(
                x["question"],
                x["test_cases"],
                code_prompt_style=code_prompt_style,
                include_tests_in_prompt=include_tests,
                num_prompt_tests=num_prompt_tests,
            ),
            "answer": {
                "solution": "",
                "tests": _acecode_test_cases_to_harness(x["test_cases"]),
            },
        }

    data = data.map(_map_row)
    return filter_dataset_drop_long_prompts(
        data, _max_prompt_drop_tokens(), dataset_name="acecode"
    )
