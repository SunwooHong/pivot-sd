"""Greedy HumanEval+ and MBPP+ evaluation with the kodcode_v3 prompt."""
import os
import json
import re
import random
import torch
from tqdm import tqdm
import argparse

from pivot_sd.lora import load_model_for_eval
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

from pivot_sd.decode import decode_response, run_dir
from pivot_sd.parsers import Parser, test_solution

KODCODE_SYSTEM_PROMPT_V2 = (
    "You are an expert Python programmer. You must write complete, valid Python code "
    "to solve the user's problem."
)
V3_FOOTER = (
    "You must output ONLY raw Python code.\n"
    "Do not include any conversational text or explanations before or after the code."
)


def kodcode_v3_messages(prompt_body: str, tests=None) -> list:
    if tests:
        if isinstance(tests, (list, tuple)):
            tests_str = "\n".join(str(t) for t in tests)
        else:
            tests_str = str(tests)
        user_text = (
            f"{prompt_body}\n\n"
            f"Your code should pass these tests:\n\n{tests_str}\n\n"
            f"{V3_FOOTER}"
        )
    else:
        user_text = f"{prompt_body}\n\n{V3_FOOTER}"
    return [
        {"role": "system", "content": KODCODE_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_text},
    ]


def _run_dir(save_dir: str, benchmark: str, alg: str, gen_length: int, block_length: int, steps: int, num_samples: int, backbone: str = "llada") -> str:
    family = "Dream" if backbone == "dream" else "LLaDA"
    return run_dir(
        save_dir, family, f"{benchmark}/kodcode_v3", alg, gen_length, block_length, steps, num_samples
    )


def mbppplus_tests_for_prompt(ex: dict) -> list:
    return list(ex.get("test_list") or [])


def mbppplus_unit_test_for_scoring(ex: dict) -> str:
    if ex.get("test"):
        return str(ex["test"])
    parts = []
    imports = ex.get("test_imports") or []
    if imports:
        parts.append("\n".join(str(x) for x in imports))
    test_list = ex.get("test_list") or []
    if test_list:
        parts.append("\n".join(str(x) for x in test_list))
    return "\n\n".join(parts)


def extract_program_from_generation(generated_text: str) -> str | None:
    program = Parser.extract_answer_code(generated_text)
    if program is not None:
        return program.strip()
    code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", generated_text, re.DOTALL)
    if code_match:
        program = code_match.group(1).strip()
    else:
        program = generated_text.strip()
    if not program:
        return None
    lines = program.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and "test" in stripped.lower():
            cut = i
            break
        if stripped.startswith("assert ") or stripped == "assert":
            cut = i
            break
        if stripped.startswith("if __name__"):
            cut = i
            break
    program = "\n".join(lines[:cut]).rstrip()
    return program or None


def score_mbpp_d1(generated_text: str, unit_test: str, *, output_dir: str | None = None):
    program = extract_program_from_generation(generated_text)
    if program is None:
        return False, None, None
    test_code = f"{program}\n\n{unit_test}"
    passed = bool(test_solution(test_code, output_dir=output_dir))
    return passed, program, test_code


def load_humanevalplus_problems() -> dict:
    raw = load_dataset("evalplus/humanevalplus", split="test")
    problems = {}
    for row in raw:
        tid = str(row["task_id"])
        problems[tid] = {
            "task_id": tid,
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "entry_point": row["entry_point"],
            "test": row["test"],
        }
    return problems


def test_humaneval_plus(
    model, tokenizer, accelerator, save_dir,
    gen_length, steps, block_length,
    alg="default",
    temperature=0.0,
    k=(1,), n_workers=4, timeout=3.0,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False,
    backbone="llada",
):
    save_dir = _run_dir(save_dir, "humanevalplus", alg, gen_length, block_length, steps, num_samples, backbone)
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    problems = load_humanevalplus_problems()
    if test_size:
        random.seed(516)
        items = list(problems.items())
        data = dict(random.sample(items, test_size) if random_sampling else items[:test_size])
    else:
        data = problems

    indexed_problems = list(enumerate(data.items()))
    local_samples = []
    local_steps = []

    with accelerator.split_between_processes(indexed_problems) as local_data:
        for i, (task_id, info) in tqdm(
            local_data, desc="Generating completions for HumanEval+", disable=not accelerator.is_main_process,
        ):
            m = kodcode_v3_messages(info["prompt"])
            prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids_original = torch.tensor(tokenizer(prompt_text)["input_ids"]).to(accelerator.device).unsqueeze(0)
            task_samples = []
            task_steps = []
            for sample_idx in range(num_samples):
                x_output, used_steps = decode_response(
                    model, tokenizer, input_ids_original, gen_length, steps, block_length,
                    temperature=temperature,
                    step_save_dir=step_save_dir, example_idx=f"q{i}_s{sample_idx}",
                    alg=alg, backbone=backbone,
                )
                task_steps.append(used_steps)
                decoded_text = tokenizer.batch_decode(
                    x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True
                )[0].split(tokenizer.eos_token)[0]
                code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", decoded_text, re.DOTALL)
                code_only = code_match.group(1).strip() if code_match else decoded_text.strip()
                local_samples.append({
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "completion": code_only,
                    "used_steps": used_steps,
                })
                task_samples.append({
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "generation": decoded_text,
                    "completion": code_only,
                    "used_steps": used_steps,
                })
            local_steps.append({
                "task_id": task_id,
                "input_prompt": prompt_text,
                "avg_steps": sum(task_steps) / len(task_steps),
                "samples": task_samples,
            })

    accelerator.wait_for_everyone()
    all_samples = gather_object(local_samples)
    all_steps = gather_object(local_steps)

    if accelerator.is_main_process:
        all_steps = sorted(all_steps, key=lambda x: x["task_id"])
        samples_file = os.path.join(save_dir, "humaneval_samples.jsonl")
        write_jsonl(samples_file, all_samples)
        problems_ref_file = os.path.join(save_dir, "humanevalplus_problems.jsonl")
        write_jsonl(problems_ref_file, data.values())
        results = evaluate_functional_correctness(
            samples_file,
            k=list(k),
            n_workers=n_workers,
            timeout=timeout,
            problem_file=problems_ref_file,
        )
        avg_steps = sum(entry["avg_steps"] for entry in all_steps) / len(all_steps) if all_steps else 0.0
        all_results = {
            "summary": {
                "accuracy": round(results["pass@1"] * 100, 2),
                "average_steps": avg_steps,
                "dataset": "evalplus/humanevalplus",
            },
            "results": all_steps,
        }
        save_path = os.path.join(save_dir, "all_results.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print("[HumanEval+]")
        print("Accuracy:", round(results["pass@1"] * 100, 2))
        print(f"Average steps: {avg_steps}")
        print(f"Results saved to {save_path}")


def test_mbppplus(
    model, tokenizer, accelerator, save_dir,
    gen_length, steps, block_length,
    alg="default",
    temperature=0.0,
    test_size=None, random_sampling=False,
    num_samples=1,
    save_steps=False,
    backbone="llada",
):
    save_dir = _run_dir(save_dir, "mbppplus", alg, gen_length, block_length, steps, num_samples, backbone)
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    raw = load_dataset("evalplus/mbppplus", split="test")
    if test_size:
        random.seed(516)
        rows = random.sample(list(raw), test_size) if random_sampling else list(raw.select(range(test_size)))
    else:
        rows = list(raw)

    indexed_rows = list(enumerate(rows))
    local_steps = []

    with accelerator.split_between_processes(indexed_rows) as local_data:
        for idx, ex in tqdm(
            local_data, desc="Generating completions for MBPP+", disable=not accelerator.is_main_process,
        ):
            task_id = ex["task_id"]
            tests = mbppplus_tests_for_prompt(ex)
            unit_test = mbppplus_unit_test_for_scoring(ex)
            m = kodcode_v3_messages(ex["prompt"], tests)
            prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids_original = torch.tensor(tokenizer(prompt_text)["input_ids"]).to(accelerator.device).unsqueeze(0)
            task_samples = []
            task_steps = []
            task_passed = False
            for sample_idx in range(num_samples):
                x_output, used_steps = decode_response(
                    model, tokenizer, input_ids_original, gen_length, steps, block_length,
                    temperature=temperature,
                    step_save_dir=step_save_dir, example_idx=f"q{idx}_s{sample_idx}",
                    alg=alg, backbone=backbone,
                )
                decoded_text = tokenizer.batch_decode(
                    x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True
                )[0].split(tokenizer.eos_token)[0]
                passed, extracted_program, test_code = score_mbpp_d1(
                    decoded_text, unit_test, output_dir=save_dir
                )
                if passed:
                    task_passed = True
                task_samples.append({
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "used_steps": used_steps,
                    "generation": decoded_text,
                    "extracted_program": extracted_program,
                    "test_code": test_code,
                    "passed": passed,
                })
                task_steps.append(used_steps)
            local_steps.append({
                "task_id": task_id,
                "input_prompt": prompt_text,
                "solution_code": ex.get("code") or "",
                "any_passed": task_passed,
                "avg_steps": sum(task_steps) / len(task_steps),
                "samples": task_samples,
            })

    accelerator.wait_for_everyone()
    all_steps = gather_object(local_steps)
    if accelerator.is_main_process:
        all_steps = sorted(all_steps, key=lambda x: x["task_id"])
        total_passed_tasks = sum(1 for task_data in all_steps if task_data["any_passed"])
        accuracy = total_passed_tasks / len(all_steps) if all_steps else 0
        average_steps = sum(entry["avg_steps"] for entry in all_steps) / len(all_steps) if all_steps else 0.0
        all_results = {
            "summary": {
                "accuracy": round(accuracy * 100, 2),
                "average_steps": average_steps,
                "total_tasks": len(all_steps),
                "passed_tasks": total_passed_tasks,
                "num_samples_per_task": num_samples,
                "dataset": "evalplus/mbppplus",
            },
            "results": all_steps,
        }
        samples_file = os.path.join(save_dir, "all_results.json")
        with open(samples_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print("[MBPP+]")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average steps: {average_steps}")
        print(f"Results saved to {samples_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["humanevalplus", "mbppplus"])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--random_sampling", action="store_true")
    parser.add_argument("--alg", type=str, default="default", choices=["default", "random", "topk_margin", "entropy", "maskgit_plus"])
    parser.add_argument("--backbone", type=str, default="llada", choices=["llada", "dream"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_steps", action="store_true")
    parser.add_argument("--humaneval_k", type=str, default="1")
    parser.add_argument("--humaneval_workers", type=int, default=4)
    parser.add_argument("--humaneval_timeout", type=float, default=3.0)
    parser.add_argument("--code_prompt_style", type=str, default="kodcode_v3")
    args = parser.parse_args()
    if args.code_prompt_style != "kodcode_v3":
        raise SystemExit("Code evaluation uses the kodcode_v3 prompt.")

    accelerator = Accelerator()
    device = accelerator.device
    model, tokenizer = load_model_for_eval(args.model_path, device)

    common = dict(
        model=model, tokenizer=tokenizer, accelerator=accelerator,
        save_dir=args.save_dir,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        alg=args.alg,
        test_size=args.test_size,
        random_sampling=args.random_sampling,
        num_samples=args.num_samples,
        save_steps=args.save_steps,
        backbone=args.backbone,
    )
    if args.dataset == "humanevalplus":
        test_humaneval_plus(
            **common,
            k=tuple(map(int, args.humaneval_k.split(","))),
            n_workers=args.humaneval_workers,
            timeout=args.humaneval_timeout,
        )
    else:
        test_mbppplus(**common)


if __name__ == "__main__":
    main()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
