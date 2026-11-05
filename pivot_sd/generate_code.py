"""Sample AceCode trajectories with the kodcode_v3 prompt and tests in the prompt."""
import os
import json
import torch
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object

from pivot_sd.answers import evaluate_task, process_file
from pivot_sd.decode import decode_response, run_dir

KODCODE_SYSTEM_PROMPT_V2 = (
    "You are an expert Python programmer. You must write complete, valid Python code "
    "to solve the user's problem."
)
ACECODE_PROMPT_NUM_TESTS = 3
V3_FOOTER = (
    "You must output ONLY raw Python code.\n"
    "Do not include any conversational text or explanations before or after the code."
)


def build_acecode_chat_messages(prompt_body: str, test_cases: list | None = None) -> list:
    parts = [prompt_body]
    if test_cases:
        prompt_tests = list(test_cases)[:ACECODE_PROMPT_NUM_TESTS]
        tests_str = "\n".join(str(t) for t in prompt_tests)
        parts.append(f"Your code should pass these tests:\n\n{tests_str}")
    parts.append(V3_FOOTER)
    return [
        {"role": "system", "content": KODCODE_SYSTEM_PROMPT_V2},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _extract_code_only(text):
    import re
    code_match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, re.DOTALL)
    return code_match.group(1).strip() if code_match else text.strip()


def _stepwise_trace_path(step_save_dir: str, example_idx: str) -> str:
    return os.path.join(step_save_dir, f"all_steps_{example_idx}.json")


def _stepwise_sample_exists(step_save_dir: str | None, q_idx: int, sample_idx: int) -> bool:
    if step_save_dir is None:
        return False
    return os.path.isfile(_stepwise_trace_path(step_save_dir, f"q{q_idx}_s{sample_idx}"))


def _stepwise_question_complete(step_save_dir: str | None, q_idx: int, num_samples: int) -> bool:
    if step_save_dir is None:
        return False
    return all(_stepwise_sample_exists(step_save_dir, q_idx, s) for s in range(num_samples))


def _load_stepwise_sample(step_save_dir: str, q_idx: int, sample_idx: int) -> tuple[str, int]:
    path = _stepwise_trace_path(step_save_dir, f"q{q_idx}_s{sample_idx}")
    with open(path, encoding="utf-8") as f:
        all_steps = json.load(f)
    if not all_steps:
        raise ValueError(f"Empty stepwise trace: {path}")
    last = all_steps[-1]
    generated_text = last.get("decoded_text") or ""
    used_steps = int(last.get("step", len(all_steps)))
    return generated_text, used_steps


def test_dataset(
    model, tokenizer, accelerator, save_dir,
    gen_length, steps, block_length,
    alg="default",
    temperature=0.0,
    test_size=None,
    slice_start=None, slice_end=None,
    num_samples=1,
    save_steps=False,
    resume=False,
    code_eval_timeout=3.0,
    data_path=None,
    backbone="llada",
):
    family = "Dream" if backbone == "dream" else "LLaDA"
    save_dir = run_dir(
        save_dir, family, "acecode/with_tests", alg, gen_length, block_length, steps, num_samples
    )
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    if data_path is None:
        raise ValueError("data_path is required (JSONL of prompts).")
    data = process_file(data_path)

    if slice_start is not None or slice_end is not None:
        start = 0 if slice_start is None else slice_start
        data = data[start:slice_end]
    elif test_size:
        data = data[:test_size]
    print("data_path:", data_path)
    print("data length: ", len(data))
    if resume and step_save_dir:
        done = sum(1 for i in range(len(data)) if _stepwise_question_complete(step_save_dir, i, num_samples))
        print(f"[resume] {done}/{len(data)} questions already have all {num_samples} stepwise traces in {step_save_dir}")

    task_id_offset = 0 if slice_start is None else slice_start
    indexed_data = [(task_id_offset + i, example) for i, example in enumerate(data)]
    local_results = []
    resumed_samples = 0

    with accelerator.split_between_processes(indexed_data) as local_data:
        for i, example in tqdm(local_data, desc="Generating Acecode", disable=not accelerator.is_main_process):
            prompt = example["question"]
            test_cases = example["test_cases"]
            m = build_acecode_chat_messages(prompt, test_cases)
            prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

            need_decode = not (
                resume and step_save_dir and _stepwise_question_complete(step_save_dir, i, num_samples)
            )
            input_ids_original = None
            if need_decode:
                input_ids_original = torch.tensor(tokenizer(prompt_text)["input_ids"]).to(accelerator.device).unsqueeze(0)

            example_samples = []
            example_correct = False
            example_steps = []
            for sample_idx in range(num_samples):
                if resume and step_save_dir and _stepwise_sample_exists(step_save_dir, i, sample_idx):
                    generated_text, used_steps = _load_stepwise_sample(step_save_dir, i, sample_idx)
                    resumed_samples += 1
                else:
                    x_output, used_steps = decode_response(
                        model, tokenizer, input_ids_original, gen_length, steps, block_length,
                        temperature=temperature,
                        step_save_dir=step_save_dir, example_idx=f"q{i}_s{sample_idx}",
                        alg=alg, backbone=backbone,
                    )
                    generated_text = tokenizer.batch_decode(
                        x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True
                    )[0]

                generated_answer = _extract_code_only(generated_text)
                is_correct = evaluate_task(
                    {"generation": generated_text},
                    test_cases,
                    timeout=code_eval_timeout,
                )
                if is_correct:
                    example_correct = True
                example_steps.append(used_steps)
                example_samples.append({
                    "task_id": i,
                    "sample_idx": sample_idx,
                    "used_steps": used_steps,
                    "generation": generated_text,
                    "parsed_answer": generated_answer,
                    "is_correct": is_correct,
                })

            local_results.append({
                "task_id": i,
                "input_prompt": prompt_text,
                "ground_truth_answer": "pass_tests",
                "any_correct": example_correct,
                "avg_steps": round(sum(example_steps) / len(example_steps), 2),
                "samples": example_samples,
            })

    if resume and accelerator.is_main_process:
        print(f"[resume] loaded {resumed_samples} samples from existing stepwise traces on this rank")

    accelerator.wait_for_everyone()
    all_results_list = gather_object(local_results)

    if accelerator.is_main_process:
        unique_results = {r["task_id"]: r for r in all_results_list}
        final_results_list = list(unique_results.values())
        final_results_list.sort(key=lambda x: x["task_id"])
        correct_count = sum(1 for r in final_results_list if r["any_correct"])
        total_steps = sum(sum(s["used_steps"] for s in r["samples"]) for r in final_results_list)
        total_samples = len(final_results_list) * num_samples
        accuracy = correct_count / len(final_results_list) if final_results_list else 0
        avg_steps = total_steps / total_samples if total_samples else 0
        results = {
            "summary": {
                "accuracy": round(accuracy * 100, 2),
                "average_steps": round(avg_steps, 2),
                "total_questions": len(final_results_list),
                "correct_questions": correct_count,
                "num_samples_per_question": num_samples,
                "acecode_include_tests_in_prompt": True,
            },
            "results": final_results_list,
        }
        save_path = f"{save_dir}/all_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print("[Acecode]")
        print(f"Accuracy: {round(accuracy * 100, 2)}")
        print(f"Average steps: {round(avg_steps, 2)}")
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["acecode"])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--slice_start", type=int, default=None)
    parser.add_argument("--slice_end", type=int, default=None)
    parser.add_argument("--alg", type=str, default="default", choices=["default", "random", "topk_margin", "entropy", "maskgit_plus"])
    parser.add_argument("--backbone", type=str, default="llada", choices=["llada", "dream"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_steps", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code_eval_timeout", type=float, default=3.0)
    parser.add_argument("--code_prompt_style", type=str, default="kodcode_v3")
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    if args.code_prompt_style != "kodcode_v3":
        raise SystemExit("AceCode generation uses kodcode_v3 with tests in the prompt.")
    if args.resume and not args.save_steps:
        parser.error("--resume requires --save_steps")

    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    test_dataset(
        model, tokenizer, accelerator,
        save_dir=args.save_dir,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
        alg=args.alg,
        test_size=args.test_size,
        slice_start=args.slice_start,
        slice_end=args.slice_end,
        num_samples=args.num_samples,
        save_steps=args.save_steps,
        resume=args.resume,
        code_eval_timeout=args.code_eval_timeout,
        data_path=args.data_path,
        backbone=args.backbone,
    )


if __name__ == "__main__":
    main()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
