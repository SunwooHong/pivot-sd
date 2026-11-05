import os
import json
import torch
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object

from pivot_sd.answers import (  # noqa: F401
    compare_answers,
    extract_math_answer,
    parse_answer,
    parse_ground_truth_answer,
    process_file,
)
from pivot_sd.decode import decode_response, run_dir
from pivot_sd.lora import load_model_for_eval


def test_dataset(
    model, tokenizer, accelerator, save_dir, dataset,
    gen_length, steps, block_length,
    alg="default",
    temperature=0.0,
    test_size=None,
    slice_start=None, slice_end=None,
    num_samples=1,
    save_steps=False,
    data_path=None,
    backbone="llada",
):
    family = "Dream" if backbone == "dream" else "LLaDA"
    save_dir = run_dir(
        save_dir, family, dataset, alg, gen_length, block_length, steps, num_samples
    )
    
    # All ranks need output dirs: non-main processes write stepwise JSON during decode.
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = None
    if save_steps:
        step_save_dir = os.path.join(save_dir, "stepwise")
        os.makedirs(step_save_dir, exist_ok=True)

    if data_path is None:
        data_path = f"./data/{dataset}_test.json"
    data = process_file(data_path)

    if slice_start is not None or slice_end is not None:
        start = 0 if slice_start is None else slice_start
        data = data[start:slice_end]
    elif test_size:
        data = data[:test_size]
    print("data_path:", data_path)
    print("data length: ", len(data))
    # Pre-calculate global indices so task_id remains correct when split
    task_id_offset = 0 if slice_start is None else slice_start
    indexed_data = [(task_id_offset + i, example) for i, example in enumerate(data)]
    local_results = []

    # 1. SHARD THE DATASET
    with accelerator.split_between_processes(indexed_data) as local_data:
        for i, example in tqdm(local_data, desc=f"Generating {dataset.capitalize()}", disable=not accelerator.is_main_process):
            if dataset == "gsm8k":
                prompt = example['question']
                answer = example['answer']
                ground_truth_answer = parse_ground_truth_answer(answer)
            elif dataset == "math":
                prompt = example['problem']
                solution = example['solution']
                ground_truth_answer = extract_math_answer(prompt, solution)

            example_samples = []
            example_correct = False
            example_steps = []

            m = [
                {"role": "system", "content": "Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format 'The final answer is'."},
                {"role": "user", "content": prompt}
            ]
            prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            
            # 2. MAP TO CORRECT DEVICE
            input_ids_original = torch.tensor(tokenizer(prompt_text)['input_ids']).to(accelerator.device).unsqueeze(0)
            
            for sample_idx in range(num_samples):
                x_output, used_steps = decode_response(
                    model, tokenizer, input_ids_original, gen_length, steps, block_length,
                    temperature=temperature,
                    step_save_dir=step_save_dir, example_idx=f"q{i}_s{sample_idx}",
                    alg=alg, backbone=backbone,
                )

                generated_text = tokenizer.batch_decode(x_output[:, input_ids_original.shape[1]:], skip_special_tokens=True)[0]
                
                if dataset == "gsm8k":
                    generated_answer = parse_answer(generated_text)
                    is_correct = generated_answer == ground_truth_answer
                elif dataset == "math":
                    generated_answer = extract_math_answer(prompt_text, generated_text)
                    is_correct = compare_answers(prompt_text, ground_truth_answer, generated_answer)

                if is_correct:
                    example_correct = True

                example_steps.append(used_steps)
                example_samples.append({
                    "task_id": i,
                    "sample_idx": sample_idx,
                    "used_steps": used_steps,
                    "generation": generated_text,
                    "parsed_answer": generated_answer,
                    "is_correct": is_correct
                })

            local_results.append({
                "task_id": i,
                "input_prompt": prompt_text,
                "ground_truth_answer": ground_truth_answer,
                "any_correct": example_correct,
                "avg_steps": round(sum(example_steps) / len(example_steps), 2),
                "samples": example_samples
            })

    # 3. SYNC AND GATHER
    accelerator.wait_for_everyone()
    all_results_list = gather_object(local_results)

    # 4. PROCESS METRICS ONLY ON MAIN GPU
    if accelerator.is_main_process:
        # Deduplicate in case accelerate padded the split arrays
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
                "accuracy": round(accuracy*100, 2),
                "average_steps": round(avg_steps, 2),
                "total_questions": len(final_results_list),
                "correct_questions": correct_count,
                "num_samples_per_question": num_samples
            },
            "results": final_results_list
        }
        
        save_path = f"{save_dir}/all_results.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"[{dataset.capitalize()}]")
        print(f"Accuracy: {round(accuracy*100, 2)}")
        print(f"Average steps: {round(avg_steps, 2)}")
        print(f"Results saved to {save_path}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'math'], help='Dataset to use')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--gen_length', type=int, default=256, help='Generation length')
    parser.add_argument('--block_length', type=int, default=64, help='Block length')
    parser.add_argument('--steps', type=int, default=256, help='Number of steps')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--test_size', type=int, default=None, help='Size of the test set')
    parser.add_argument('--slice_start', type=int, default=None, help='Start index for dataset slice (inclusive)')
    parser.add_argument('--slice_end', type=int, default=None, help='End index for dataset slice (exclusive)')
    parser.add_argument('--alg', type=str, default='default', choices=['default', 'random', 'topk_margin', 'entropy', 'maskgit_plus'])
    parser.add_argument('--backbone', type=str, default='llada', choices=['llada', 'dream'])
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per question/task')
    parser.add_argument('--save_steps', action='store_true', help='Save the results of each step')
    parser.add_argument('--data_path', type=str, default=None, help='Test JSON/JSONL. Default: ./data/{dataset}_test.json')
    args = parser.parse_args()

    print("Parsed arguments:", args)

    accelerator = Accelerator()
    device = accelerator.device

    model, tokenizer = load_model_for_eval(args.model_path, device)

    if args.dataset in ['gsm8k', 'math']:
        test_dataset(
            model, tokenizer, accelerator,
            save_dir=args.save_dir,
            dataset=args.dataset, 
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
            data_path=args.data_path,
            backbone=args.backbone,
        )

if __name__ == '__main__':
    main()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
