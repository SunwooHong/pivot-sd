"""Build the pivot JSONL used for sample-200 training.

Main method (``ig_density``): for each trajectory, rank decode steps by
``info_gain_density`` and keep ``--top-k`` steps (default 10). The token
committed on a kept step is a pivot. Correct trajectories get a positive
advantage, incorrect ones a negative advantage. Block-boundary steps and
EOS / end-of-turn tokens are dropped before ranking.

``random_step`` keeps the same number of steps uniformly at random.
``entropy`` keeps the ``--top-k`` committed tokens with the highest predictive
entropy (token ranking, block boundaries kept).

Advantages are then min-max scaled with the 1st and 99th percentiles. Reported
training runs pass ``--no-advantage-in-loss``, so this scale does not change
the loss magnitude; the sign still chooses cross-entropy versus unlikelihood.
"""
from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pivot_sd.scoring import (
    extract_step_pivots,
    inject_info_gain_density,
    scale_advantages,
    topk_by_token_entropy,
    trajectory_max_step,
)

_STEPWISE_RE = re.compile(r"all_steps_q(\d+)_s(\d+)\.json$")
SYSTEM_PROMPT = (
    "Your task is to answer the question below. Give step by step reasoning before you answer, "
    "and when you're ready to answer, please use the format 'The final answer is'."
)


def parse_stepwise_filename(path: Path) -> Optional[Tuple[int, int]]:
    m = _STEPWISE_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_jsonl_or_json(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("train"), list):
        return payload["train"]
    raise ValueError(f"unsupported train file: {path}")


def load_stepwise(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"stepwise file is not a list: {path}")
    return payload


def last_decoded_text(steps: Sequence[dict]) -> str:
    if not steps:
        return ""
    last = max(steps, key=lambda s: int(s.get("step", -1)))
    return str(last.get("decoded_text") or "")


def load_correctness(path: Path) -> Dict[Tuple[int, int], bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError(f"{path} has no results list")
    out: Dict[Tuple[int, int], bool] = {}
    for row in results:
        if not isinstance(row, dict) or "task_id" not in row:
            continue
        samples = row.get("samples")
        if isinstance(samples, list) and samples:
            for sample in samples:
                if not isinstance(sample, dict) or "is_correct" not in sample:
                    continue
                sid = int(sample.get("sample_idx", 0))
                out[(int(row["task_id"]), sid)] = bool(sample["is_correct"])
        elif "is_correct" in row:
            out[(int(row["task_id"]), 0)] = bool(row["is_correct"])
    if not out:
        raise ValueError(f"no is_correct entries in {path}")
    return out


def normalize_row(raw: dict, dataset: str) -> Optional[dict]:
    if dataset == "math":
        problem = raw.get("problem")
        solution = raw.get("solution")
        if not problem or not solution:
            return None
        return {"problem": str(problem).strip(), "solution": str(solution).strip()}
    if dataset == "gsm8k":
        question = raw.get("question")
        answer = raw.get("answer")
        if not question or not answer:
            return None
        return {"problem": str(question).strip(), "solution": str(answer).strip()}
    if dataset == "acecode":
        question = raw.get("question")
        if not question:
            return None
        return {"problem": str(question).strip(), "solution": ""}
    raise ValueError(dataset)


def trajectory_correct(
    steps: Sequence[dict],
    item: dict,
    prompt_text: str,
    dataset: str,
) -> bool:
    from pivot_sd.answers import (
        compare_answers,
        extract_math_answer,
        parse_answer,
        parse_ground_truth_answer,
    )

    gen_text = last_decoded_text(steps)
    if dataset == "gsm8k":
        return parse_answer(gen_text) == parse_ground_truth_answer(item["solution"])
    gt = extract_math_answer(prompt_text, item["solution"])
    pred = extract_math_answer(prompt_text, gen_text)
    return compare_answers(prompt_text, gt, pred)


def exclude_token_ids(tokenizer) -> List[int]:
    from pivot_sd.ig_pivot_utils import supervision_stop_token_id

    ids: List[int] = []
    stop_id = supervision_stop_token_id(tokenizer)
    if stop_id is not None:
        ids.append(int(stop_id))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and int(eos_id) not in ids:
        ids.append(int(eos_id))
    return ids


def random_step_seed(path: Path, qid: int, sample_seed: int, base_seed: int) -> int:
    import os

    slot = zlib.adler32(os.path.abspath(path).encode("utf-8", errors="replace")) & 0xFFFFFFFF
    return int(base_seed) + int(slot) + int(qid) * 1_000_003 + int(sample_seed) * 97_621_393


def rows_for_trajectory(
    steps: List[dict],
    *,
    selector: str,
    top_k: int,
    exclude_ids: Optional[Sequence[int]],
    is_correct: bool,
    path: Path,
    orig_idx: int,
    sample_seed: int,
    gen_length: int,
    random_seed: int,
    block_size: int = 64,
) -> List[dict]:
    if selector == "entropy":
        chosen = topk_by_token_entropy(steps, k=top_k, exclude_token_ids=exclude_ids)
        mag_key = "token_entropy"
    else:
        inject_info_gain_density(steps, block_size)
        step_pick = "random" if selector == "random_step" else "top"
        seed = random_step_seed(path, orig_idx, sample_seed, random_seed) if step_pick == "random" else 0
        chosen = extract_step_pivots(
            steps,
            block_size=block_size,
            keep_top_k=top_k,
            exclude_token_ids=exclude_ids,
            step_pick=step_pick,
            random_seed=seed,
        )
        mag_key = "advantage_mag"
    tmax = trajectory_max_step(steps)
    text = last_decoded_text(steps)
    rows: List[dict] = []
    for pv in chosen:
        mag = float(pv[mag_key] if mag_key in pv else pv["advantage_mag"])
        adv = mag if is_correct else -mag
        row = {
            "stepwise_path": str(path.resolve()),
            "orig_idx": int(orig_idx),
            "seed": int(sample_seed),
            "is_correct": bool(is_correct),
            "pivot_step": int(pv["pivot_step"]),
            "position": int(pv["position"]),
            "target_token_id": int(pv["token_id"]),
            "advantage": adv,
            "gen_length": int(gen_length),
            "max_step_in_file": tmax,
            "stepwise_text_len": len(text),
        }
        if selector == "entropy":
            row["token_entropy"] = mag
        else:
            row["info_gain_density"] = mag
        rows.append(row)
    return rows


def build(
    *,
    stepwise_dir: Path,
    train_json: Path,
    out_jsonl: Path,
    dataset: str,
    selector: str,
    tokenizer_name: str,
    top_k: int = 10,
    qid_offset: int = 0,
    all_results_json: Optional[Path] = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
    random_seed: int = 42,
    gen_length: int = 256,
    block_size: int = 64,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    if dataset == "acecode" and all_results_json is None:
        raise ValueError("acecode requires --all-results-json (per-sample is_correct)")
    train_rows = load_jsonl_or_json(train_json)
    precomputed = load_correctness(all_results_json) if all_results_json else None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    exclude_ids = exclude_token_ids(tokenizer)

    paths = sorted(stepwise_dir.glob("all_steps_q*_s*.json"))
    raw_rows: List[dict] = []
    n_skip = 0
    for path in paths:
        parsed = parse_stepwise_filename(path)
        if parsed is None:
            continue
        file_qid, sample_seed = parsed
        orig_idx = file_qid + qid_offset
        if orig_idx < 0 or orig_idx >= len(train_rows):
            n_skip += 1
            continue
        item = normalize_row(train_rows[orig_idx], dataset)
        if item is None:
            n_skip += 1
            continue
        steps = load_stepwise(path)
        if not steps or not last_decoded_text(steps):
            n_skip += 1
            continue
        if precomputed is not None:
            key = (file_qid, sample_seed)
            if key not in precomputed:
                n_skip += 1
                continue
            ok = precomputed[key]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ok = trajectory_correct(steps, item, prompt_text, dataset)
        raw_rows.extend(
            rows_for_trajectory(
                steps,
                selector=selector,
                top_k=top_k,
                exclude_ids=exclude_ids,
                is_correct=ok,
                path=path,
                orig_idx=orig_idx,
                sample_seed=sample_seed,
                gen_length=gen_length,
                random_seed=random_seed,
                block_size=block_size,
            )
        )
    if not raw_rows:
        raise SystemExit(f"no pivot rows written from {stepwise_dir} (skipped {n_skip})")
    scaled, meta = scale_advantages(raw_rows, p_low=p_low, p_high=p_high)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in scaled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": dataset,
        "selector": selector,
        "top_k": top_k,
        "n_rows": len(scaled),
        "n_stepwise_skipped": n_skip,
        "output_pivot_jsonl": str(out_jsonl),
        **meta,
    }
    meta_path = out_jsonl.with_name(out_jsonl.stem + "_meta.json")
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(scaled)} rows -> {out_jsonl}")
    print(f"Meta -> {meta_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stepwise-dir", type=Path, required=True)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--dataset", choices=("math", "gsm8k", "acecode"), required=True)
    p.add_argument(
        "--selector",
        choices=("ig_density", "random_step", "entropy"),
        default="ig_density",
        help="ig_density is Pivot-SD. random_step and entropy are ablations.",
    )
    p.add_argument("--tokenizer", type=str, required=True, help="LLaDA tokenizer or model directory")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--qid-offset", type=int, default=0)
    p.add_argument("--all-results-json", type=Path, default=None)
    p.add_argument("--p-low", type=float, default=1.0)
    p.add_argument("--p-high", type=float, default=99.0)
    p.add_argument("--random-seed", type=int, default=42, help="Base seed for --selector random_step")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-size", type=int, default=64, help="0 disables the block-boundary filter (Dream)")
    args = p.parse_args()
    build(
        stepwise_dir=args.stepwise_dir,
        train_json=args.train_json,
        out_jsonl=args.out_jsonl,
        dataset=args.dataset,
        selector=args.selector,
        tokenizer_name=args.tokenizer,
        top_k=args.top_k,
        qid_offset=args.qid_offset,
        all_results_json=args.all_results_json,
        p_low=args.p_low,
        p_high=args.p_high,
        random_seed=args.random_seed,
        gen_length=args.gen_length,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
