"""
Utilities for IG-sanitized pivot extraction from stepwise JSON and M_t reconstruction.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

_STEPWISE_RE = re.compile(r"all_steps_q(\d+)_s(\d+)\.json$")


def parse_stepwise_filename(path: str) -> Optional[Tuple[int, int]]:
    m = _STEPWISE_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_block_boundary_step(step_num: int, block_size: int, max_step: int) -> bool:
    if block_size <= 0:
        return False
    return step_num > 0 and step_num % block_size == 0 and step_num < max_step


def stabilize_ig(
    cu: Optional[float],
    nu: Optional[float],
    ig_json: Optional[float],
    n_masked: int,
    u_floor: float,
    clip_lo: float,
    clip_hi: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Return stabilized IG or (None, drop reason)."""
    if cu is None or ig_json is None:
        return None, "missing_info"
    cu = float(cu)
    if cu < u_floor:
        return None, "below_u_floor"
    ig = float(ig_json)
    ig = max(clip_lo, min(clip_hi, ig))
    # optional per-mask scaling for ranking (not applied to stored advantage by default)
    return ig, None


def count_masked_positions(step_obj: Dict[str, Any]) -> int:
    return len(step_obj.get("all_tokens") or [])


def file_generation_start_index(steps: List[Dict[str, Any]]) -> Optional[int]:
    """
    Absolute index in the stepwise decode canvas where **generation-relative** position 0 starts.

    KLASS stepwise logs include, on the **earliest** ``step``, an ``all_tokens`` list (typically
    sorted by ``position``). The **smallest** ``position`` in that list is the first slot of the
    tracked generation span for that file (often the first masked block of the solution).

    Training alignment: indices into the **solution** span of ``clean_input_ids`` (which start at
    ``prompt_len``) satisfy ``clean_index = file_position + (prompt_len - start)``, i.e. pass
    ``position_offset = prompt_len - start`` to :func:`cumulative_unmask_map_before` and the same
    offset for pivot ``position`` values stored in the same file axis.

    Falls back to the earliest step's ``decoded_tokens`` if ``all_tokens`` is empty.
    """
    if not steps:
        return None
    first = min(steps, key=lambda s: int(s["step"]))
    toks = list(first.get("all_tokens") or ())
    if not toks:
        toks = list(first.get("decoded_tokens") or ())
    if not toks:
        return None
    return min(int(t["position"]) for t in toks)


def stepwise_position_offset_for_file_gen_start(
    prompt_len: int, file_gen_start: Optional[int]
) -> int:
    """Return ``prompt_len - file_gen_start`` when ``file_gen_start`` is known, else ``0``."""
    if file_gen_start is None:
        return 0
    return int(prompt_len) - int(file_gen_start)


def minmax_decoded_token_positions(steps: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Return (min, max) of ``position`` over all ``decoded_tokens`` in ``steps``, or None if empty."""
    mn, mx = 10**18, -1
    for st in steps:
        for dt in st.get("decoded_tokens") or []:
            p = int(dt["position"])
            mn = min(mn, p)
            mx = max(mx, p)
    if mx < 0:
        return None
    return mn, mx


def inferred_stepwise_position_offset(
    steps: List[Dict[str, Any]], prompt_len: int, gen_length: int
) -> int:
    """
    Heuristic: KLASS ``stable_confident_decode`` stepwise JSON uses **absolute** indices into the
    full decode tensor ``x`` (frozen prompt + gen), so the smallest ``position`` is usually
    ``>= prompt_len`` when ``prompt_len`` matches the decode run.

    Some other pipelines store **generation-only** indices in ``[0, gen_length)``. In that case
    ``min(position)`` is often small and ``max(position) < gen_length``; add ``prompt_len`` to map
    into the same absolute layout as ``clean_input_ids``.

    If ambiguous, returns 0 (treat as absolute).
    """
    mm = minmax_decoded_token_positions(steps)
    if mm is None:
        return 0
    mn, mx = mm
    pl = int(prompt_len)
    gl = int(gen_length)
    if mn >= pl:
        return 0
    if mn <= 8 and mx < gl:
        return pl
    return 0


def cumulative_unmask_map_before(
    steps: List[Dict[str, Any]],
    before_step: int,
    *,
    position_offset: int = 0,
) -> Dict[int, int]:
    """
    Positions revealed strictly before decode step ``before_step`` (1-based ``step`` field).

    Each ``decoded_tokens`` entry is ``{"position": p, "token_id": t, ...}``. Keys stored in the
    returned map are ``p + position_offset``.

    Typical KLASS alignment: take ``start = file_generation_start_index(steps)`` from the first
    step's ``all_tokens``, then ``position_offset = prompt_len - start`` so
    ``p + offset == prompt_len + (p - start)`` indexes the solution span in ``clean_input_ids``.

    Legacy **absolute** file positions (file gen start equals training ``prompt_len``): use
    ``position_offset=0``. Pure **0-based gen-only** positions: use ``position_offset=prompt_len``.
    """
    steps = sorted(steps, key=lambda s: int(s["step"]))
    m: Dict[int, int] = {}
    off = int(position_offset)
    for st in steps:
        s = int(st["step"])
        if s >= before_step:
            break
        for dt in st.get("decoded_tokens") or []:
            abs_pos = int(dt["position"]) + off
            m[abs_pos] = int(dt["token_id"])
    return m


def supervision_stop_token_id(tokenizer) -> Optional[int]:
    """
    Token id used as the **end of the supervised assistant turn** in chat models.

    Prefer ``<|eot_id|>`` when present in the vocabulary (Llama-style chat). Many LLaDA / GLM-style
    checkpoints set ``tokenizer.eos_token_id`` to ``<|endoftext|>`` (e.g. 126081), which can appear
    **before** the real turn end in the prompt; supervision should stop at ``<|eot_id|>`` (e.g.
    126348), not at the first ``<|endoftext|>`` in the generation span.

    If ``<|eot_id|>`` is absent, falls back to ``tokenizer.eos_token_id``.
    """
    if tokenizer is None:
        return None
    enc = getattr(tokenizer, "added_tokens_encoder", None) or {}
    if isinstance(enc, dict) and "<|eot_id|>" in enc:
        return int(enc["<|eot_id|>"])
    try:
        tid = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        tid = None
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is not None and tid >= 0 and (unk is None or int(tid) != int(unk)):
        return int(tid)
    return getattr(tokenizer, "eos_token_id", None)


def supervision_exclusive_end_1d(
    clean_input_ids: torch.Tensor,
    prompt_len: int,
    stop_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> int:
    """
    Same rule as ``IGPivotTrainer`` / ``_supervision_exclusive_end``: exclusive end index for the
    supervised span ``[prompt_len, exclusive_end)``. Stops one past the **first** ``stop_token_id``
    in the generation span (use :func:`supervision_stop_token_id` — usually ``<|eot_id|>``), else one
    past the last non-pad token in gen.
    """
    clean_input_ids = clean_input_ids.view(-1)
    L = int(clean_input_ids.numel())
    pl = max(0, min(int(prompt_len), L))
    device = clean_input_ids.device
    idx = torch.arange(L, device=device, dtype=torch.long)

    in_gen = idx >= pl
    if pad_token_id is not None:
        non_pad = clean_input_ids != int(pad_token_id)
    else:
        non_pad = torch.ones(L, device=device, dtype=torch.bool)

    valid_for_last = in_gen & non_pad
    if not valid_for_last.any():
        return pl

    last_idx = idx[valid_for_last].max().item()
    end_no_eos = int(last_idx) + 1

    if stop_token_id is None:
        return min(end_no_eos, L)

    is_stop = (clean_input_ids == int(stop_token_id)) & in_gen
    if not is_stop.any():
        return min(end_no_eos, L)

    first_stop = int(idx[is_stop].min().item())
    return min(first_stop + 1, L)


def count_unmask_map_clean_mismatches(
    unmask_map: Dict[int, int],
    clean_input_ids: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    gen_end_exclusive: Optional[int] = None,
) -> Tuple[int, int, Optional[int]]:
    """
    Count positions p in the generation window where unmask_map[p] != clean[p].

    Window is ``[prompt_len, gen_end)`` with ``gen_end = min(prompt_len+gen_length, L)`` unless
    ``gen_end_exclusive`` is set (e.g. from :func:`supervision_exclusive_end_1d`).

    Stepwise logs store the token on the decode canvas at reveal time; before convergence that
    can differ from the **final** supervised token at the same index. Large disagreement rates
    across many pivots often mean **absolute position** in the stepwise tensor does not match
    this ``clean_input_ids`` layout (different prompt length, max_length, or truncation vs decode).
    """
    L = int(clean_input_ids.numel())
    if gen_end_exclusive is not None:
        gen_end = min(int(gen_end_exclusive), L)
    else:
        gen_end = min(int(prompt_len) + int(gen_length), L)
    bad = 0
    good = 0
    first_bad: Optional[int] = None
    for p, tid in unmask_map.items():
        pi = int(p)
        if pi < int(prompt_len) or pi >= gen_end:
            continue
        c = int(clean_input_ids[pi].item())
        if int(tid) != c:
            bad += 1
            if first_bad is None:
                first_bad = pi
        else:
            good += 1
    return good, bad, first_bad


def build_m_t_input_ids(
    clean_input_ids: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    mask_id: int,
    pad_id: int,
    unmask_map: Dict[int, int],
    *,
    stop_token_id: Optional[int] = None,
    clip_to_supervision_end: bool = True,
) -> torch.Tensor:
    """
    Reconstruct masked state ``M_t`` before IG ``pivot_step``: **indices are never permuted.**

    - ``x = clean_input_ids.clone()`` (same length everywhere).
    - Indices ``[0, prompt_len)`` and ``[gen_end, L)`` are left as in ``clean`` (prompt + tail/pad).
    - Only ``pos in [prompt_len, gen_end)`` are overwritten: ``mask_id`` if ``pos`` not in
      ``unmask_map``, else ``unmask_map[pos]``.

    **Generation window:** by default ``gen_end = min(prompt_len + gen_length, L)``. If
    ``clip_to_supervision_end`` and ``stop_token_id`` are set (from :func:`supervision_stop_token_id`,
    typically ``<|eot_id|>``, **not** ``<|endoftext|>`` when those differ), ``gen_end`` is also
    capped by :func:`supervision_exclusive_end_1d` so masking matches the CE span.

    **Why ``<|redacted_start_header_id|>`` can still appear inside** ``[prompt_len, gen_end)``: chat
    templates often place assistant header tokens **immediately after** ``add_generation_prompt``;
    those ids are part of the supervised assistant span until the stop token. They are not
    stripped unless you redefine ``prompt_len`` to start *after* those markers.

    Stepwise ``position`` alignment: see :func:`cumulative_unmask_map_before`.
    """
    x = clean_input_ids.clone()
    L = x.numel()
    pl = int(prompt_len)
    gen_cap = min(pl + int(gen_length), L)
    if clip_to_supervision_end and stop_token_id is not None:
        excl = supervision_exclusive_end_1d(x, pl, stop_token_id, pad_id)
        gen_end = min(gen_cap, excl)
    else:
        gen_end = gen_cap
    for pos in range(pl, gen_end):
        if pos in unmask_map:
            x[pos] = unmask_map[pos]
        else:
            x[pos] = mask_id
    # import pdb; pdb.set_trace()
    # do not leak future pad into gen region — keep original padding right of gen_end
    return x


def load_stepwise(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def load_train_rows(path: str) -> List[Dict[str, Any]]:
    """
    Load training examples: a JSON array, {\"train\": [...]}, or JSONL (one object per line).
    """
    with open(path, "r") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "train" in raw:
            return raw["train"]
    except json.JSONDecodeError:
        pass
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_problem_text(item: Dict[str, Any]) -> Optional[str]:
    """
    User message text only (problem or question). Used when assistant labels come from stepwise
    decode logs, not from train.json.
    """
    if not isinstance(item, dict):
        return None
    p = item.get("problem") or item.get("question")
    if p is None or not str(p).strip():
        return None
    return str(p).strip()


def normalize_math_train_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a row to canonical keys: problem, solution.

    Required for **build_ig_pivot_jsonl** (correctness vs reference) and for **--completion-source train**:
      - problem or question
      - solution (reference completion / true answer text for grading)
    """
    if not isinstance(item, dict):
        return None
    problem = item.get("problem") or item.get("question")
    sol = item.get("solution")
    if problem is None or not str(problem).strip():
        return None
    if sol is None or not str(sol).strip():
        return None
    return {"problem": str(problem).strip(), "solution": str(sol).strip()}


def last_step_decoded_text(steps: List[Dict[str, Any]]) -> str:
    """
    Full assistant completion from the stepwise log: `decoded_text` at the maximum `step`
    (e.g. final denoising step 256), not necessarily the last list element if order varies.
    """
    if not steps:
        return ""
    last = max(steps, key=lambda s: int(s["step"]))
    return (last.get("decoded_text") or "").strip()


def _align_prompt_prefix_len(full_ids: torch.Tensor, prompt_ids: torch.Tensor) -> Optional[int]:
    """Return prompt token length if ``full_ids`` starts with ``prompt_ids`` (or longest match)."""
    pl = int(prompt_ids.numel())
    fl = int(full_ids.numel())
    if fl < pl:
        return None
    if torch.equal(full_ids[:pl], prompt_ids):
        return pl
    for p in range(min(pl, fl), 0, -1):
        if torch.equal(full_ids[:p], prompt_ids[:p]):
            return p
    return None


def _truncate_sft_to_max_length(
    prompt_text: str,
    full_text: str,
    tokenizer,
    max_length: int,
    *,
    min_gen_tokens: int = 1,
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Build padded SFT ``input_ids`` (length ``max_length``) with prompt-aware truncation.

    When prompt+assistant exceeds ``max_length``, shrink the prompt first so at least
    ``min_gen_tokens`` of assistant supervision remain (fixes whole-sequence trunc dropping gen).
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.squeeze(0)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.squeeze(0)
    pl = _align_prompt_prefix_len(full_ids, prompt_ids)
    if pl is None:
        return None

    gen_ids = full_ids[pl:]
    if int(gen_ids.numel()) < min_gen_tokens:
        return None

    if int(full_ids.numel()) <= max_length:
        seq = full_ids
        out_pl = pl
    else:
        if pl >= max_length:
            out_pl = max_length - min_gen_tokens
            if out_pl <= 0:
                return None
            prompt_part = prompt_ids[:out_pl]
            gen_part = gen_ids[: min_gen_tokens]
        else:
            prompt_part = prompt_ids
            out_pl = pl
            gen_part = gen_ids[: max_length - pl]
        if int(gen_part.numel()) < min_gen_tokens:
            return None
        seq = torch.cat([prompt_part, gen_part])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    if int(seq.numel()) < max_length:
        pad = torch.full((max_length - int(seq.numel()),), pad_id, dtype=seq.dtype)
        seq = torch.cat([seq, pad])
    elif int(seq.numel()) > max_length:
        seq = seq[:max_length]
        out_pl = min(int(out_pl), max_length - min_gen_tokens)

    return seq.long(), int(out_pl)


def tokenize_math_row(item: Dict[str, Any], tokenizer, max_length: int, system_prompt: str):
    """
    Tokenize one row after normalize_math_train_row (problem + solution).
    Returns (input_ids 1d LongTensor padded max_length, prompt_len int) or None.
    """
    norm = normalize_math_train_row(item)
    if norm is None:
        return None
    trajectory = norm["solution"]

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": norm["problem"]},
    ]
    response = [{"role": "assistant", "content": trajectory}]
    inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return _truncate_sft_to_max_length(prompt_text, inputs, tokenizer, max_length)


def tokenize_math_row_with_assistant(
    question: str,
    assistant_content: str,
    tokenizer,
    max_length: int,
    system_prompt: str,
) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Same chat layout as tokenize_math_row, but assistant text is given explicitly (e.g. last-step
    decoded_text from stepwise) so token ids align with positions in the stepwise log.
    """
    if not question or not assistant_content:
        return None

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = [{"role": "assistant", "content": assistant_content}]
    inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return _truncate_sft_to_max_length(prompt_text, inputs, tokenizer, max_length)


def trajectory_max_step(steps: List[Dict[str, Any]]) -> int:
    if not steps:
        return 0
    return max(int(s["step"]) for s in steps)


def _filter_candidates_by_top_steps(
    candidates: List[Dict[str, Any]],
    *,
    keep_top_frac: float,
    keep_top_k: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Keep only candidates whose ``pivot_step`` is in the top fraction or top-k steps
    by that step's score (``ig_rank`` is identical for all tokens from the same step).
    """
    if not candidates:
        return []
    step_score: Dict[int, float] = {}
    for c in candidates:
        s = int(c["pivot_step"])
        r = float(c["ig_rank"])
        if s not in step_score or r > step_score[s]:
            step_score[s] = r
    steps_sorted = sorted(step_score.keys(), key=lambda s: (-step_score[s], s))
    if keep_top_k is not None and keep_top_k > 0:
        chosen = set(steps_sorted[: min(int(keep_top_k), len(steps_sorted))])
    else:
        scores = np.array([step_score[s] for s in steps_sorted], dtype=np.float64)
        thr = np.quantile(scores, max(0.0, 1.0 - keep_top_frac))
        chosen = {s for s in steps_sorted if step_score[s] >= thr}
    return [c for c in candidates if int(c["pivot_step"]) in chosen]


def _filter_candidates_by_random_steps(
    candidates: List[Dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Keep only candidates whose ``pivot_step`` is in a uniform random sample of ``k`` steps."""
    if not candidates or k <= 0:
        return []
    steps_unique = sorted(set(int(c["pivot_step"]) for c in candidates))
    if not steps_unique:
        return []
    kk = min(int(k), len(steps_unique))
    chosen = set(rng.sample(steps_unique, kk))
    return [c for c in candidates if int(c["pivot_step"]) in chosen]


def extract_pivot_records_from_steps(
    steps: List[Dict[str, Any]],
    *,
    ig_field: str = "info_gain",
    block_size: int,
    u_floor: float,
    clip_lo: float,
    clip_hi: float,
    keep_top_frac: float,
    keep_top_k: Optional[int] = None,
    min_advantage: float = 0.0,
    exclude_token_ids: Optional[Iterable[int]] = None,
    normalize_by_sqrt_n_masked: bool = True,
    aggregate_topk_by_step: bool = False,
    step_pick: str = "top",
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    One record per (step, decoded token) after sanitization; then global filter within
    trajectory by top keep_top_frac of stabilized IG (or top-k by ig_rank).
    By default, ig_rank = ig / sqrt(n_masked). Set normalize_by_sqrt_n_masked=False
    to rank by pure stabilized IG directly.

    If ``exclude_token_ids`` is set, decoded tokens with those ids are dropped before
    the top-frac / top-k ranking, so e.g. top-k means k non-EOS pivots when EOS is excluded.

    If ``aggregate_topk_by_step`` is True, selection applies to **decode steps**
    (one score per ``pivot_step``); every non-excluded token pivot from a kept step is returned.

    ``step_pick`` (only when ``aggregate_topk_by_step`` is True):
      - ``"top"`` (default): highest-``ig_rank`` steps via ``keep_top_k`` or ``keep_top_frac``.
      - ``"random"``: sample ``keep_top_k`` distinct steps uniformly among steps that have
        at least one candidate (requires ``keep_top_k`` > 0; ``keep_top_frac`` is ignored).
        Use ``random_seed`` for reproducibility.
    """
    tmax = trajectory_max_step(steps)
    candidates: List[Dict[str, Any]] = []
    excl: Optional[Set[int]] = None
    if exclude_token_ids is not None:
        excl = {int(x) for x in exclude_token_ids}

    for st in steps:
        sn = int(st["step"])
        info = st.get("information") or {}
        cu = info.get("current_uncertainty")
        ig_raw = info.get(ig_field)
        n_m = count_masked_positions(st)
        if is_block_boundary_step(sn, block_size, tmax):
            continue
        ig, reason = stabilize_ig(cu, None, ig_raw, n_m, u_floor, clip_lo, clip_hi)
        if ig is None:
            continue
        if ig < min_advantage:
            continue
        # per-token ranking helper
        if normalize_by_sqrt_n_masked:
            ig_rank = ig / max(math.sqrt(float(n_m)), 1.0)
        else:
            ig_rank = float(ig)
        for dt in st.get("decoded_tokens") or []:
            tid = int(dt["token_id"])
            if excl is not None and tid in excl:
                continue
            candidates.append(
                {
                    "pivot_step": sn,
                    "position": int(dt["position"]),
                    "token_id": tid,
                    "advantage_mag": float(ig),
                    "ig_rank": float(ig_rank),
                    "n_masked": int(n_m),
                }
            )

    if not candidates:
        return []
    if aggregate_topk_by_step:
        if step_pick not in ("top", "random"):
            raise ValueError(f"step_pick must be 'top' or 'random', got {step_pick!r}")
        if step_pick == "random":
            if keep_top_k is None or int(keep_top_k) <= 0:
                raise ValueError("step_pick='random' with aggregate_topk_by_step requires keep_top_k > 0.")
            rng = random.Random(int(random_seed))
            return _filter_candidates_by_random_steps(
                candidates,
                k=int(keep_top_k),
                rng=rng,
            )
        return _filter_candidates_by_top_steps(
            candidates,
            keep_top_frac=keep_top_frac,
            keep_top_k=keep_top_k,
        )

    if keep_top_k is not None and keep_top_k > 0:
        k = min(int(keep_top_k), len(candidates))
        # Deterministic top-k: highest ig_rank first; tie-break by earlier step/position.
        return sorted(
            candidates,
            key=lambda c: (-float(c["ig_rank"]), int(c["pivot_step"]), int(c["position"])),
        )[:k]

    scores = np.array([c["ig_rank"] for c in candidates], dtype=np.float64)
    thr = np.quantile(scores, max(0.0, 1.0 - keep_top_frac))
    filtered = [c for c in candidates if c["ig_rank"] >= thr]
    return filtered
