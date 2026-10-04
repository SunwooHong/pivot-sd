"""Pivot scores used to build the sample-200 training sets.

The ranking score is ``info_gain_density``, not the relative ratio
``(H_pre - H_post) / (H_pre + eps)``.

For a decode step that commits ``D(k)`` tokens inside the active block::

    density(k) = (U_excl(k) - U_next(k)) / max(|M_B(k) \\ D(k)|, 1)

``U_excl`` is the sum of predictive entropies, on the pre-commit forward, over
masked positions in the active block that are not committed at this step.
``U_next`` is the active-block entropy sum on the next forward, after those
tokens have been written. With one token committed per step this denominator
is the number of masks still left in the block.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def masked_in_block_at_step_start(
    step_1based: int, block_size: int, decoded_by_step: Dict[int, int]
) -> int:
    """Active-block mask count at the start of a 1-based step."""
    b = (step_1based - 1) // block_size
    s0 = b * block_size + 1
    used = sum(decoded_by_step.get(i, 0) for i in range(s0, step_1based))
    return max(block_size - used, 0)


def inject_info_gain_density(steps: List[dict], block_size: int) -> int:
    """Write ``information.info_gain_density`` when it is missing.

    Same rule as ``d1/SFT_IG/build_ig_density.py``.
    """
    if block_size <= 0:
        return 0
    by_step: Dict[int, dict] = {}
    decoded_by_step: Dict[int, int] = {}
    for obj in steps:
        try:
            sk = int(obj["step"])
        except (KeyError, TypeError, ValueError):
            continue
        by_step[sk] = obj
        raw_d = obj.get("decoded_tokens_num", 0)
        try:
            decoded_by_step[sk] = int(raw_d)
        except (TypeError, ValueError):
            decoded_by_step[sk] = 0

    filled = 0
    for k in sorted(by_step.keys()):
        info = by_step[k].get("information")
        if not isinstance(info, dict):
            continue
        if info.get("info_gain_density") is not None:
            continue
        u_excl = info.get("current_uncertainty_excl_decoded")
        nxt = info.get("next_uncertainty")
        if u_excl is None or nxt is None:
            continue
        try:
            v_excl = float(u_excl)
            v_nxt = float(nxt)
        except (TypeError, ValueError):
            continue
        m_k = masked_in_block_at_step_start(k, block_size, decoded_by_step)
        d_k = decoded_by_step.get(k, 0)
        den = max(m_k - d_k, 1)
        info["info_gain_density"] = round((v_excl - v_nxt) / den, 6)
        filled += 1
    return filled


def is_block_boundary_step(step_num: int, block_size: int, max_step: int) -> bool:
    if block_size <= 0:
        return False
    return step_num > 0 and step_num % block_size == 0 and step_num < max_step


def trajectory_max_step(steps: Sequence[dict]) -> int:
    best = 0
    for st in steps:
        try:
            best = max(best, int(st["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    return best


def _filter_candidates_by_top_steps(
    candidates: List[Dict[str, Any]],
    *,
    keep_top_frac: float,
    keep_top_k: Optional[int],
) -> List[Dict[str, Any]]:
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
        scores = [step_score[s] for s in steps_sorted]
        thr = _linear_quantile(scores, max(0.0, 1.0 - keep_top_frac))
        chosen = {s for s in steps_sorted if step_score[s] >= thr}
    return [c for c in candidates if int(c["pivot_step"]) in chosen]


def _filter_candidates_by_random_steps(
    candidates: List[Dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not candidates or k <= 0:
        return []
    steps_unique = sorted({int(c["pivot_step"]) for c in candidates})
    kk = min(int(k), len(steps_unique))
    chosen = set(rng.sample(steps_unique, kk))
    return [c for c in candidates if int(c["pivot_step"]) in chosen]


def _linear_quantile(values: Sequence[float], q: float) -> float:
    """NumPy ``quantile`` default (linear) on a 1-d sample."""
    if not values:
        raise ValueError("empty")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def extract_step_pivots(
    steps: List[dict],
    *,
    ig_field: str = "info_gain_density",
    block_size: int = 64,
    u_floor: float = 0.0,
    clip_lo: float = -1e9,
    clip_hi: float = 1e9,
    keep_top_k: int = 10,
    keep_top_frac: float = 0.2,
    exclude_token_ids: Optional[Iterable[int]] = None,
    step_pick: str = "top",
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Top-K (or random-K) decode steps. One record per decoded token on a kept step.

    Matches ``extract_pivot_records_from_steps(..., normalize_by_sqrt_n_masked=False,
    aggregate_topk_by_step=True)`` from the training-data builder.
    """
    tmax = trajectory_max_step(steps)
    excl: Optional[Set[int]] = None
    if exclude_token_ids is not None:
        excl = {int(x) for x in exclude_token_ids}
    candidates: List[Dict[str, Any]] = []
    for st in steps:
        sn = int(st["step"])
        info = st.get("information") or {}
        cu = info.get("current_uncertainty")
        ig_raw = info.get(ig_field)
        if is_block_boundary_step(sn, block_size, tmax):
            continue
        if cu is None or ig_raw is None:
            continue
        cu_f = float(cu)
        if cu_f < u_floor:
            continue
        ig = max(clip_lo, min(clip_hi, float(ig_raw)))
        if ig < 0.0:
            continue
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
                    "ig_rank": float(ig),
                }
            )
    if not candidates:
        return []
    if step_pick == "random":
        if keep_top_k <= 0:
            raise ValueError("random step selection requires keep_top_k > 0")
        rng = random.Random(int(random_seed))
        return _filter_candidates_by_random_steps(candidates, k=int(keep_top_k), rng=rng)
    if step_pick != "top":
        raise ValueError(f"step_pick must be 'top' or 'random', got {step_pick!r}")
    return _filter_candidates_by_top_steps(
        candidates,
        keep_top_frac=keep_top_frac,
        keep_top_k=keep_top_k if keep_top_k > 0 else None,
    )


def topk_by_token_entropy(
    steps: Sequence[dict],
    *,
    k: int,
    exclude_token_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Highest-entropy committed tokens, across the whole trajectory.

    This is the Entropy-pivot ablation. It does not drop block-boundary steps
    and it ranks tokens, not steps.
    """
    excl = {int(x) for x in exclude_token_ids} if exclude_token_ids is not None else set()
    cands: List[Dict[str, Any]] = []
    for st in steps:
        step_num = int(st["step"])
        for dt in st.get("decoded_tokens") or []:
            if "entropy" not in dt or "position" not in dt or "token_id" not in dt:
                continue
            tid = int(dt["token_id"])
            if tid in excl:
                continue
            cands.append(
                {
                    "pivot_step": step_num,
                    "position": int(dt["position"]),
                    "token_id": tid,
                    "token_entropy": float(dt["entropy"]),
                }
            )
    cands.sort(key=lambda c: (-c["token_entropy"], c["pivot_step"], c["position"]))
    return cands[: min(k, len(cands))]


def percentile_nearest_rank(sorted_vals: Sequence[float], p: float) -> float:
    """Percentile used before min-max scaling advantages (nearest-rank interpolation)."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty")
    if n == 1 or p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (n - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])
    t = k - lo
    return float(sorted_vals[lo]) * (1.0 - t) + float(sorted_vals[hi]) * t


def scale_advantages(
    rows: List[Dict[str, Any]],
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map ``|advantage|`` through the 1st/99th percentile into ``sign * (0.5 + 0.5 x_norm)``.

    Training runs used ``--no-advantage-in-loss``, so only the sign of this value
    selects cross-entropy versus unlikelihood. The raw score is kept on
    ``advantage_raw`` and, for IG rows, ``info_gain_density``.
    """
    abs_sorted = sorted(abs(float(r["advantage"])) for r in rows)
    ig_min = percentile_nearest_rank(abs_sorted, p_low)
    ig_max = percentile_nearest_rank(abs_sorted, p_high)
    if ig_max <= ig_min:
        raise ValueError(f"IG_max ({ig_max}) must exceed IG_min ({ig_min})")
    denom = ig_max - ig_min
    out: List[Dict[str, Any]] = []
    new_advs: List[float] = []
    for row in rows:
        x = float(row["advantage"])
        copied = dict(row)
        m = abs(x)
        m_clipped = max(ig_min, min(m, ig_max))
        x_norm = (m_clipped - ig_min) / denom
        w_p = 0.5 + 0.5 * x_norm
        sgn = 0.0 if x == 0 else (1.0 if x > 0 else -1.0)
        adv_new = sgn * w_p
        copied["advantage_raw"] = x
        copied["advantage"] = adv_new
        out.append(copied)
        new_advs.append(adv_new)
    meta = {
        "n_rows": len(rows),
        "percentile_low": p_low,
        "percentile_high": p_high,
        "IG_min_abs_pctl": ig_min,
        "IG_max_abs_pctl": ig_max,
        "out_advantage_min": min(new_advs) if new_advs else None,
        "out_advantage_max": max(new_advs) if new_advs else None,
    }
    return out, meta
