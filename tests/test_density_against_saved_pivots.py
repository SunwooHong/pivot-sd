"""Check info_gain_density against a trajectory that is already in the pivot JSONL."""
from __future__ import annotations

import json
from pathlib import Path

from pivot_sd.scoring import extract_step_pivots, inject_info_gain_density

ROOT = Path(__file__).resolve().parents[2]
STEPWISE = Path(
    "/home/seohyun/scratch/data/math/llada/train_token_entropy_all_s2000/"
    "q500/LLaDA/math/default/all/len_256_block_64/steps_256/s4/stepwise/all_steps_q3_s0.json"
)
PIVOTS = ROOT / "d1/SFT_IG/data/llada/sample_200_v1/math/pivots_adv_scaled_200_g0.jsonl"


def test_density_formula_on_one_block() -> None:
    steps = []
    for step, n_decoded, excl, nxt in (
        (1, 1, 63.0, 53.0),
        (2, 1, 20.0, 10.0),
    ):
        steps.append(
            {
                "step": step,
                "decoded_tokens_num": n_decoded,
                "decoded_tokens": [{"position": step, "token_id": 1}],
                "information": {
                    "current_uncertainty": excl + 1.0,
                    "current_uncertainty_excl_decoded": excl,
                    "next_uncertainty": nxt,
                },
            }
        )
    inject_info_gain_density(steps, 64)
    # Remaining masks: 63, then 62.
    assert steps[0]["information"]["info_gain_density"] == round((63.0 - 53.0) / 63, 6)
    assert steps[1]["information"]["info_gain_density"] == round((20.0 - 10.0) / 62, 6)


def test_density_matches_saved_rows() -> None:
    if not STEPWISE.is_file() or not PIVOTS.is_file():
        return
    steps = json.loads(STEPWISE.read_text())
    inject_info_gain_density(steps, 64)
    by_step = {int(s["step"]): s for s in steps}
    saved = []
    with PIVOTS.open() as f:
        for line in f:
            row = json.loads(line)
            if row["stepwise_path"] == str(STEPWISE):
                saved.append(row)
    assert saved, "expected pivots for the q500 q3 trajectory"
    for row in saved:
        got = by_step[int(row["pivot_step"])]["information"]["info_gain_density"]
        assert got == float(row["info_gain_density"])
    chosen = extract_step_pivots(steps, keep_top_k=10, exclude_token_ids=None)
    got_steps = sorted({int(c["pivot_step"]) for c in chosen})
    saved_steps = sorted({int(r["pivot_step"]) for r in saved})
    assert got_steps == saved_steps
