"""Check info_gain_density on a two-step block."""
from __future__ import annotations

from pivot_sd.scoring import inject_info_gain_density


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
    assert steps[0]["information"]["info_gain_density"] == round((63.0 - 53.0) / 63, 6)
    assert steps[1]["information"]["info_gain_density"] == round((20.0 - 10.0) / 62, 6)
