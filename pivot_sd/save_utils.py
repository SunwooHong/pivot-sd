"""Shared checkpoint scheduling helpers for SFT trainers."""
from __future__ import annotations

import inspect
from typing import Any

from transformers import TrainerCallback, TrainingArguments


class SaveAtStepsCallback(TrainerCallback):
    """Save checkpoints only at explicit global steps."""

    def __init__(self, save_at: set[int]):
        self.save_at = save_at

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_at:
            control.should_save = True
        return control


class StopAtStepsCallback(TrainerCallback):
    """Stop training once global_step reaches ``stop_at`` (still uses max_steps for LR)."""

    def __init__(self, stop_at: int):
        self.stop_at = int(stop_at)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_at:
            control.should_save = True
            control.should_training_stop = True
        return control


def parse_save_at_steps(raw: str | None) -> list[int]:
    if not raw or not raw.strip():
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step <= 0:
            raise SystemExit(f"--save-at-steps entries must be positive integers, got {step}")
        out.append(step)
    return sorted(set(out))


def apply_save_at_steps(
    *,
    save_at_steps: str | None,
    save_steps: int,
    save_kw: dict[str, Any],
    trainer_callbacks: list[TrainerCallback],
    stop_at_steps: int | None = None,
) -> None:
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    save_at = parse_save_at_steps(save_at_steps)
    if save_at:
        save_kw["save_strategy"] = "no"
        if "save_on_train_end" in ta_params:
            save_kw["save_on_train_end"] = False
        trainer_callbacks.append(SaveAtStepsCallback(set(save_at)))
        print(f"[train] save-at-steps={save_at}", flush=True)
    elif save_steps > 0:
        save_kw["save_strategy"] = "steps"
        save_kw["save_steps"] = save_steps
        if "save_on_train_end" in ta_params:
            save_kw["save_on_train_end"] = True
    else:
        save_kw["save_strategy"] = "no"

    if stop_at_steps is not None and int(stop_at_steps) > 0:
        trainer_callbacks.append(StopAtStepsCallback(int(stop_at_steps)))
        print(f"[train] stop-at-steps={int(stop_at_steps)} (LR still follows max_steps)", flush=True)
