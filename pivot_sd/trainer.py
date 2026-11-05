"""
Pivot training on a masked state ``M_t``.

Correct rows (advantage >= 0) use cross-entropy. Incorrect rows use unlikelihood.
The advantage magnitude is not a loss weight; only the sign selects the term.

``positive_mask_loss`` / ``negative_mask_loss``:
  - ``pivot_only``: the committed pivot token.
  - ``uniform``: every supervised masked position in the state.
  - ``block_random_pivot_only``: one random supervised masked index in the pivot's block.

Supervision span: ``[prompt_len, first <|eot_id|> in the generation)``.
"""
from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from pivot_sd.ig_pivot_utils import (
    build_m_t_input_ids,
    count_unmask_map_clean_mismatches,
    cumulative_unmask_map_before,
    file_generation_start_index,
    inferred_stepwise_position_offset,
    stepwise_position_offset_for_file_gen_start,
    supervision_exclusive_end_1d,
    supervision_stop_token_id,
    get_problem_text,
    last_step_decoded_text,
    load_stepwise,
    load_train_rows,
    normalize_math_train_row,
    tokenize_math_row,
    tokenize_math_row_with_assistant,
)

_IGPIVOT_UNMASK_WARN_PRINTED = 0

# Like ``sft_trainer_ig_weighted._cached_stepwise`` (there maxsize=128): bound RAM from huge stepwise JSONs.
# Raise ``IGPIVOT_STEPWISE_LRU`` if you have RAM and hate disk re-read; lower if CPU OOM.
_disk_stepwise_lru_loaded: Optional[Any] = None


def _disk_stepwise_from_file(path: str) -> List[Dict[str, Any]]:
    """Process-wide LRU over parsed stepwise files (paths as keys)."""
    global _disk_stepwise_lru_loaded
    if _disk_stepwise_lru_loaded is None:
        raw = os.environ.get("IGPIVOT_STEPWISE_LRU", "512").strip()
        try:
            maxsize = max(32, int(raw))
        except ValueError:
            maxsize = 512

        @lru_cache(maxsize=maxsize)
        def _inner(p: str) -> List[Dict[str, Any]]:
            return load_stepwise(p)

        _disk_stepwise_lru_loaded = _inner

    return _disk_stepwise_lru_loaded(os.path.abspath(path))


def _disk_stepwise_cache_line() -> str:
    fn = _disk_stepwise_lru_loaded
    if fn is None:
        return "disk LRU inactive"
    ci = fn.cache_info()
    return f"disk LRU {ci.currsize}/{ci.maxsize} hits={ci.hits} misses={ci.misses}"


KODCODE_SYSTEM_PROMPT_V2 = (
    "You are an expert Python programmer. You must write complete, valid Python code "
    "to solve the user's problem."
)
ACECODE_PROMPT_NUM_TESTS = 3


def _is_acecode_row(item: Dict[str, Any]) -> bool:
    return isinstance(item, dict) and item.get("test_cases") is not None


def _acecode_v3_user_prompt_text(
    item: Dict[str, Any],
    *,
    include_tests_in_prompt: bool = True,
    num_prompt_tests: int = ACECODE_PROMPT_NUM_TESTS,
) -> Optional[str]:
    """AceCode kodcode_v3 layout, with the first test cases in the prompt."""
    if not isinstance(item, dict):
        return None
    prompt = (item.get("question") or item.get("problem") or "").strip()
    if not prompt:
        return None
    footer = (
        "You must output ONLY raw Python code.\n"
        "Do not include any conversational text or explanations before or after the code."
    )
    parts = [prompt]
    if include_tests_in_prompt:
        test_cases = item.get("test_cases")
        if test_cases:
            prompt_tests = list(test_cases)[:num_prompt_tests]
            tests_str = "\n".join(str(t) for t in prompt_tests)
            parts.append(f"Your code should pass these tests:\n\n{tests_str}")
    parts.append(footer)
    return "\n\n".join(parts)


def _stepwise_user_and_system(
    train_row: Dict[str, Any],
    default_system: str,
    code_prompt_style: str = "kodcode_v3",
    acecode_include_tests_in_prompt: bool = False,
) -> tuple[Optional[str], str]:
    del code_prompt_style
    if _is_acecode_row(train_row):
        return (
            _acecode_v3_user_prompt_text(
                train_row, include_tests_in_prompt=acecode_include_tests_in_prompt
            ),
            KODCODE_SYSTEM_PROMPT_V2,
        )
    return get_problem_text(train_row), default_system


SYSTEM_PROMPT = """Your task is to answer the question below. Give step by step reasoning before you answer, and when you're ready to answer, please use the format 'The final answer is'."""


def _supervision_exclusive_end(
    clean_input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    stop_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """
    Per-row exclusive end index: supervised (loss-eligible) positions i satisfy
    ``prompt_len <= i < exclusive_end`` within the cleaned sequence.

    If ``stop_token_id`` (typically ``<|eot_id|>``) appears in the generation span, ``exclusive_end``
    is one past the **first** such index. If not, supervision runs through the last non-padding
    token in the generation span.
    """
    B, L = clean_input_ids.shape
    device = clean_input_ids.device
    pl = prompt_lengths.to(device=device, dtype=torch.long).clamp(min=0)
    idx = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    in_gen = idx >= pl.unsqueeze(1)

    if pad_token_id is not None:
        non_pad = clean_input_ids != pad_token_id
    else:
        non_pad = torch.ones(B, L, device=device, dtype=torch.bool)

    valid_for_last = in_gen & non_pad
    masked_np = torch.where(
        valid_for_last, idx, torch.full((B, L), -1, device=device, dtype=torch.long)
    )
    last_np = masked_np.max(dim=1).values
    has_gen_content = valid_for_last.any(dim=1)

    end_no_eos = torch.where(has_gen_content, last_np + 1, pl)

    if stop_token_id is None:
        return end_no_eos

    is_stop = (clean_input_ids == stop_token_id) & in_gen
    masked_s = torch.where(
        is_stop, idx, torch.full((B, L), L, device=device, dtype=torch.long)
    )
    first_stop = masked_s.min(dim=1).values
    has_stop_in_gen = is_stop.any(dim=1)
    end_at_first_stop = first_stop + 1
    return torch.where(has_stop_in_gen, end_at_first_stop, end_no_eos)


def _safe_decode_ids(tok, ids_1d, max_chars: int = 8000) -> str:
    if tok is None:
        return "(no tokenizer: cannot decode)"
    try:
        x = ids_1d
        if hasattr(x, "detach"):
            x = x.detach()
        t = x.tolist() if hasattr(x, "tolist") else list(x)
        s = tok.decode(t, skip_special_tokens=False)
        if len(s) > max_chars:
            return s[:max_chars] + f"... [truncated, total_chars={len(s)}]"
        return s
    except Exception as e:
        return f"<decode error: {e}>"


def _one_piece(tok, tid: int) -> str:
    if tok is None:
        return "?"
    try:
        pieces = tok.convert_ids_to_tokens([int(tid)])
        return repr(pieces[0]) if pieces else str(tid)
    except Exception:
        return str(tid)


def _gen_span_token_markers(
    input_ids_row,
    valid_row: torch.Tensor,
    pl: int,
    excl: int,
    position: int,
    mask_id: int,
) -> str:
    """One char per token index in [pl, excl): ^ pivot, M masked input, u unmasked, x not supervised."""
    out: List[str] = []
    L = int(input_ids_row.shape[0])
    hi = min(excl, L)
    lo = max(0, pl)
    for i in range(lo, hi):
        if i == position:
            out.append("^")
        elif not bool(valid_row[i].item()):
            out.append("x")
        elif int(input_ids_row[i].item()) == mask_id:
            out.append("M")
        else:
            out.append("u")
    return "".join(out)


_CACHE_VERSION = 1


def _file_stat_blob(path: str) -> Dict[str, Any]:
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return {"abs": ap, "missing": True}
    st = os.stat(ap)
    return {"abs": ap, "missing": False, "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _dataset_init_fingerprint(
    pivot_jsonl: str,
    train_json_path: str,
    tokenizer_name: str,
    max_length: int,
    mask_id: int,
    use_stepwise_completion: bool,
    stepwise_positions: str,
    positive_advantage_only: bool,
    code_prompt_style: str = "kodcode",
    acecode_include_tests_in_prompt: bool = False,
) -> Dict[str, Any]:
    return {
        "v": _CACHE_VERSION,
        "pivot": _file_stat_blob(pivot_jsonl),
        "train": _file_stat_blob(train_json_path),
        "tokenizer_name": str(tokenizer_name),
        "max_length": int(max_length),
        "mask_id": int(mask_id),
        "use_stepwise_completion": bool(use_stepwise_completion),
        "stepwise_positions": str(stepwise_positions).strip().lower(),
        "positive_advantage_only": bool(positive_advantage_only),
        "code_prompt_style": "kodcode_v3" if acecode_include_tests_in_prompt else "default",
        "acecode_include_tests_in_prompt": bool(acecode_include_tests_in_prompt),
    }


def _pivot_record_advantage(rec: Dict[str, Any]) -> Optional[float]:
    v = rec.get("advantage", None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class IGPivotDataset(Dataset):
    def __init__(
        self,
        pivot_jsonl: str,
        train_json_path: str,
        tokenizer,
        max_length: int,
        mask_id: int = 126336,
        system_prompt: str = SYSTEM_PROMPT,
        use_stepwise_completion: bool = True,
        stepwise_positions: str = "auto",
        positive_advantage_only: bool = False,
        init_cache_path: Optional[str] = None,
        code_prompt_style: str = "kodcode",
        acecode_include_tests_in_prompt: bool = False,
    ):
        super().__init__()
        self.train_rows = load_train_rows(train_json_path)
        if not self.train_rows:
            raise ValueError("train-json is empty or could not be parsed")

        tok_name = getattr(tokenizer, "name_or_path", None) or str(tokenizer)
        norm_code_style = "kodcode_v3" if acecode_include_tests_in_prompt else "default"
        fp = _dataset_init_fingerprint(
            pivot_jsonl,
            train_json_path,
            tok_name,
            max_length,
            mask_id,
            use_stepwise_completion,
            stepwise_positions,
            positive_advantage_only,
            code_prompt_style=norm_code_style,
            acecode_include_tests_in_prompt=acecode_include_tests_in_prompt,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_id = mask_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.system_prompt = system_prompt
        self.code_prompt_style = norm_code_style
        self.acecode_include_tests_in_prompt = bool(acecode_include_tests_in_prompt)
        self.use_stepwise_completion = use_stepwise_completion
        sp = (stepwise_positions or "auto").strip().lower()
        if sp not in ("absolute", "relative_gen", "auto", "file_gen_start"):
            raise ValueError(
                "stepwise_positions must be one of: absolute, relative_gen, auto, file_gen_start "
                f"(got {stepwise_positions!r})"
            )
        self.stepwise_positions = sp
        self._file_gen_start_cache: Dict[str, Optional[int]] = {}
        self._stepwise_offset_cache: Dict[tuple, int] = {}
        self._tok_cache: Dict[tuple, tuple] = {}
        self._invalid_keys: Dict[tuple, str] = {}
        path_final_text: Dict[str, str] = {}
        self._path_final_text = path_final_text
        self.rows: List[Dict[str, Any]] = []

        cache_path = (init_cache_path or "").strip()
        if cache_path:
            ap_cache = os.path.abspath(cache_path)
            if os.path.isfile(ap_cache):
                try:
                    with open(ap_cache, "rb") as cf:
                        blob = pickle.load(cf)
                    if (
                        isinstance(blob, dict)
                        and blob.get("fingerprint") == fp
                        and isinstance(blob.get("rows"), list)
                    ):
                        self.rows = blob["rows"]
                        pft = blob.get("path_final_text")
                        if isinstance(pft, dict):
                            self._path_final_text = {str(k): str(v) for k, v in pft.items()}
                        print(
                            f"[IGPivotDataset] loaded init cache: {ap_cache} "
                            f"({len(self.rows)} rows; skipped pivot scan).",
                            flush=True,
                        )
                        return
                    print(
                        f"[IGPivotDataset] init cache fingerprint mismatch or bad format: {ap_cache} "
                        f"(rebuilding...)",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[IGPivotDataset] init cache load failed ({ap_cache}): {e}; rebuilding...",
                        flush=True,
                    )

        raw_rows: List[Dict[str, Any]] = []
        with open(pivot_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_rows.append(json.loads(line))

        if positive_advantage_only:
            n0 = len(raw_rows)
            pruned: List[Dict[str, Any]] = []
            dropped_adv = 0
            for rec in raw_rows:
                adv = _pivot_record_advantage(rec)
                if adv is None or adv < 0:
                    dropped_adv += 1
                    continue
                pruned.append(rec)
            raw_rows = pruned
            print(
                f"[IGPivotDataset] positive_advantage_only: kept {len(raw_rows)}/{n0} pivot rows "
                f"(dropped {dropped_adv} with advantage<0 or invalid/missing advantage).",
                flush=True,
            )

        _log_every = max(1, int(os.environ.get("IGPIVOT_DATASET_LOG_EVERY", "2000")))
        _any_embedded = any(
            isinstance(r.get("training_steps"), list) and len(r.get("training_steps") or []) > 0
            for r in raw_rows
        )
        _mode = (
            "embedded training_steps (no per-path stepwise files)"
            if _any_embedded
            else "stepwise JSON cached per path in RAM"
        )
        print(
            f"[IGPivotDataset] loaded {len(raw_rows)} pivot rows from {pivot_jsonl}; "
            f"single-pass scan ({_mode}; can take many minutes). "
            f"Progress every {_log_every} rows (set IGPIVOT_DATASET_LOG_EVERY to change).",
            flush=True,
        )

        dropped = 0
        dropped_bad_position = 0
        for i, rec in enumerate(raw_rows, start=1):
            if i % _log_every == 0:
                print(
                    f"[IGPivotDataset] scan {i}/{len(raw_rows)} "
                    f"(kept {len(self.rows)}; dropped {dropped}; "
                    f"{_disk_stepwise_cache_line()}; "
                    f"path_final_text keys: {len(self._path_final_text)})",
                    flush=True,
                )
            try:
                orig_idx = int(rec["orig_idx"])
            except Exception:
                dropped += 1
                continue
            try:
                pos = int(rec["position"])
            except Exception:
                dropped += 1
                continue
            if pos < 0:
                dropped += 1
                dropped_bad_position += 1
                continue

            embedded = isinstance(rec.get("training_steps"), list) and len(rec.get("training_steps") or []) > 0
            if self.use_stepwise_completion:
                p_key = str(rec.get("original_stepwise_path") or rec.get("stepwise_path") or "").strip()
                emb_inline = (rec.get("stepwise_decoded_text") or "").strip()
                if emb_inline and p_key and p_key not in self._path_final_text:
                    self._path_final_text[p_key] = emb_inline
                elif p_key and p_key not in self._path_final_text and not embedded:
                    try:
                        st0 = self._ensure_stepwise(p_key)
                        self._path_final_text[p_key] = last_step_decoded_text(st0)
                    except Exception:
                        self._path_final_text[p_key] = ""

            cache_key, assistant_content = self._cache_key_and_assistant(rec, orig_idx)
            if self.use_stepwise_completion and not assistant_content:
                dropped += 1
                continue
            out = self._get_clean_tokenized(cache_key, orig_idx, assistant_content, strict=False)
            if out is None:
                dropped += 1
                continue
            clean_ids, prompt_len = out
            if embedded:
                st = rec["training_steps"]
                sp_key = str(rec.get("original_stepwise_path") or "").strip() or "embedded"
            else:
                sp_path = str(rec.get("stepwise_path", "") or "").strip()
                if not sp_path:
                    dropped += 1
                    continue
                st = self._ensure_stepwise(sp_path)
                sp_key = sp_path
            try:
                gen_length = int(rec.get("gen_length", 256))
                pos_off = self._stepwise_position_offset(sp_key, st, int(prompt_len), gen_length)
                final_pos = int(rec["position"]) + pos_off
                L = int(clean_ids.numel())
                if final_pos < 0 or final_pos >= L:
                    dropped += 1
                    dropped_bad_position += 1
                    continue
            except Exception:
                dropped += 1
                continue
            self.rows.append(rec)

        print(
            f"[IGPivotDataset] init done: {len(self.rows)} trainable rows "
            f"(from {len(raw_rows)} pivot records; dropped {dropped}).",
            flush=True,
        )
        if dropped:
            msg = f"[IGPivotDataset] dropped {dropped} invalid pivot rows during init"
            if dropped_bad_position:
                msg += (
                    f" ({dropped_bad_position} with pivot index out of range after tokenization / "
                    f"position offset; max_length={self.max_length})"
                )
            print(msg)

        if cache_path:
            ap_cache = os.path.abspath(cache_path)
            try:
                os.makedirs(os.path.dirname(ap_cache) or ".", exist_ok=True)
                blob = {
                    "fingerprint": fp,
                    "rows": self.rows,
                    "path_final_text": dict(self._path_final_text),
                }
                with open(ap_cache, "wb") as cf:
                    pickle.dump(blob, cf, protocol=4)
                print(
                    f"[IGPivotDataset] wrote init cache ({len(self.rows)} rows) -> {ap_cache}",
                    flush=True,
                )
            except Exception as e:
                print(f"[IGPivotDataset] warning: could not write init cache: {e}", flush=True)

    def _ensure_stepwise(self, path: str) -> List[Dict[str, Any]]:
        """Load stepwise JSON; shared process LRU caps RAM (aligned with weighted trainer intent)."""
        return _disk_stepwise_from_file(path)

    def _cached_file_generation_start(self, stepwise_path: str, steps: List[Dict[str, Any]]) -> Optional[int]:
        if stepwise_path not in self._file_gen_start_cache:
            self._file_gen_start_cache[stepwise_path] = file_generation_start_index(steps)
        return self._file_gen_start_cache[stepwise_path]

    def _stepwise_position_offset(
        self,
        stepwise_path: str,
        steps: List[Dict[str, Any]],
        prompt_len: int,
        gen_length: int,
    ) -> int:
        """Map stepwise JSON ``position`` into ``clean_input_ids`` indices (see ig_pivot_utils)."""
        if self.stepwise_positions == "absolute":
            return 0
        if self.stepwise_positions == "relative_gen":
            return int(prompt_len)
        if self.stepwise_positions == "file_gen_start":
            gs = self._cached_file_generation_start(stepwise_path, steps)
            return stepwise_position_offset_for_file_gen_start(int(prompt_len), gs)
        # auto: first-step all_tokens min (gen stream start) vs training prompt_len, else legacy heuristic
        gs = self._cached_file_generation_start(stepwise_path, steps)
        if gs is not None:
            return stepwise_position_offset_for_file_gen_start(int(prompt_len), gs)
        key = (stepwise_path, int(prompt_len), int(gen_length))
        if key not in self._stepwise_offset_cache:
            self._stepwise_offset_cache[key] = inferred_stepwise_position_offset(
                steps, prompt_len, gen_length
            )
        return self._stepwise_offset_cache[key]

    def _cache_key_and_assistant(self, rec: Dict[str, Any], orig_idx: int) -> tuple[tuple, str]:
        if not self.use_stepwise_completion:
            return (orig_idx, "train"), ""
        path = str(rec.get("stepwise_path") or rec.get("original_stepwise_path") or "").strip()
        text = self._path_final_text.get(path, "").strip()
        return (orig_idx, path), text

    def __len__(self):
        return len(self.rows)

    def _get_clean_tokenized(
        self,
        cache_key: tuple,
        orig_idx: int,
        assistant_content: str,
        strict: bool = True,
    ):
        if cache_key in self._invalid_keys:
            if strict:
                raise ValueError(self._invalid_keys[cache_key])
            return None

        if cache_key not in self._tok_cache:
            if orig_idx < 0 or orig_idx >= len(self.train_rows):
                msg = f"orig_idx {orig_idx} out of range for train rows len={len(self.train_rows)}"
                self._invalid_keys[cache_key] = msg
                if strict:
                    raise ValueError(msg)
                return None
            raw = self.train_rows[orig_idx]
            if self.use_stepwise_completion:
                u, sys_p = _stepwise_user_and_system(
                    raw,
                    self.system_prompt,
                    self.code_prompt_style,
                    acecode_include_tests_in_prompt=self.acecode_include_tests_in_prompt,
                )
                if not u:
                    msg = f"train row {orig_idx} missing problem or question"
                    self._invalid_keys[cache_key] = msg
                    if strict:
                        raise ValueError(msg)
                    return None
                if not assistant_content:
                    msg = f"empty stepwise completion for {cache_key}"
                    self._invalid_keys[cache_key] = msg
                    if strict:
                        raise ValueError(msg)
                    return None
                out = tokenize_math_row_with_assistant(
                    u,
                    assistant_content,
                    self.tokenizer,
                    self.max_length,
                    sys_p,
                )
            else:
                out = tokenize_math_row(raw, self.tokenizer, self.max_length, self.system_prompt)
            if out is None:
                src = "stepwise" if self.use_stepwise_completion else "train"
                msg = f"Could not tokenize train row {orig_idx} (completion_source={src})"
                self._invalid_keys[cache_key] = msg
                if strict:
                    raise ValueError(msg)
                return None
            self._tok_cache[cache_key] = out
        return self._tok_cache[cache_key]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.rows[idx]
        orig_idx = int(rec["orig_idx"])
        cache_key, assistant_content = self._cache_key_and_assistant(rec, orig_idx)
        clean_ids, prompt_len = self._get_clean_tokenized(cache_key, orig_idx, assistant_content)
        ts = rec.get("training_steps")
        if isinstance(ts, list) and len(ts) > 0:
            steps = ts
            sp_key = str(rec.get("original_stepwise_path") or "").strip() or "embedded"
        else:
            sp_file = str(rec.get("stepwise_path") or rec.get("original_stepwise_path") or "").strip()
            if not sp_file:
                raise ValueError("row missing stepwise_path and training_steps")
            steps = self._ensure_stepwise(sp_file)
            sp_key = sp_file
        pivot_step = int(rec["pivot_step"])
        gen_length = int(rec.get("gen_length", 256))
        pos_off = self._stepwise_position_offset(sp_key, steps, int(prompt_len), gen_length)
        unmap = cumulative_unmask_map_before(steps, pivot_step, position_offset=pos_off)
        pl_i = int(prompt_len)
        stop_id = supervision_stop_token_id(self.tokenizer)
        Lseq = int(clean_ids.numel())
        gen_cap = min(pl_i + gen_length, Lseq)
        gen_excl: Optional[int] = None
        if stop_id is not None:
            gen_excl = min(gen_cap, supervision_exclusive_end_1d(clean_ids, pl_i, stop_id, self.pad_id))
        m_t = build_m_t_input_ids(
            clean_ids,
            prompt_len=pl_i,
            gen_length=gen_length,
            mask_id=self.mask_id,
            pad_id=self.pad_id,
            unmask_map=unmap,
            stop_token_id=stop_id,
            clip_to_supervision_end=stop_id is not None,
        )
        global _IGPIVOT_UNMASK_WARN_PRINTED
        if os.environ.get("IGPIVOT_WARN_UNMASK_MISMATCH", "").strip() in ("1", "true", "yes"):
            g, b, fb = count_unmask_map_clean_mismatches(
                unmap, clean_ids, pl_i, gen_length, gen_end_exclusive=gen_excl
            )
            if b > 0 and _IGPIVOT_UNMASK_WARN_PRINTED < 8:
                _IGPIVOT_UNMASK_WARN_PRINTED += 1
                print(
                    f"[IGPivotDataset] unmask_map vs clean disagree in gen window: "
                    f"agree={g} disagree={b} first_bad_pos={fb} "
                    f"(orig_idx={orig_idx} pivot_step={pivot_step}). "
                    "Usually intermediate decode tokens OR stepwise positions not aligned with "
                    "this tokenization (try --stepwise-positions absolute|relative_gen|file_gen_start|auto; "
                    "see ig_pivot_utils.file_generation_start_index)."
                )
        pos = int(rec["position"]) + pos_off
        Lm = int(m_t.numel())
        if pos < 0 or pos >= Lm:
            raise ValueError(
                f"bad position {pos} for idx {idx} (raw={int(rec['position'])} + offset={pos_off}, "
                f"seq_len={Lm}, max_length={self.max_length}). "
                "Init should have dropped this; if you see this, clear dataset cache or report."
            )
        # Supervised token = clean sequence at pivot (same tokenizer as stepwise completion when
        # use_stepwise_completion is True).
        tid = int(clean_ids[pos].item())
        adv = float(rec["advantage"])
        attn = (m_t != self.pad_id).long()
        raw = self.train_rows[orig_idx]
        u_q, _ = _stepwise_user_and_system(
            raw,
            self.system_prompt,
            self.code_prompt_style,
            acecode_include_tests_in_prompt=self.acecode_include_tests_in_prompt,
        )
        qtext = u_q or ""

        # import pdb; pdb.set_trace()
        return {
            "input_ids": m_t,
            "attention_mask": attn,
            "clean_input_ids": clean_ids.clone(),
            "prompt_len": int(prompt_len),
            "position": pos,
            "target_token_id": tid,
            "advantage": adv,
            "question": qtext,
            "orig_idx": int(orig_idx),
            "stepwise_path": str(rec.get("stepwise_path") or rec.get("original_stepwise_path") or ""),
        }


def ig_pivot_collate(features: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
    attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
    clean_input_ids = torch.stack([f["clean_input_ids"] for f in features], dim=0)
    positions = torch.tensor([f["position"] for f in features], dtype=torch.long)
    target_token_id = torch.tensor([f["target_token_id"] for f in features], dtype=torch.long)
    advantage = torch.tensor([f["advantage"] for f in features], dtype=torch.float32)
    prompt_lengths = torch.tensor([f["prompt_len"] for f in features], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "clean_input_ids": clean_input_ids,
        "prompt_lengths": prompt_lengths,
        "positions": positions,
        "target_token_id": target_token_id,
        "advantage": advantage,
        "questions": [f.get("question", "") for f in features],
        "orig_indices": [int(f.get("orig_idx", -1)) for f in features],
        "stepwise_paths": [str(f.get("stepwise_path", "") or "") for f in features],
    }


_BLOCK_RANDOM_PIVOT_ONLY = "block_random_pivot_only"

_VALID_POS_SCOPE = frozenset(
    {"uniform", "pivot_only", _BLOCK_RANDOM_PIVOT_ONLY}
)
_VALID_NEG_SCOPE = frozenset({"uniform", "pivot_only", _BLOCK_RANDOM_PIVOT_ONLY})


def _uses_block_random_pivot(positive_mask_loss: str, negative_mask_loss: str) -> bool:
    return positive_mask_loss == _BLOCK_RANDOM_PIVOT_ONLY or negative_mask_loss == _BLOCK_RANDOM_PIVOT_ONLY


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = (mask.float() * values).sum(dim=1)
    den = mask.sum(dim=1).float().clamp(min=1.0)
    return torch.where(mask.sum(dim=1) > 0, num / den, torch.zeros_like(num))


def _pos_gate(adv: torch.Tensor) -> torch.Tensor:
    """1 on correct trajectories. Advantage magnitude is not a loss weight."""
    return (adv >= 0).float()


def _neg_gate(adv: torch.Tensor) -> torch.Tensor:
    """1 on incorrect trajectories."""
    return (adv < 0).float()


def _random_masked_in_pivot_block(
    masked: torch.Tensor,
    prompt_lengths: torch.Tensor,
    ref_positions: torch.Tensor,
    exclusive_end: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per row, pick a uniform random index among supervised masked positions in the gen block
    that contains ``ref_positions`` (blocks [pl + k*B, pl + (k+1)*B) clipped to the supervision
    span exclusive end).

    Returns:
        torch.LongTensor [B] chosen indices (fallback to ref when no candidate; pair with ok).
        torch.BoolTensor [B] True iff at least one masked candidate existed in the block.
    """
    B, L = masked.shape
    device = masked.device
    chosen = ref_positions.clone()
    ok = torch.zeros(B, dtype=torch.bool, device=device)
    bs = max(int(block_size), 1)
    for i in range(B):
        pl_i = int(prompt_lengths[i].item())
        p_i = int(ref_positions[i].item())
        ex_i = int(min(int(exclusive_end[i].item()), L))
        gen_off = max(0, p_i - pl_i)
        bi = gen_off // bs
        lo = pl_i + bi * bs
        hi = min(lo + bs, ex_i)
        if hi <= lo:
            ok[i] = False
            chosen[i] = p_i
            continue
        idxs = torch.arange(lo, hi, device=device, dtype=torch.long)
        row_m = masked[i, idxs]
        cand = idxs[row_m]
        n = int(cand.numel())
        if n == 0:
            ok[i] = False
            chosen[i] = p_i
        else:
            j = int(torch.randint(n, (1,), device=device).item())
            chosen[i] = int(cand[j].item())
            ok[i] = True
    return chosen, ok


class IGPivotTrainer(Trainer):
    def __init__(
        self,
        *args,
        positive_mask_loss: str = "pivot_only",
        negative_mask_loss: str = "pivot_only",
        mask_id: int = 126336,
        negative_weight_multiplier: float = 0.5,
        debug_train: bool = False,
        debug_max_steps: int = 3,
        pivot_block_size: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_id = int(mask_id)
        self.negative_weight_multiplier = float(negative_weight_multiplier)
        self.debug_train = bool(debug_train)
        self.debug_max_steps = int(debug_max_steps)
        self.pivot_block_size = max(1, int(pivot_block_size))
        self._custom_log_sums: Dict[str, float] = {}
        self._custom_log_counts: Dict[str, int] = {}

        self.positive_mask_loss = positive_mask_loss
        self.negative_mask_loss = negative_mask_loss
        if self.positive_mask_loss not in _VALID_POS_SCOPE:
            raise ValueError(
                f"positive_mask_loss must be one of {_VALID_POS_SCOPE}, got {self.positive_mask_loss!r}"
            )
        if self.negative_mask_loss not in _VALID_NEG_SCOPE:
            raise ValueError(
                f"negative_mask_loss must be one of {_VALID_NEG_SCOPE}, got {self.negative_mask_loss!r}"
            )

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        merged = dict(logs) if logs is not None else {}
        if self._custom_log_sums:
            for k, s in self._custom_log_sums.items():
                c = max(self._custom_log_counts.get(k, 1), 1)
                merged[k] = float(s / c)
            self._custom_log_sums = {}
            self._custom_log_counts = {}
        super().log(merged, *args, **kwargs)

    def _log_ig_loss_breakdown(
        self,
        *,
        is_training: bool,
        log_probs: torch.Tensor,
        inputs: Dict[str, Any],
        clean_input_ids: torch.Tensor,
        positions: torch.Tensor,
        advantage: torch.Tensor,
        per_example_total: torch.Tensor,
        pivot_term: Optional[torch.Tensor],
        pivot_neg_term: Optional[torch.Tensor] = None,
        other_term: Optional[torch.Tensor],
        rank_term: Optional[torch.Tensor],
        loss: torch.Tensor,
        tgt_lp: torch.Tensor,
    ) -> None:
        if not is_training:
            return
        adv = advantage.detach().float()
        logs: Dict[str, float] = {}
        pos = adv >= 0
        neg = adv < 0
        logs["advantage/frac_negative"] = float(neg.float().mean().item())
        if pivot_term is not None:
            pt = pivot_term.detach().float()
            if pos.any():
                logs["loss/pivot_pos"] = float(pt[pos].mean().item())
            if neg.any():
                if pivot_neg_term is not None:
                    pn = pivot_neg_term.detach().float()
                    logs["loss/pivot_neg"] = float(pn[neg].mean().item())
                else:
                    logs["loss/pivot_neg"] = float(pt[neg].mean().item())
        if other_term is not None:
            logs["loss/other_masked"] = float(other_term.detach().float().mean().item())
        if rank_term is not None:
            logs["loss/rank"] = float(rank_term.detach().float().mean().item())
        p_target = torch.exp(tgt_lp.detach()).clamp(max=1.0)
        if pos.any():
            logs["prob/pivot_pos_target"] = float(p_target[pos].mean().item())
        if neg.any():
            logs["prob/pivot_neg_target"] = float(p_target[neg].mean().item())
        for k, v in logs.items():
            self._custom_log_sums[k] = self._custom_log_sums.get(k, 0.0) + float(v)
            self._custom_log_counts[k] = self._custom_log_counts.get(k, 0) + 1

    def _maybe_log_training_debug(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clean_input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_token_id: torch.Tensor,
        advantage: torch.Tensor,
        loss: torch.Tensor,
        tgt_lp: torch.Tensor,
        prompt_lengths: torch.Tensor,
        exclusive_end: torch.Tensor,
        valid_supervision: torch.Tensor,
        pivot_ok: torch.Tensor,
        supervision_stop_id: Optional[int],
        pad_token_id: Optional[int],
        questions: Optional[List[str]],
        orig_indices: Optional[List[int]],
        stepwise_paths: Optional[List[str]],
        pivot_term: Optional[torch.Tensor] = None,
        other_term: Optional[torch.Tensor] = None,
        rank_term: Optional[torch.Tensor] = None,
        pos_loss: Optional[torch.Tensor] = None,
        neg_loss: Optional[torch.Tensor] = None,
        per_ex: Optional[torch.Tensor] = None,
        num_masked: Optional[torch.Tensor] = None,
        num_other: Optional[torch.Tensor] = None,
        loss_masked_bool: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.debug_train or not self.is_world_process_zero():
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        if step >= self.debug_max_steps:
            return
        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        b0 = 0
        ids = input_ids[b0].detach().cpu()
        clean = clean_input_ids[b0].detach().cpu()
        pos = int(positions[b0].item())
        tid = int(target_token_id[b0].item())
        adv = float(advantage[b0].item())
        clean_at_p = int(clean[pos].item())
        pl = int(prompt_lengths[b0].item())
        ex = int(exclusive_end[b0].item())
        p_ok = bool(pivot_ok[b0].item())
        vs = valid_supervision[b0].detach().cpu()
        q = (questions or [""])[b0] if questions else ""
        oid = (orig_indices or [-1])[b0] if orig_indices else -1
        swp = (stepwise_paths or [""])[b0] if stepwise_paths else ""
        pivot_tok = _safe_decode_ids(tok, clean[pos : pos + 1], max_chars=200)
        lines: List[str] = [
            "",
            "=" * 88,
            f"[IGPivotTrainer debug] global_step={step}  example[0]  "
            f"pos_scope={self.positive_mask_loss!r}  neg_scope={self.negative_mask_loss!r}  "
            f"neg=unlikelihood",
            f"orig_idx={oid}  pivot_ok={p_ok}",
            "",
            "--- Raw question ---",
            (q[:4000] + ("..." if len(q) > 4000 else "")) if q else "(empty)",
            "",
            "--- stepwise_path ---",
            swp or "(none)",
            "",
            "--- Bounds ---",
            f"prompt_len={pl}  exclusive_end={ex}",
            "",
            "--- Pivot ---",
            f"position={pos}  target_token_id={tid}  advantage={adv}",
            f"clean[pos]==target_token_id: {clean_at_p == tid}",
            f"decode(clean[pivot]): {pivot_tok!r}",
            f"input_ids[pivot]==mask_id: {bool(ids[pos].item() == self.mask_id)}",
            "",
            "--- Loss weights (trainer hyperparameters) ---",
            f"negative_weight_multiplier (lambda_neg): {self.negative_weight_multiplier}",
            "",
            "--- Loss (values used in backward; example[0]) ---",
        ]
        if per_ex is not None:
            lines.append(f"per_example total (pos_loss + neg_loss [+ rank]): {float(per_ex[b0].item()):.6f}")
        if pos_loss is not None:
            lines.append(f"pos_loss: {float(pos_loss[b0].item()):.6f}")
        if neg_loss is not None:
            lines.append(f"neg_loss: {float(neg_loss[b0].item()):.6f}")
        if rank_term is not None:
            lines.append(f"rank_term: {float(rank_term[b0].item()):.6f}")
        lines.append(f"batch mean loss (Trainer.backward target): {float(loss.item()):.6f}")
        if per_ex is not None:
            m = float(per_ex.detach().float().mean().item())
            lines.append(f"per_ex.mean() (should match batch mean): {m:.6f}")
        lines.extend(
            [
                "",
                "--- Reference (not additive loss terms) ---",
                f"log p(target) at pivot: {float(tgt_lp[b0].item()):.6f}",
            ]
        )
        if pivot_term is not None:
            lines.append(
                f"raw pivot NLL -log p(y_p|x) at pivot (unweighted; excludes lambda_p/adv): "
                f"{float(pivot_term[b0].item()):.6f}"
            )
        if other_term is not None:
            lines.append(
                f"legacy log field other_term (== pos_loss when adv>=0 for breakdown logging): "
                f"{float(other_term[b0].item()):.6f}"
            )
        if loss_masked_bool is not None and tok is not None:
            mb = loss_masked_bool[b0].detach().cpu().bool()
            lines.append(f"masked count in span: {int(mb.sum().item())}")
        lines.extend(["", _gen_span_token_markers(ids, vs, pl, ex, pos, self.mask_id), ""])
        print("\n".join(lines))

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        inputs = dict(inputs)
        questions = inputs.pop("questions", None)
        orig_indices = inputs.pop("orig_indices", None)
        stepwise_paths = inputs.pop("stepwise_paths", None)
        advantage = inputs.pop("advantage")
        positions = inputs.pop("positions")
        target_token_id = inputs.pop("target_token_id")
        clean_input_ids = inputs.pop("clean_input_ids")
        prompt_lengths = inputs.pop("prompt_lengths")

        attn = inputs["attention_mask"]
        dream = getattr(getattr(model, "config", None), "model_type", None) == "dream"
        if dream:
            valid = attn[:, None, None, :].to(dtype=torch.float32)
            inputs["attention_mask"] = (1.0 - valid) * torch.finfo(torch.float32).min
        outputs = model(**inputs)
        logits = outputs.logits
        if dream:
            from pivot_sd.dream_decode import align_dream_logits

            logits = align_dream_logits(logits)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        b = torch.arange(logits.size(0), device=logits.device)

        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        stop_id = supervision_stop_token_id(tok)
        pad_id_tok = getattr(tok, "pad_token_id", None) if tok is not None else None
        pl = prompt_lengths.to(device=logits.device, dtype=torch.long)
        excl = _supervision_exclusive_end(clean_input_ids, pl, stop_id, pad_id_tok)
        L = clean_input_ids.size(1)
        excl = excl.clamp(max=L)
        pos_idx = torch.arange(L, device=logits.device, dtype=torch.long).unsqueeze(0).expand(
            logits.size(0), -1
        )
        valid_supervision = (pos_idx >= pl.unsqueeze(1)) & (pos_idx < excl.unsqueeze(1))
        attn = attn.bool()
        masked = (inputs["input_ids"] == self.mask_id) & attn & valid_supervision

        if _uses_block_random_pivot(self.positive_mask_loss, self.negative_mask_loss):
            positions, pivot_ok = _random_masked_in_pivot_block(
                masked=masked,
                prompt_lengths=pl,
                ref_positions=positions,
                exclusive_end=excl,
                block_size=self.pivot_block_size,
            )
            target_token_id = clean_input_ids[b, positions]
        else:
            pivot_ok = valid_supervision[b, positions]

        tgt_lp = log_probs[b, positions, target_token_id]

        adv = advantage.to(logits.device)
        lam_neg = self.negative_weight_multiplier

        nll = -torch.gather(
            log_probs,
            2,
            clean_input_ids.unsqueeze(-1).clamp(min=0, max=log_probs.size(-1) - 1),
        ).squeeze(-1)

        pivot_nll = nll[b, positions]
        p_pivot = torch.exp(tgt_lp).clamp(max=1.0 - 1e-6)
        ul_pivot = -torch.log1p(-p_pivot)
        p_all = torch.exp(-nll).clamp(max=1.0 - 1e-6)
        ul_all = -torch.log1p(-p_all)

        pos_gate = _pos_gate(adv)
        neg_gate = _neg_gate(adv)

        if self.positive_mask_loss == "uniform":
            pos_loss = _masked_mean(nll, masked) * pos_gate
        elif self.positive_mask_loss in ("pivot_only", _BLOCK_RANDOM_PIVOT_ONLY):
            pos_loss = pivot_nll * pivot_ok.float() * pos_gate
        else:
            raise RuntimeError("unreachable positive_mask_loss")

        if self.negative_mask_loss in ("pivot_only", _BLOCK_RANDOM_PIVOT_ONLY):
            neg_core = ul_pivot * pivot_ok.float()
        elif self.negative_mask_loss == "uniform":
            neg_core = _masked_mean(ul_all, masked)
        else:
            raise RuntimeError("unreachable negative_mask_loss")
        neg_loss = lam_neg * neg_core * neg_gate
        per_ex = pos_loss + neg_loss
        loss = per_ex.mean()

        pivot_term = pivot_nll * pivot_ok.float() * (adv >= 0).float()
        pivot_neg_term_log = ul_pivot * pivot_ok.float() * (adv < 0).float()
        other_term = torch.where(adv >= 0, pos_loss, torch.zeros_like(pos_loss))
        if self.positive_mask_loss in ("pivot_only", _BLOCK_RANDOM_PIVOT_ONLY):
            other_term = torch.zeros_like(pos_loss)

        self._log_ig_loss_breakdown(
            is_training=model.training,
            log_probs=log_probs,
            inputs=inputs,
            clean_input_ids=clean_input_ids,
            positions=positions,
            advantage=advantage,
            per_example_total=per_ex,
            pivot_term=pivot_term,
            pivot_neg_term=pivot_neg_term_log,
            other_term=other_term,
            rank_term=None,
            loss=loss,
            tgt_lp=tgt_lp,
        )
        self._maybe_log_training_debug(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            clean_input_ids=clean_input_ids,
            positions=positions,
            target_token_id=target_token_id,
            advantage=advantage,
            loss=loss.detach(),
            tgt_lp=tgt_lp.detach(),
            prompt_lengths=prompt_lengths,
            exclusive_end=excl,
            valid_supervision=valid_supervision,
            pivot_ok=pivot_ok,
            supervision_stop_id=stop_id,
            pad_token_id=pad_id_tok,
            questions=questions,
            orig_indices=orig_indices,
            stepwise_paths=stepwise_paths,
            pivot_term=pivot_term,
            other_term=other_term,
            rank_term=None,
            pos_loss=pos_loss.detach(),
            neg_loss=neg_loss.detach(),
            per_ex=per_ex.detach(),
            num_masked=masked.sum(dim=1),
            num_other=None,
            loss_masked_bool=masked,
        )
        return (loss, outputs) if return_outputs else loss
def make_data_collator_with_padding(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _collate(features):
        return ig_pivot_collate(features, pad_id)

    return _collate
