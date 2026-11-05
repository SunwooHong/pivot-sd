"""Dream-v0-Instruct-7B maskgit sampler with the same stepwise log as LLaDA.

Dream's generation shifts logits by one position and unmasks several tokens
per step. Information gain uses every still-masked response position, since
Dream does not decode inside a fixed block.
"""
import json
import os

import torch
import torch.nn.functional as F


def align_dream_logits(logits):
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    drop = cumulative > top_p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, drop)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _sample_tokens(logits, temperature, top_p):
    sampling = logits
    if temperature > 0:
        sampling = sampling / temperature
    if top_p is not None and top_p < 1:
        sampling = _top_p(sampling, top_p)
    probs = F.softmax(sampling, dim=-1)
    if temperature > 0:
        x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    return confidence, x0


@torch.no_grad()
def dream_maskgit_decode(
    model,
    tokenizer,
    input_ids_original,
    gen_length,
    steps,
    temperature=0.0,
    top_p=0.95,
    mask_id=None,
    step_save_dir=None,
    example_idx=0,
    eps=1e-3,
):
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Dream tokenizer has no mask_token_id")
    mask_id = int(mask_id)

    prompt_len = input_ids_original.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt_len] = input_ids_original

    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    track = step_save_dir is not None
    records = []
    pending = None
    eps_ent = 1e-10
    used_steps = 0

    for i in range(steps):
        mask_index = x == mask_id
        if not mask_index.any():
            break
        logits = align_dream_logits(model(x, attention_mask="full", use_cache=False).logits)
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + eps_ent)).sum(dim=-1)

        if pending is not None:
            u_next = float(entropy[mask_index].sum().item())
            info = records[pending]["information"]
            info["next_uncertainty"] = round(u_next, 6)
            den = max(int(info["n_excl"]), 1)
            info["info_gain_density"] = round(
                (info["current_uncertainty_excl_decoded"] - info["next_uncertainty"]) / den, 6
            )
            pending = None

        t = timesteps[i]
        s = timesteps[i + 1]
        mask_logits = logits[mask_index]
        confidence, x0 = _sample_tokens(mask_logits, temperature, top_p if temperature > 0 else None)
        n_mask = mask_index.sum() / mask_index.shape[0]
        n_transfer = int(n_mask * (1 - s / t)) if i < steps - 1 else int(n_mask)

        transfer = torch.zeros_like(x, dtype=torch.bool)
        if n_transfer > 0:
            full_conf = torch.full(x.shape, -torch.inf, device=x.device, dtype=confidence.dtype)
            full_conf[mask_index] = confidence
            _, chosen = torch.topk(full_conf, n_transfer)
            rows = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(chosen)
            picked = torch.zeros_like(x, dtype=torch.long) + mask_id
            picked[mask_index] = x0
            x[rows, chosen] = picked[rows, chosen]
            transfer[rows, chosen] = True

        if track and transfer.any():
            remain = mask_index & ~transfer
            decoded = []
            for idx in torch.where(transfer[0])[0]:
                token_id = int(x[0, idx].item())
                decoded.append({
                    "position": int(idx),
                    "token_id": token_id,
                    "decoded_token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "confidence": round(float(full_conf[0, idx].item()), 4),
                    "entropy": round(float(entropy[0, idx].item()), 6),
                })
            text = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
            records.append({
                "step": i + 1,
                "decoded_text": text,
                "decoded_tokens_num": len(decoded),
                "decoded_tokens": decoded,
                "information": {
                    "current_uncertainty": round(float(entropy[mask_index].sum().item()), 6),
                    "current_uncertainty_excl_decoded": round(float(entropy[remain].sum().item()), 6),
                    "next_uncertainty": None,
                    "n_excl": int(remain.sum().item()),
                },
            })
            pending = len(records) - 1

        used_steps = i + 1

    if pending is not None:
        info = records[pending]["information"]
        info["next_uncertainty"] = 0.0
        den = max(int(info["n_excl"]), 1)
        info["info_gain_density"] = round(info["current_uncertainty_excl_decoded"] / den, 6)

    if step_save_dir:
        path = os.path.join(step_save_dir, f"all_steps_{example_idx}.json")
        with open(path, "w") as handle:
            json.dump(records, handle, indent=2)
    return x, used_steps
