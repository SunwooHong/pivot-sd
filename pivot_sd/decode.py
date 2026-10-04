import os
import json
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. Float64 matches the LLaDA generation used for the pivot sets."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """How many masked positions to commit at each step of one block."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def _confidence(alg, p_metrics, x0):
    """Score used to choose which masked positions to commit.

    ``default`` is the sampler that built the pivot sets: probability of the
    sampled token. ``random``, ``topk_margin``, and ``entropy`` are the other
    confidence scores kept from the decoder.
    """
    eps = 1e-10
    if alg == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(p_metrics, dim=-1, descending=True)
        return sorted_probs[..., 0] - sorted_probs[..., 1]
    if alg == "entropy":
        log_p = torch.log(p_metrics + eps)
        return -torch.sum(p_metrics * log_p, dim=-1)
    return torch.squeeze(torch.gather(p_metrics, dim=-1, index=torch.unsqueeze(x0, -1)), -1)


@torch.no_grad()
def stable_confident_decode(
    model, tokenizer, input_ids_original, gen_length, steps, block_length, temperature=0., mask_id=126336,
    step_save_dir=None, example_idx=0,
    alg="default",
):
    """Commit tokens inside one LLaDA block at a time.

    With ``gen_length == steps`` this commits exactly one token per step, which
    is the generation used for the sample-200 pivot sets. Stepwise JSON stores
    the active-block entropy sums that ``info_gain_density`` is computed from,
    plus per-token entropy for the entropy-pivot ablation.
    """
    mask_id = 126336
    x = torch.full((1, input_ids_original.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids_original.shape[1]] = input_ids_original.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    used_steps = 0
    all_step_outputs = []
    track = step_save_dir is not None
    pending_idx = None
    eps_ent = 1e-10

    for num_block in range(num_blocks):
        block_start = input_ids_original.shape[1] + num_block * block_length
        block_end = input_ids_original.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            mask_index = mask_index & block_mask
            if not mask_index[:, block_start:block_end].any():
                break

            logits = model(x).logits
            p_metrics = F.softmax(logits.to(torch.float64), dim=-1)
            if temperature > 0:
                logits_sampling = add_gumbel_noise(logits, temperature)
                p_sample = F.softmax(logits_sampling.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_sample, dim=-1)
            else:
                x0 = torch.argmax(p_metrics, dim=-1)

            token_entropy = None
            total_entropy_before_block = None
            if track:
                token_entropy = -torch.sum(p_metrics * torch.log(p_metrics + eps_ent), dim=-1)
                total_entropy_before_block = token_entropy[mask_index].sum()
                if pending_idx is not None:
                    all_step_outputs[pending_idx]["information"]["next_uncertainty"] = round(
                        float(total_entropy_before_block.item()), 6
                    )
                    pending_idx = None

            curr_conf = _confidence(alg, p_metrics, x0)

            decoded_token_info = []
            all_tokens_info = []
            if track:
                full_mask_index = (x == mask_id)
                for j in range(full_mask_index.shape[0]):
                    for idx in torch.where(full_mask_index[j])[0]:
                        all_tokens_info.append({
                            "position": int(idx),
                            "token_id": int(x0[j, idx].item()),
                            "confidence": round(float(curr_conf[j, idx].item()), 4),
                            "entropy": round(float(token_entropy[j, idx].item()), 6),
                        })

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
            for j in range(curr_conf.shape[0]):
                curr_conf[:, input_ids_original.shape[1] + (num_block + 1) * block_length:] = -np.inf
                confidence = torch.where(mask_index, curr_conf, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                _, selected_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, step].item())
                transfer_index[j, selected_indices] = True
                if track:
                    for idx in selected_indices:
                        token_id = int(x0[j, idx].item())
                        decoded_token_info.append({
                            "position": int(idx),
                            "token_id": token_id,
                            "decoded_token": tokenizer.decode([token_id], skip_special_tokens=False),
                            "confidence": round(float(curr_conf[j, idx].item()), 4),
                            "entropy": round(float(token_entropy[j, idx].item()), 6),
                        })

            if track:
                remain_block_masked = mask_index & (~transfer_index)
                total_entropy_excl = token_entropy[remain_block_masked].sum()

            x[transfer_index] = x0[transfer_index]

            if track:
                decoded_text = tokenizer.batch_decode(
                    x[:, input_ids_original.shape[1]:], skip_special_tokens=True
                )[0]
                all_step_outputs.append({
                    "step": used_steps + 1,
                    "decoded_text": decoded_text,
                    "decoded_tokens_num": len(decoded_token_info),
                    "decoded_tokens": decoded_token_info,
                    "all_tokens": all_tokens_info,
                    "information": {
                        "current_uncertainty": round(float(total_entropy_before_block.item()), 6),
                        "current_uncertainty_excl_decoded": round(float(total_entropy_excl.item()), 6),
                        "next_uncertainty": None,
                    },
                })
                pending_idx = used_steps

            del logits
            del p_metrics
            if temperature > 0:
                del logits_sampling
                del p_sample
            used_steps += 1

    if track and pending_idx is not None:
        all_step_outputs[pending_idx]["information"]["next_uncertainty"] = 0.0

    if step_save_dir:
        all_steps_path = os.path.join(step_save_dir, f"all_steps_{example_idx}.json")
        with open(all_steps_path, "w") as f:
            json.dump(all_step_outputs, f, indent=2)

    return x, used_steps
