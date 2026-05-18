"""Diffusion sampler for evaluation. Adapted from https://github.com/ML-GSAI/LLaDA."""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm


def _add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    out = base.expand(-1, steps).clone()
    if rem.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        out[indices.unsqueeze(0) < rem] += 1
    return out.to(torch.int64)


@torch.no_grad()
def generate(
    model, prompt, tokenizer, steps=64, gen_length=128, block_length=32,
    temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=126336,
):
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
            dtype=torch.long, device=prompt.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        for nb in tqdm(range(num_blocks), disable=not rank0):
            start_idx = prompt.shape[1] + nb * block_length
            end_idx = prompt.shape[1] + (nb + 1) * block_length
            block_mask = x[:, start_idx:end_idx] == mask_id
            num_transfer = _get_num_transfer_tokens(block_mask, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = _add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, end_idx:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
                for j in range(confidence.shape[0]):
                    n = num_transfer[j, i].item()
                    if n > 0:
                        _, sel = torch.topk(confidence[j], k=n)
                        x[j, sel] = x0[j, sel]
        return x
