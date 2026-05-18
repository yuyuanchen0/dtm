"""Evaluate a DTM checkpoint on Countdown or Sudoku.

Saves per-rank generation JSONs to args.output_dir; downstream metric extraction
is done by `python -m eval.parser_json`.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from eval.datasets import CountdownDataset, SudokuDataset
from eval.generate import generate

DATASET_MAP = {"countdown": CountdownDataset, "sudoku": SudokuDataset}


def select_lora_adapter_state(state_dict, device):
    adapter_prefix = "model_adapter." if any(k.startswith("model_adapter.") for k in state_dict) else "base_adapter."
    adapter_state = {
        k[len(adapter_prefix):]: v.to(device)
        for k, v in state_dict.items()
        if k.startswith(adapter_prefix)
    }
    if not adapter_state:
        raise ValueError("No LoRA adapter weights found in checkpoint")
    return adapter_state


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(model, tokenizer, dataloader, gen_length, temperature, cfg_scale, steps, block_length, remasking):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        out = generate(
            model, input_ids, tokenizer,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, cfg_scale=cfg_scale, remasking=remasking,
        )
        gens = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        for j in range(len(batch["answers"])):
            all_generations.append({
                "question": batch["questions"][j],
                "prompt_input": batch["prompts"][j],
                "generations": gens[j],
                "ground_truth": batch["answers"][j],
                "nfes": steps,
            })
        total_processed += len(gens)
        wall_times.append(time.time() - t0)

    return {
        "wall_time": sum(wall_times) / max(1, len(wall_times)),
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }


class _NoPadDistributedSampler(DistributedSampler):
    """DistributedSampler with drop_last=False that does NOT pad to multiples of world_size."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = False
        self.total_size = len(self.dataset)
        self.num_samples = len(self.dataset) // self.num_replicas + int(rank < (self.total_size % self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed


def main():
    init_seed(42)
    local_rank = setup_ddp()

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, default="")
    p.add_argument("--dataset", choices=list(DATASET_MAP.keys()), required=True)
    p.add_argument("--few_shot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--remasking", type=str, default="low_confidence")
    p.add_argument("--output_dir", type=str, default="results/")
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--add_reasoning", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_eval", type=int, default=None, help="Number of examples to evaluate; defaults to the task standard.")
    args = p.parse_args()

    if args.seed is not None:
        init_seed(args.seed)

    if args.checkpoint_path:
        ck = args.checkpoint_path.rstrip("/").split("/")
        model_name = f"{ck[-2]}_{ck[-1]}"
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"
    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"
    if args.suffix:
        model_name += f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    rank0_filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_0_generations.json"
    if os.path.exists(filename) or os.path.exists(rank0_filename):
        if dist.get_rank() == 0:
            print(f"Output already exists ({filename}); exiting")
        cleanup_ddp()
        return

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "device_map": {"": local_rank},
    }

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        lora_cfg = LoraConfig(
            r=hp.get("lora_r", 128),
            lora_alpha=hp.get("lora_alpha", 64),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=hp.get("peft_task_type", "CAUSAL_LM"),
            lora_dropout=hp.get("lora_dropout", 0.05),
        )
        peft_model = get_peft_model(model, lora_cfg, adapter_name="teacher")
        sd = ckpt.get("state_dict", ckpt)
        adapter_state = select_lora_adapter_state(sd, peft_model.device)
        set_peft_model_state_dict(peft_model, adapter_state, adapter_name="teacher")
        peft_model.set_adapter("teacher")
        model = peft_model

        if dist.get_world_size() > 1:
            dist.barrier()

    n_eval = {"countdown": 256, "sudoku": 256}
    if args.n_eval is not None:
        n_eval[args.dataset] = args.n_eval
    dataset_cls = DATASET_MAP[args.dataset]

    def _build():
        return dataset_cls(
            tokenizer,
            num_examples=args.few_shot,
            add_reasoning=args.add_reasoning,
            subsample=n_eval[args.dataset],
        )

    if dist.is_initialized() and dist.get_world_size() > 1:
        for r in range(dist.get_world_size()):
            if r == dist.get_rank():
                dataset = _build()
            dist.barrier()
    else:
        dataset = _build()

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=_NoPadDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    metrics = evaluate(
        model, tokenizer, dataloader,
        gen_length=args.gen_length, temperature=args.temperature, cfg_scale=0.0,
        steps=args.diffusion_steps, block_length=args.block_length, remasking=args.remasking,
    )
    with open(filename, "w") as f:
        json.dump(
            {
                "generations": metrics["generations"],
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "total_processed": metrics["total_processed"],
                },
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
            },
            f,
            indent=2,
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
