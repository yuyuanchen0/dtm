import json
import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from lightning_fabric.plugins import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .data_utils import get_countdown_questions, get_sudoku_questions, set_random_seed
from .lightning_module import DTMModule
from .reward_func import countdown_reward_func, sudoku_reward_func


class _PickleSafeCheckpointIO(TorchCheckpointIO):
    """Lightning's default loader passes weights_only=True; LoRA + custom state dict needs the full pickle."""

    def load_checkpoint(self, path, map_location=None, weights_only=None):
        return super().load_checkpoint(path, map_location=map_location, weights_only=False)


def _build_dataset(cfg: DictConfig):
    if cfg.dataset == "countdown":
        train_set = get_countdown_questions("train")
        cur_path = os.path.dirname(os.path.abspath(__file__))
        test_jsonl = os.path.normpath(os.path.join(cur_path, "..", "data", "countdown_cd3_test.jsonl"))
        with open(test_jsonl, "r") as f:
            validation_set = [json.loads(line) for line in f]
        reward_funcs = [countdown_reward_func]
    elif cfg.dataset == "sudoku":
        train_set = get_sudoku_questions(few_shot=cfg.few_shot)
        validation_set = []  # eval split is reserved for the standalone eval/eval.py script
        reward_funcs = [sudoku_reward_func]
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset!r}; expected one of {{countdown, sudoku}}")

    train_set = train_set.shuffle(seed=cfg.seed)
    train_set = train_set.select(range(0, max(0, len(train_set) - 500)))
    return train_set, validation_set, reward_funcs


def train(cfg: DictConfig):
    set_random_seed(cfg.seed)

    if "wandb" in cfg and rank_zero_only.rank == 0:
        init_kwargs = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if cfg.get("resume_path"):
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)
        wandb_logger = WandbLogger(project=wandb.run.project, name=wandb.run.name, log_model=False)
    else:
        wandb_logger = None

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_set, validation_set, reward_funcs = _build_dataset(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModel.from_pretrained(
        cfg.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": torch.cuda.current_device()},
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.use_cache = False

    model = DTMModule(
        base_model=base_model,
        tokenizer=tokenizer,
        train_set=train_set,
        validation_set=validation_set,
        reward_funcs=reward_funcs,
        **cfg,
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=cfg.checkpoint_dir,
        save_top_k=-1,
        every_n_train_steps=cfg.ckpt_freq,
        save_on_train_epoch_end=False,
        filename="checkpoint-{ckpt_a:.2f}-{ckpt_counter}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        num_nodes=cfg.nodes,
        accelerator="gpu",
        devices=cfg.devices,
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        default_root_dir=cfg.checkpoint_dir,
        enable_progress_bar=False,
        max_steps=-1,
        max_epochs=10**12,
        plugins=[_PickleSafeCheckpointIO()],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    dummy_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(1)), batch_size=1
    )
    resume_path = cfg.get("resume_path") or None
    if resume_path:
        trainer.fit(model, train_dataloaders=dummy_loader, ckpt_path=resume_path)
    else:
        trainer.fit(model, train_dataloaders=dummy_loader)


@hydra.main(config_path="../configs", config_name="countdown", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
