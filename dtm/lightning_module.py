"""Discrete Tilt Matching (DTM) Lightning module.

The training step:
  1. Maintain a replay buffer of fully-generated rollouts x_1 with rewards r(x_1).
  2. Sample partially-masked x_t from the buffered x_1 via the SAR-aligned interpolant.
  3. Run two forward passes (frozen base π_a / trainable student π_θ) and minimize the
     c-DTM cross-entropy loss.
  4. Anneal the tilt parameter a in steps of h every steps_per_h optimizer updates;
     at each phase boundary copy the EMA snapshot of the model into the teacher.
"""

import copy
import math
import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from .data_utils import SYSTEM_PROMPT, _parse_countdown_record


class DTMModule(pl.LightningModule):
    """Discrete Tilt Matching trainer (c-DTM with SAR-aligned interpolant)."""

    LLADA_MASK_ID = 126336

    def __init__(self, base_model, tokenizer, train_set, validation_set, reward_funcs, **cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(
            ignore=["base_model", "tokenizer", "train_set", "validation_set", "reward_funcs"],
            logger=False,
        )
        self.tokenizer = tokenizer

        # --- Wrap base model with two LoRA adapters (student is trained, teacher is the frozen π_a) ---
        peft_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=self.hparams.peft_task_type,
            lora_dropout=self.hparams.lora_dropout,
        )
        peft_wrapped = get_peft_model(base_model, peft_config, adapter_name="student")
        peft_wrapped.add_adapter("teacher", peft_config)
        student_state = get_peft_model_state_dict(peft_wrapped, adapter_name="student")
        set_peft_model_state_dict(peft_wrapped, student_state, adapter_name="teacher")

        for name, param in peft_wrapped.named_parameters():
            if ".teacher" in name:
                param.requires_grad = False
        peft_wrapped.set_adapter("student")
        self.model = peft_wrapped

        # --- Replay buffer / dataset state ---
        self.train_set = train_set
        self.train_set_len = len(train_set)
        self.validation_set = validation_set
        self.reward_funcs = reward_funcs

        self.curr_prompt_counter = 0
        self.buffer = None
        self.buffer_rewards = None
        self._rebuild_buffer_next_phase = False
        self.num_buffer_prompts = self.hparams.tm.num_buffer_prompts
        self.comps_per_prompt = self.hparams.tm.num_completions_per_prompt
        self.buffer_update_counter = 0

        self._grad_accum_counter = 0
        self._step_counter = 0
        self._eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if self._eos_id is None:
            raise ValueError("tokenizer.eos_token_id must be set to exclude post-EOS positions.")

        # --- DTM hyperparameters ---
        self.a = 0.0
        self.h = float(self.hparams.tm.h)
        self.steps_per_h = int(self.hparams.tm.steps_per_h)
        self.a_end = float(self.hparams.tm.a_end)
        self.cv = float(self.hparams.tm.control_variate)  # constant c-DTM control variate (Prop 3.3, c=1)

        self.lr = float(self.hparams.learning_rate)
        self.lr_scheduler_type = self.hparams.lr_scheduler_type
        self.lr_decay_ratio = float(self.hparams.lr_decay_ratio)
        self.lr_warmup_ratio = float(getattr(self.hparams, "lr_warmup_ratio", 0.0))
        self.lr_min = float(getattr(self.hparams, "lr_min", 0.0))
        self._tm_sched_state = None

        # EMA over the student adapter weights, refreshed once per h-phase
        self.use_ema = bool(getattr(self.hparams.tm, "use_ema", True))
        self.ema_decay = float(getattr(self.hparams.tm, "ema_decay", 0.95))
        self._ema_shadow = None

        self.dict_for_logs = {}
        self.ckpt_counter = 0
        self._checkpoint_save_mode = "mid_phase_eval_only"
        self._phase_boundary_checkpoint_payload = None
        self._last_phase_boundary_ckpt_step = None
        self._loaded_eval_only_checkpoint = False

        # micro-step metric accumulators for logging
        self._micro_log_sums = {}
        self._micro_log_counts = {}

    # ---------------------------------------------------------------
    # Adapter / state-dict plumbing
    # ---------------------------------------------------------------

    @contextmanager
    def _use_adapter(self, adapter_name):
        prev = self.model.active_adapter
        self.model.set_adapter(adapter_name)
        try:
            yield
        finally:
            self.model.set_adapter(prev)

    def _clone_state_value(self, value, keep_vars=False, to_cpu=False):
        if not torch.is_tensor(value):
            return value
        out = value if keep_vars else value.detach().clone()
        return out.to("cpu") if to_cpu else out

    def _adapter_state(self, adapter_name, keep_vars=False, to_cpu=False):
        state = get_peft_model_state_dict(self.model, adapter_name=adapter_name)
        return OrderedDict(
            (key, self._clone_state_value(value, keep_vars=keep_vars, to_cpu=to_cpu))
            for key, value in state.items()
        )

    def _add_prefixed_adapter_state(self, destination, prefix, adapter_state, keep_vars=False):
        for key, value in adapter_state.items():
            destination[f"{prefix}.{key}"] = self._clone_state_value(value, keep_vars=keep_vars, to_cpu=True)

    def _state_to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu")
        if isinstance(obj, dict):
            return {key: self._state_to_cpu(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._state_to_cpu(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._state_to_cpu(value) for value in obj)
        return copy.deepcopy(obj)

    def state_dict(self, destination=None, keep_vars=False):
        destination = OrderedDict() if destination is None else destination
        if self._checkpoint_save_mode == "phase_boundary_resume":
            payload = self._phase_boundary_checkpoint_payload
            if payload is None:
                raise RuntimeError("Missing phase-boundary checkpoint payload.")
            next_teacher = payload["next_phase_teacher_adapter"]
            # Match the reference phase boundary: EMA becomes both student and teacher.
            self._add_prefixed_adapter_state(destination, "model_adapter", next_teacher, keep_vars=keep_vars)
            self._add_prefixed_adapter_state(destination, "base_adapter", next_teacher, keep_vars=keep_vars)
            return destination

        student = self._adapter_state("student", keep_vars=keep_vars, to_cpu=True)
        self._add_prefixed_adapter_state(destination, "model_adapter", student, keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict: bool = True):
        expected_student = set(get_peft_model_state_dict(self.model, adapter_name="student").keys())
        expected_teacher = set(get_peft_model_state_dict(self.model, adapter_name="teacher").keys())
        student_state, teacher_state, unexpected_keys = OrderedDict(), OrderedDict(), []
        for key, value in state_dict.items():
            if key.startswith("model_adapter."):
                student_state[key[len("model_adapter."):]] = value.to(self.model.device)
            elif key.startswith("base_adapter."):
                teacher_state[key[len("base_adapter."):]] = value.to(self.model.device)
            else:
                unexpected_keys.append(key)
        if student_state:
            set_peft_model_state_dict(self.model, student_state, adapter_name="student")
        if teacher_state:
            set_peft_model_state_dict(self.model, teacher_state, adapter_name="teacher")
        missing = [f"model_adapter.{k}" for k in (expected_student - set(student_state.keys()))]
        if teacher_state:
            missing += [f"base_adapter.{k}" for k in (expected_teacher - set(teacher_state.keys()))]
        if strict and (missing or unexpected_keys):
            raise RuntimeError(f"missing keys {missing}; unexpected {unexpected_keys}")
        return {"missing_keys": missing, "unexpected_keys": unexpected_keys}

    # ---------------------------------------------------------------
    # Lightning lifecycle
    # ---------------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            params,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )

    def on_train_start(self):
        super().on_train_start()
        if self._loaded_eval_only_checkpoint:
            raise RuntimeError(
                "This is a mid-phase eval-only checkpoint. Resume training from a "
                "phase_boundary_resume checkpoint instead."
            )
        self._start_step = getattr(self, "global_step", 0)

        global_world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        logical_world_size = min(global_world_size, self.hparams.world_size)
        logical_rank = global_rank % logical_world_size
        self.g = torch.Generator(device=self.device)
        self.g.manual_seed(12345 + logical_rank)

        self.tm_opt = self.optimizers()
        for g in self.tm_opt.param_groups:
            g["lr"] = self.lr
        self._init_tm_scheduler()

        self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
        self._reset_ema()

    def on_train_batch_start(self, batch, batch_idx):
        if getattr(self, "_rebuild_buffer_next_phase", False):
            self._update_buffer(self.model, self.num_buffer_prompts, self.comps_per_prompt)
            self._rebuild_buffer_next_phase = False

    # ---------------------------------------------------------------
    # Micro-step metric averaging (within one optimizer step)
    # ---------------------------------------------------------------

    def _reset_micro_log_accum(self):
        self._micro_log_sums = {}
        self._micro_log_counts = {}

    def _accumulate_micro_log_dict(self, log_dict):
        for k, v in log_dict.items():
            val = v.detach().item() if isinstance(v, torch.Tensor) else v
            self._micro_log_sums[k] = self._micro_log_sums.get(k, 0.0) + float(val)
            self._micro_log_counts[k] = self._micro_log_counts.get(k, 0) + 1

    def _finalize_micro_log_dict(self):
        return {k: s / self._micro_log_counts.get(k, 1) for k, s in self._micro_log_sums.items()}

    # ---------------------------------------------------------------
    # EMA on the student adapter
    # ---------------------------------------------------------------

    def _student_param_iter(self):
        state = get_peft_model_state_dict(self.model, adapter_name="student")
        for key, value in state.items():
            yield key, value

    def _reset_ema(self):
        if not self.use_ema:
            self._ema_shadow = None
            return
        self._ema_shadow = {key: param.detach().clone() for key, param in self._student_param_iter()}

    @torch.no_grad()
    def _sync_phase_models_from_ema(self):
        student_state = get_peft_model_state_dict(self.model, adapter_name="student")
        if self.use_ema:
            if self._ema_shadow is None:
                raise RuntimeError("EMA shadow is missing at phase boundary.")
            missing = sorted(set(student_state.keys()) - set(self._ema_shadow.keys()))
            if missing:
                raise RuntimeError(f"EMA shadow missing keys: {missing}")
            set_peft_model_state_dict(self.model, self._ema_shadow, adapter_name="student")
            set_peft_model_state_dict(self.model, self._ema_shadow, adapter_name="teacher")
        else:
            set_peft_model_state_dict(self.model, student_state, adapter_name="teacher")
        for name, p in self.model.named_parameters():
            if ".teacher" in name:
                p.requires_grad_(False)
        self.model.set_adapter("student")

    @torch.no_grad()
    def _update_ema(self):
        if self._ema_shadow is None:
            self._reset_ema()
            return
        decay = float(self.ema_decay)
        for key, param in self._student_param_iter():
            if key not in self._ema_shadow:
                self._ema_shadow[key] = param.detach().clone()
            else:
                self._ema_shadow[key].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def _prompt_loader_state(self):
        return {
            "curr_prompt_counter": int(self.curr_prompt_counter),
            "buffer_update_counter": int(self.buffer_update_counter),
            "rebuild_buffer_next_phase": bool(getattr(self, "_rebuild_buffer_next_phase", False)),
            "last_prompt_indices": list(getattr(self, "_last_prompt_indices", [])),
        }

    def _restore_prompt_loader_state(self, state):
        if not state:
            return
        self.curr_prompt_counter = int(state.get("curr_prompt_counter", self.curr_prompt_counter))
        self.buffer_update_counter = int(state.get("buffer_update_counter", self.buffer_update_counter))
        self._rebuild_buffer_next_phase = bool(
            state.get("rebuild_buffer_next_phase", getattr(self, "_rebuild_buffer_next_phase", False))
        )
        self._last_prompt_indices = list(state.get("last_prompt_indices", []))

    def _capture_phase_boundary_checkpoint_payload(self):
        student = self._adapter_state("student", to_cpu=True)
        if self.use_ema:
            if self._ema_shadow is None:
                raise RuntimeError("EMA shadow is missing at phase boundary.")
            next_teacher = OrderedDict(
                (key, self._clone_state_value(value, to_cpu=True))
                for key, value in self._ema_shadow.items()
            )
        else:
            next_teacher = copy.deepcopy(student)
        prompt_loader_state = self._prompt_loader_state()
        prompt_loader_state["rebuild_buffer_next_phase"] = True
        return {
            "pre_phase_student_adapter": student,
            "next_phase_teacher_adapter": next_teacher,
            "prompt_loader_state": prompt_loader_state,
        }

    def _phase_boundary_checkpoint_path(self):
        filename = f"phase-boundary-a{float(self.a):.2f}-step{int(self.global_step)}.ckpt"
        return os.path.join(self.hparams.checkpoint_dir, filename)

    def _save_phase_boundary_checkpoint(self):
        if self._phase_boundary_checkpoint_payload is None:
            return
        step = int(self.global_step)
        if self._last_phase_boundary_ckpt_step == step:
            return
        path = self._phase_boundary_checkpoint_path()
        previous_mode = self._checkpoint_save_mode
        self._checkpoint_save_mode = "phase_boundary_resume"
        try:
            self.trainer.save_checkpoint(path)
            self._last_phase_boundary_ckpt_step = step
        finally:
            self._checkpoint_save_mode = previous_mode
            self._phase_boundary_checkpoint_payload = None

    # ---------------------------------------------------------------
    # Training step (manual gradient accumulation; DDP no_sync between micro-steps)
    # ---------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        opt = self.tm_opt
        accum = int(self.hparams.tm.grad_accum_steps)

        # At the start of an accumulation window, sample flattened prompt/completion indices.
        if (self._grad_accum_counter % accum) == 0:
            total_needed = self.hparams.tm.num_batch_prompts * accum
            self._accum_prompts_idx = torch.randperm(
                self.buffer.shape[0] * self.comps_per_prompt,
                device=self.device,
                generator=self.g,
            )[:total_needed]
            self._reset_micro_log_accum()

        loss, micro_log_dict = self._tm_step()
        self._accumulate_micro_log_dict(micro_log_dict)
        loss_scaled = loss / float(accum)

        self._grad_accum_counter += 1
        is_update_step = (self._grad_accum_counter % accum) == 0
        if not is_update_step:
            with self.trainer.model.no_sync():
                self.manual_backward(loss_scaled)
            self.dict_for_logs = {}
            return
        self.manual_backward(loss_scaled)

        # Step the LR schedule, clip grads, and update.
        self._step_tm_scheduler()
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_before = clip_grad_norm_(params, self.hparams.max_grad_norm).item()
        grad_norm_after = clip_grad_norm_(params, float("inf")).item()
        grad_clipped = float(grad_norm_before > self.hparams.max_grad_norm + 1e-6)

        opt.step()
        opt.zero_grad(set_to_none=True)
        if self.use_ema:
            self._update_ema()

        self.dict_for_logs = self._finalize_micro_log_dict()
        self.dict_for_logs["train/lr"] = opt.param_groups[0]["lr"]
        self.dict_for_logs["grads/grad_norm_before"] = grad_norm_before
        self.dict_for_logs["grads/grad_norm_after"] = grad_norm_after
        self.dict_for_logs["grads/grad_clipped"] = grad_clipped

        # Phase boundary: anneal a, sync adapters from the EMA snapshot, clear Adam
        # moments, then reset the LR schedule.
        if (self.global_step - self._start_step) % self.steps_per_h == 0:
            self.a += self.h
            if self.a + self.h > self.a_end:
                self.h = self.a_end - self.a
            self._phase_boundary_checkpoint_payload = self._capture_phase_boundary_checkpoint_payload()
            with torch.no_grad():
                self._sync_phase_models_from_ema()
            print(f"Phase boundary: a={self.a:.4f} at global step {self.global_step}")
            if self.a >= self.a_end:
                print(f"Reached final a={self.a_end:.2f}. Training stopped.", flush=True)
                self.trainer.should_stop = True
            if getattr(self, "tm_opt", None) is not None:
                self.tm_opt.state.clear()
            for g in opt.param_groups:
                g["lr"] = self.lr
            self._init_tm_scheduler()
            self._reset_ema()

        self.log("ckpt_a", self.a, on_step=True, on_epoch=False, sync_dist=True)

    # ---------------------------------------------------------------
    # The c-DTM step
    # ---------------------------------------------------------------

    def _tm_step(self):
        num_buffer_prompts, comps_per_prompt, L = self.buffer.shape
        num_batch_prompts = self.hparams.tm.num_batch_prompts
        gen_length = self.hparams.max_completion_length

        # ---- Draw a mixed batch of x_1's from the flattened prompt/completion buffer ----
        B = num_batch_prompts
        start = (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) * B
        prompts_idx = self._accum_prompts_idx[start:start + B]
        x1s = self.buffer.reshape(-1, L)[prompts_idx]
        rwds = self.buffer_rewards.reshape(-1, self.buffer_rewards.shape[-1])[prompts_idx]

        # Aggregate (possibly multiple) reward functions into a single scalar reward.
        weights = torch.ones(rwds.shape[1], device=self.device, dtype=rwds.dtype)
        rwd = torch.nansum(rwds * weights.unsqueeze(0), dim=1)  # [B]
        correct_frac = torch.isclose(
            rwd, self.hparams.max_rwd * torch.ones_like(rwd), atol=1e-6, rtol=0.0
        ).float().mean()

        # ---- Build x_t via the SAR-aligned interpolant ----
        num_to_mask = torch.randint(low=1, high=gen_length + 1, size=(x1s.shape[0],), device=self.device)
        itpl_block_len = int(getattr(self.hparams.tm, "itpl_block_length", self.hparams.block_length))
        completion_ids = x1s[:, -gen_length:]
        xts, mask_indices, active_block_mask = self._build_interpolant(x1s, num_to_mask, itpl_block_len)
        # Loss-weighted positions: the SAR-aligned active block. By construction every row has
        # at least one masked active-block position (`remainder >= 1` in `_build_interpolant`),
        # so the row-wise normaliser is always positive.
        use_sar = bool(getattr(self.hparams.tm, "use_sar_active_block_norm", False))
        aux_mask = active_block_mask if use_sar else mask_indices
        loss_weights = aux_mask.to(torch.float32)
        # ---- Two forward passes (frozen π_a / trainable π_θ) ----
        with torch.no_grad(), self._use_adapter("teacher"):
            self.model.eval()
            old_logits = self._new_forward(self.model, xts, gen_length)
        V = old_logits.shape[-1]
        x1_equals_v = F.one_hot(completion_ids.long(), num_classes=V)
        with self._use_adapter("student"):
            self.model.train()
            curr_logits = self._new_forward(self.model, xts, gen_length)

        old_probs = F.softmax(old_logits, dim=-1)

        # ---- c-DTM weighted target ----
        hr = self.h * rwd  # [B]
        c = float(self.cv)
        target = c * old_probs + x1_equals_v * (1.0 - c + torch.expm1(hr)).view(-1, 1, 1)

        per_position_losses = -(target * F.log_softmax(curr_logits, dim=-1)).sum(dim=-1)  # [B, gen_length]
        loss_weights = loss_weights.to(per_position_losses.dtype)
        per_row_losses = (per_position_losses * loss_weights).sum(dim=1)
        loss = (per_row_losses / loss_weights.sum(dim=1)).mean()

        log_dict = {
            "train/loss": loss,
            "train/a": self.a,
            "train/h": self.h,
            "train/effective_gen_len": self._compute_effective_gen_lengths(completion_ids).float().mean(),
            "train/rwd_mean": rwd.mean(),
            "train/rwd_std": rwd.std(),
            "train/correct_frac": correct_frac,
        }
        return loss, log_dict

    # ---------------------------------------------------------------
    # Logging / checkpointing
    # ---------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (self._grad_accum_counter % self.hparams.tm.grad_accum_steps) == 0:
            if (self.global_step - self._start_step + 1) % self.hparams.ckpt_freq == 0:
                self.ckpt_counter += 1
                self.log("ckpt_counter", self.ckpt_counter, on_step=True, on_epoch=False, sync_dist=True)
            # Eval the student on validation set every steps_per_h optimizer steps (phase boundary).
            if (self.global_step - self._start_step) % self.steps_per_h == 0:
                self._save_phase_boundary_checkpoint()
                if self.validation_set:
                    self._log_eval_accuracy(self.model)
                self._rebuild_buffer_next_phase = True
            elif (self.global_step - self._start_step) % self.hparams.tm.buffer_refresh_steps == 0:
                self._update_buffer(self.model, self.hparams.tm.num_buffer_refresh, self.comps_per_prompt)

        if not self.dict_for_logs or (self.global_step - self._start_step - 1) % self.hparams.metrics_log_every != 0:
            return
        self.log_dict(self.dict_for_logs, on_step=True, on_epoch=False, sync_dist=True)
        self.dict_for_logs = {}
        self._step_counter += 1

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["dtm_checkpoint_type"] = self._checkpoint_save_mode
        checkpoint["tilt_a"] = self.a
        prompt_loader_state = self._prompt_loader_state()
        checkpoint["prompt_loader_state"] = prompt_loader_state
        checkpoint["prompt_counter"] = prompt_loader_state["curr_prompt_counter"]
        checkpoint["tm_sched_state"] = copy.deepcopy(getattr(self, "_tm_sched_state", None))
        checkpoint["step_counter"] = self._step_counter + 1
        checkpoint["ckpt_counter"] = self.ckpt_counter
        if self._checkpoint_save_mode == "phase_boundary_resume":
            payload = self._phase_boundary_checkpoint_payload
            if payload is None:
                raise RuntimeError("Missing phase-boundary checkpoint payload.")
            checkpoint["prompt_loader_state"] = copy.deepcopy(payload["prompt_loader_state"])
            checkpoint["prompt_counter"] = payload["prompt_loader_state"]["curr_prompt_counter"]
            checkpoint["phase_boundary_pre_student_adapter"] = copy.deepcopy(payload["pre_phase_student_adapter"])
            checkpoint["phase_boundary_next_teacher_adapter"] = copy.deepcopy(payload["next_phase_teacher_adapter"])
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)
            checkpoint["phase_boundary_resume_note"] = (
                "state_dict model_adapter/base_adapter are both the EMA snapshot used to start "
                "the next phase; optimizer state is intentionally omitted."
            )
            return

        checkpoint.pop("optimizer_states", None)
        checkpoint.pop("lr_schedulers", None)

    def on_load_checkpoint(self, checkpoint: dict):
        self._loaded_eval_only_checkpoint = checkpoint.get("dtm_checkpoint_type") == "mid_phase_eval_only"
        self.a = float(checkpoint.get("tilt_a", 0.0))
        self.curr_prompt_counter = int(checkpoint.get("prompt_counter", 0))
        self._restore_prompt_loader_state(checkpoint.get("prompt_loader_state"))
        self._tm_sched_state = checkpoint.get("tm_sched_state", None)
        self._step_counter = int(checkpoint.get("step_counter", 0))
        self.ckpt_counter = int(checkpoint.get("ckpt_counter", 0))

    # ---------------------------------------------------------------
    # Replay buffer: generate new rollouts via diffusion and score them
    # ---------------------------------------------------------------

    def _prepare_prompts(self, num_distinct_prompts, num_completions_per_prompt):
        global_world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        logical_world_size = min(global_world_size, self.hparams.world_size)
        logical_rank = global_rank % logical_world_size

        indices = []
        for offset in range(num_distinct_prompts):
            idx = (self.curr_prompt_counter + offset * logical_world_size + logical_rank) % self.train_set_len
            indices.append(idx)
        self.curr_prompt_counter = (self.curr_prompt_counter + num_distinct_prompts * logical_world_size) % self.train_set_len
        self._last_prompt_indices = indices

        structured_prompts = [self.train_set[i]["prompt"] for i in indices]
        prompts_text = []
        for sp in structured_prompts:
            if isinstance(sp, str):
                prompts_text.append(sp)
            elif isinstance(sp, list):
                prompts_text.append(self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True))
            else:
                raise TypeError(f"Unsupported prompt type {type(sp)}")

        tokenized = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_prompt_length,
            padding_side="left",
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        return input_ids.repeat_interleave(num_completions_per_prompt, dim=0), None

    def _update_buffer(self, model, num_buffer_updates, num_completions_per_prompt):
        device = self.device
        prev_adapter = model.active_adapter
        model.set_adapter("teacher")
        model.eval()
        t0 = datetime.now()

        if num_buffer_updates == self.num_buffer_prompts:
            update_rows = list(range(self.num_buffer_prompts))
            self.buffer_update_counter = 0
            self.buffer = None
            self.buffer_rewards = None
        else:
            update_rows = [
                (self.buffer_update_counter + u) % self.num_buffer_prompts for u in range(num_buffer_updates)
            ]
            self.buffer_update_counter = (self.buffer_update_counter + num_buffer_updates) % self.num_buffer_prompts

        prompt_ids, _ = self._prepare_prompts(num_buffer_updates, num_completions_per_prompt)
        total_batch, prompt_len = prompt_ids.shape
        chunk_size = max(1, min(self.hparams.tm.buffer_chunk_size, total_batch))
        gen_length = self.hparams.max_completion_length
        seq_len = prompt_len + gen_length

        prompt_completion_ids = torch.empty((total_batch, seq_len), device=device, dtype=prompt_ids.dtype)
        for s in range(0, total_batch, chunk_size):
            e = min(s + chunk_size, total_batch)
            with torch.no_grad():
                prompt_completion_ids[s:e].copy_(
                    self._generate(
                        model=model,
                        prompt=prompt_ids[s:e],
                        steps=self.hparams.diffusion_steps,
                        gen_length=gen_length,
                        block_length=self.hparams.block_length,
                        temperature=self.hparams.sampling_temperature,
                        cfg_scale=self.hparams.cfg_scale,
                        remasking=self.hparams.remasking_strategy,
                    )
                )

        new_buffer_block = prompt_completion_ids.view(num_buffer_updates, -1, seq_len)
        if self.buffer is None:
            self.buffer = new_buffer_block
        else:
            self.buffer[update_rows, :, :] = new_buffer_block

        # ---- Score completions ----
        completion_ids = prompt_completion_ids[:, prompt_len:]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        data_keys = [k for k in self.train_set[0].keys() if k != "prompt"]
        prompts_for_rewards = []
        reward_kwargs = {key: [] for key in data_keys}
        for row_idx in self._last_prompt_indices:
            row = self.train_set[row_idx]
            for _ in range(num_completions_per_prompt):
                prompts_for_rewards.append(row["prompt"])
                for key in data_keys:
                    reward_kwargs[key].append(row[key])
        completions_for_rewards = [[{"role": "assistant", "content": t}] for t in completions_text]

        num_funcs = len(self.reward_funcs)
        rewards_per_func = torch.zeros(total_batch, num_funcs, device=device)
        for j, fn in enumerate(self.reward_funcs):
            scores = fn(prompts=prompts_for_rewards, completions=completions_for_rewards, **reward_kwargs)
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32).clamp(min=0.0)

        new_rewards_block = rewards_per_func.view(num_buffer_updates, -1, num_funcs)
        if self.buffer_rewards is None:
            self.buffer_rewards = new_rewards_block
        else:
            self.buffer_rewards[update_rows, :, :] = new_rewards_block

        avg_rwd = float(new_rewards_block.mean() * new_rewards_block.shape[-1])
        dt = (datetime.now() - t0).total_seconds()
        if int(getattr(self.trainer, "global_rank", 0)) == 0:
            tag = "build" if num_buffer_updates == self.num_buffer_prompts else "refresh"
            print(f"[buffer-{tag}] avg_reward={avg_rwd:.3f} took {dt:.1f}s", flush=True)

        model.set_adapter(prev_adapter)
        model.train()
        # Free the diffusion-sampling intermediates before training_step double-forward.
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Evaluation accuracy (eval/correct_frac → wandb)
    # ---------------------------------------------------------------

    def _log_eval_accuracy(self, model):
        if not self.validation_set:
            return
        device = self.device
        prev_adapter = model.active_adapter
        model.set_adapter("student")
        model.eval()
        t0 = datetime.now()

        global_world_size = int(getattr(self.trainer, "world_size", 1))
        global_rank = int(getattr(self.trainer, "global_rank", 0))
        total_val = len(self.validation_set)
        per_gpu = total_val // global_world_size
        start_idx = global_rank * per_gpu
        end_idx = total_val if global_rank == global_world_size - 1 else start_idx + per_gpu
        subset = list(range(start_idx, end_idx))
        num_val = len(subset)

        if self.hparams.dataset == "countdown":
            structured_prompts, targets, numbers_list = [], [], []
            for idx in subset:
                v = self.validation_set[idx]
                numbers, target = _parse_countdown_record(v)
                content = (
                    f"{SYSTEM_PROMPT}\nUsing only the numbers {numbers}, create an arithmetic expression "
                    f"that evaluates to exactly {target}. You must use all numbers from the list, and each "
                    f"number must be used exactly once. You may use the operations +, -, *, and / as needed. "
                    f"After reasoning, provide only your final expression inside <answer></answer> tags "
                    f"without including an equals sign or the target number. For example, if the numbers are "
                    f"[2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
                )
                structured_prompts.append([{"role": "user", "content": content}])
                targets.append(target)
                numbers_list.append(numbers)
            reward_kwargs = {"target": targets, "numbers": numbers_list}
        else:
            structured_prompts = [self.validation_set[i]["prompt"] for i in subset]
            data_keys = [k for k in self.validation_set[0].keys() if k != "prompt"]
            reward_kwargs = {key: [self.validation_set[i][key] for i in subset] for key in data_keys}

        prompts_text = [
            self.tokenizer.apply_chat_template(sp, tokenize=False, add_generation_prompt=True)
            for sp in structured_prompts
        ]
        tokenized = self.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = tokenized["input_ids"].to(device)
        total_batch, prompt_len = prompt_ids.shape

        chunk_size = max(1, min(self.hparams.tm.buffer_chunk_size, total_batch) // 4)
        gen_length = self.hparams.max_completion_length
        prompt_completion_ids = torch.empty(
            (total_batch, prompt_len + gen_length), device=device, dtype=prompt_ids.dtype
        )
        for s in range(0, total_batch, chunk_size):
            e = min(s + chunk_size, total_batch)
            with torch.no_grad():
                prompt_completion_ids[s:e].copy_(
                    self._generate(
                        model=model,
                        prompt=prompt_ids[s:e],
                        steps=self.hparams.diffusion_steps,
                        gen_length=gen_length,
                        block_length=self.hparams.block_length,
                        temperature=0.0,
                        cfg_scale=self.hparams.cfg_scale,
                        remasking="low_confidence",
                    )
                )

        completions_text = self.tokenizer.batch_decode(prompt_completion_ids[:, prompt_len:], skip_special_tokens=True)
        completions_for_rewards = [[{"role": "assistant", "content": t}] for t in completions_text]
        rewards_per_func = torch.zeros(total_batch, len(self.reward_funcs), device=device)
        for j, fn in enumerate(self.reward_funcs):
            scores = fn(prompts=structured_prompts, completions=completions_for_rewards, **reward_kwargs)
            rewards_per_func[:, j] = torch.tensor(scores, device=device, dtype=torch.float32)

        correct = torch.isclose(
            rewards_per_func, self.hparams.max_rwd * torch.ones_like(rewards_per_func), atol=1e-3, rtol=0.0
        ).float().sum().item()
        total_rwd = rewards_per_func.sum().item()

        local = torch.tensor([correct, total_rwd, num_val], device=device, dtype=torch.float32)
        gathered = self.all_gather(local)
        denom = gathered[:, 2].sum().clamp_min(1.0)
        eval_metrics = {
            "eval/correct_frac": (gathered[:, 0].sum() / denom).item(),
            "eval/avg_rwd": (gathered[:, 1].sum() / denom).item(),
        }
        if global_rank == 0:
            dt = (datetime.now() - t0).total_seconds()
            print(
                f"[eval@step {self.global_step}] correct_frac={eval_metrics['eval/correct_frac']:.4f} "
                f"avg_rwd={eval_metrics['eval/avg_rwd']:.4f}  ({dt:.1f}s)",
                flush=True,
            )
        self.dict_for_logs.update(eval_metrics)

        model.set_adapter(prev_adapter)
        model.train()

    # ---------------------------------------------------------------
    # LR scheduler (per h-phase: warmup → constant → linear decay)
    # ---------------------------------------------------------------

    def _init_tm_scheduler(self):
        opt = self.tm_opt
        self._tm_sched_state = None
        if opt is None or self.lr_scheduler_type in (None, "constant"):
            return
        if self.lr_scheduler_type != "linear":
            raise NotImplementedError("Only linear LR schedule is implemented")
        assert self.lr_warmup_ratio + self.lr_decay_ratio <= 1.0
        total = self.steps_per_h
        warmup = math.floor(self.lr_warmup_ratio * total)
        decay = math.floor(self.lr_decay_ratio * total)
        const = total - warmup - decay
        base_lrs = [pg["lr"] for pg in opt.param_groups]
        scale = self.lr_min / self.lr
        min_lrs = [lr * scale for lr in base_lrs]
        for pg in opt.param_groups:
            pg["lr"] = 0.0
        self._tm_sched_state = {
            "step": 0, "total": total, "warmup": warmup, "const": const, "decay": decay,
            "base_lrs": base_lrs, "min_lrs": min_lrs,
        }

    def _step_tm_scheduler(self):
        st = self._tm_sched_state
        opt = self.tm_opt
        if st is None or opt is None:
            return
        step, total, warmup, const, decay = st["step"], st["total"], st["warmup"], st["const"], st["decay"]
        base_lrs, min_lrs = st["base_lrs"], st["min_lrs"]
        if step >= total:
            for pg, m in zip(opt.param_groups, min_lrs):
                pg["lr"] = float(m)
            return
        if warmup > 0 and step < warmup:
            frac = float(step + 1) / float(warmup)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(b) * frac
        elif step < warmup + const:
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(b)
        elif decay > 0:
            k = step - warmup - const
            frac = 1.0 if decay == 1 else float(k) / float(decay - 1)
            for pg, b, m in zip(opt.param_groups, base_lrs, min_lrs):
                pg["lr"] = float(b + (m - b) * frac)
        else:
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = float(b)
        st["step"] = step + 1

    # ---------------------------------------------------------------
    # Helpers: eos masks, interpolant, attention mask, sampling
    # ---------------------------------------------------------------

    def _compute_effective_gen_lengths(self, completion_ids):
        eos_hits = completion_ids.eq(self._eos_id)
        has_eos = eos_hits.any(dim=1)
        first = eos_hits.to(torch.int64).argmax(dim=1)
        full = torch.full_like(first, completion_ids.shape[1])
        return torch.where(has_eos, first, full)

    def _build_interpolant(self, x1s, num_to_mask, block_size):
        """SAR-compatible interpolant: fully mask the suffix blocks plus a partial active block."""
        device = x1s.device
        B, L = x1s.shape
        prompt_len = self.hparams.max_prompt_length
        gen_len = self.hparams.max_completion_length
        num_blocks = gen_len // block_size

        assert (num_to_mask <= gen_len).all() and (num_to_mask >= 1).all()
        assert L == prompt_len + gen_len
        assert gen_len % block_size == 0

        xts = x1s.clone()
        full_blocks = (num_to_mask - 1) // block_size
        remainder = (num_to_mask - 1) % block_size + 1

        comp_pos = torch.arange(gen_len, device=device)
        block_ids = (comp_pos // block_size).unsqueeze(0).expand(B, -1)
        full_blocks_threshold = (num_blocks - full_blocks).unsqueeze(1)
        full_blocks_to_mask = block_ids >= full_blocks_threshold

        scores = torch.rand(B, block_size, device=device)
        ranks = scores.argsort(dim=1).argsort(dim=1)
        masks_within_block = ranks < remainder.unsqueeze(1)
        partial_block_start = (full_blocks_threshold - 1) * block_size
        idx = partial_block_start + torch.arange(block_size, device=device)
        partial_to_mask = torch.zeros(B, gen_len, dtype=torch.bool, device=device)
        partial_to_mask.scatter_(1, idx, masks_within_block)
        mask_indices = full_blocks_to_mask | partial_to_mask

        completion_region = xts[:, prompt_len:]
        completion_region = torch.where(
            mask_indices, torch.full_like(completion_region, self.LLADA_MASK_ID), completion_region
        )
        xts[:, prompt_len:] = completion_region
        return xts, mask_indices.bool(), partial_to_mask.bool()

    # ---------------------------------------------------------------
    # Diffusion sampling (block-wise low-confidence remasking).
    # Adapted from https://github.com/ML-GSAI/LLaDA.
    # ---------------------------------------------------------------

    def _generate(
        self, model, prompt,
        steps=128, gen_length=128, block_length=128,
        temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
    ):
        with torch.amp.autocast("cuda", enabled=True):
            mask_id = self.LLADA_MASK_ID
            bs = prompt.shape[0]
            dtype = model.dtype
            prompt_len = prompt.shape[1]
            x = torch.full((bs, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device)
            x[:, :prompt_len] = prompt.clone()
            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            steps_per_block = max(1, steps // num_blocks)

            for nb in range(num_blocks):
                start_idx = prompt_len + nb * block_length
                end_idx = prompt_len + (nb + 1) * block_length
                block_mask = x[:, start_idx:end_idx] == mask_id
                num_transfer = self._get_num_transfer_tokens(block_mask, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x[:, prompt_len:] == mask_id
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = self._new_forward(model, x_, gen_length)
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = self._new_forward(model, x, gen_length)

                    logits_with_noise = self._add_gumbel_noise(logits, temperature, dtype)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == "random":
                        x0_p = torch.rand_like(x0, dtype=torch.float32)
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[:, end_idx - prompt_len:] = float("-inf")
                    x0 = torch.where(mask_index, x0, x[:, prompt_len:])
                    confidence = torch.where(mask_index, x0_p, float("-inf"))
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        n = num_transfer[j, i].item()
                        if n > 0:
                            _, sel = torch.topk(confidence[j], k=n)
                            transfer_index[j, sel] = True
                    x[:, prompt_len:][transfer_index] = x0[transfer_index]
            return x

    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        rem = mask_num % steps
        out = base.expand(-1, steps).clone()
        if rem.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            out[indices.unsqueeze(0) < rem] += 1
        return out.to(torch.int64)

    @staticmethod
    def _add_gumbel_noise(logits, temperature, dtype):
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        return logits.exp() / ((-torch.log(noise)) ** temperature)

    # ---------------------------------------------------------------
    # LLaDA forward path (compute logits only on the gen-length suffix)
    # ---------------------------------------------------------------

    def _unwrap_llada_core(self, m):
        assert isinstance(m, PeftModelForCausalLM)
        lm = m.base_model
        core = getattr(lm, "model", None)
        if core is None or not hasattr(core.base_model, "transformer"):
            raise ValueError("Expected a LLaDA HF model with .model.transformer")
        return core.base_model

    def _llada_hidden_no_logits(self, model, input_ids):
        core = self._unwrap_llada_core(model)
        cfg = core.config
        tfm = core.transformer
        assert not cfg.alibi and cfg.rope, "DTM training requires LLaDA with rope=True, alibi=False"
        x = tfm.wte(input_ids)
        if cfg.input_emb_norm:
            x = x * (cfg.d_model ** 0.5)
        x = tfm.emb_drop(x)

        attention_bias = None  # rely on the model's internal default bidirectional bias

        if cfg.block_group_size == 1:
            from .configuration_llada import ActivationCheckpointingStrategy
            for block_idx, block in enumerate(tfm.blocks):
                strat = core.activation_checkpointing_strategy
                use_ckpt = (
                    strat == ActivationCheckpointingStrategy.whole_layer
                    or (strat == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                    or (strat == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                    or (strat == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
                )
                if use_ckpt:
                    x, _ = core._activation_checkpoint_fn(block, x, attention_bias=attention_bias, layer_past=None, use_cache=False)
                else:
                    x, _ = block(x, attention_bias=attention_bias, layer_past=None, use_cache=False)
        else:
            for block_group in tfm.block_groups:
                x, _ = block_group(x, attention_bias=attention_bias, layers_past=None, use_cache=False)
        return tfm.ln_f(x)

    def _llada_logits_on_suffix(self, model, hidden, gen_len):
        core = self._unwrap_llada_core(model)
        cfg = core.config
        hidden_suffix = hidden[:, -gen_len:, :]
        out_module = model.base_model.get_output_embeddings()
        if isinstance(out_module, torch.nn.Embedding):
            logits = F.linear(hidden_suffix, out_module.weight, None)
        elif isinstance(out_module, torch.nn.Linear):
            logits = out_module(hidden_suffix)
        else:
            raise TypeError(f"Unsupported output embeddings: {type(out_module)}")
        if getattr(cfg, "scale_logits", False):
            logits = logits * (1.0 / math.sqrt(cfg.d_model))
        return logits

    def _new_forward(self, model, x, gen_length):
        hidden = self._llada_hidden_no_logits(model, x)
        return self._llada_logits_on_suffix(model, hidden, gen_length)
