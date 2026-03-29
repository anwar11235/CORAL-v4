"""CORAL v4 — Training loop with deep supervision.

Training structure:
  For each batch:
    z1 = adapter.encode(inputs)
    z_states = [z1, z2_init, ...]

    For segment k in 0..K_max-1:
      z_states, outputs = coral_core(z_states, segment_k)
      logits = adapter.decode(z_states[0])
      segment_loss = loss_fn(logits, labels, outputs)
      total_loss += segment_loss
      z_states = [z.detach() for z in z_states]   ← deep supervision boundary
      if should_halt: break

    total_loss.backward()
    clip_grad_norm_(params, 1.0)
    optimizer.step()

The inner loop (backbone applications within a segment) stays in the same
computation graph. Only the detach between segments prevents OOM.
"""

import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig
from coral.model.coral_core import CoralCore, CoralOutput
from coral.model.halting import should_halt
from coral.training.losses import CoralLoss
from coral.training.optimizer import build_optimizer, build_scheduler

log = logging.getLogger(__name__)


class TrainerV4:
    """Training loop for CORAL v4 with deep supervision.

    Args:
        adapter:      GridAdapter (or other adapter) for encode/decode.
        core:         CoralCore reasoning module.
        loss_fn:      CoralLoss instance.
        config:       Full CoralConfig.
        wandb_run:    Optional W&B run object for logging.
    """

    def __init__(
        self,
        adapter: GridAdapter,
        core: CoralCore,
        loss_fn: CoralLoss,
        config: CoralConfig,
        wandb_run=None,
    ) -> None:
        self.adapter = adapter
        self.core = core
        self.loss_fn = loss_fn
        self.config = config
        self.wandb_run = wandb_run
        self.device = torch.device(config.device)

        # Move models to device and dtype
        dtype = torch.bfloat16 if config.training.precision == "bfloat16" else torch.float32
        self.dtype = dtype
        self.adapter = adapter.to(self.device).to(dtype)
        self.core = core.to(self.device).to(dtype)

        # Build optimizer over all parameters
        all_params = list(adapter.parameters()) + list(core.parameters())
        self.optimizer = build_optimizer(
            nn.ModuleList([adapter, core]),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=tuple(config.training.betas),
            optimizer_type=config.training.optimizer,
        )

        self.step = 0

    def _forward_segment(
        self,
        z_states: List[torch.Tensor],
        labels: torch.Tensor,
        is_last_segment: bool = False,
    ) -> tuple:
        """Run one deep supervision segment.

        Returns:
            (updated_z_states, segment_loss, breakdown, halting_info)
        """
        # Run each level's backbone steps
        device = z_states[0].device
        cfg = self.core.config
        n_levels = cfg.n_levels

        # Top-down predictions
        predictions = [None] * n_levels
        if cfg.use_predictive_coding:
            for i in range(n_levels - 2, -1, -1):
                predictions[i] = self.core.pc_modules[i].predict(z_states[i + 1])

        # Bottom-up recurrence
        seg_eps = {}
        seg_pi = {}
        error_signals = [None] * n_levels

        for i in range(n_levels):
            cond = None
            if predictions[i] is not None:
                cond = predictions[i]
            if i > 0 and error_signals[i] is not None:
                cond = cond + error_signals[i] if cond is not None else error_signals[i]

            n_steps = self.core.level_steps[i]
            z_states[i] = self.core._run_level(z_states[i], i, n_steps, conditioning=cond)

            if cfg.use_predictive_coding and i < n_levels - 1:
                mu, eps, pi, xi, xi_up = self.core.pc_modules[i](z_states[i], z_states[i + 1])
                seg_eps[f"level_{i}"] = eps
                seg_pi[f"level_{i}"] = pi
                error_signals[i + 1] = xi_up

        # Decode and compute loss
        logits = self.adapter.decode(z_states[0])
        h_k, q_halt_logit, q_continue_logit = self.core.halting(z_states)

        seg_loss, breakdown = self.loss_fn(
            logits=logits,
            labels=labels,
            pred_errors=seg_eps if seg_eps else None,
            precisions=seg_pi if seg_pi else None,
            q_halt_logits=q_halt_logit,
            q_continue_logits=q_continue_logit,
        )

        return z_states, seg_loss, breakdown, (h_k, q_halt_logit, q_continue_logit)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run one training step (one batch, all segments).

        Args:
            batch: Dict with "inputs" [B, 81] and "labels" [B, 81].

        Returns:
            Dict of scalar metrics for logging.
        """
        self.adapter.train()
        self.core.train()

        inputs = batch["inputs"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            # Encode inputs
            z1_init = self.adapter.encode(inputs)

            # Initialise level states
            B, L, _ = z1_init.shape
            z_states = [z1_init]
            for i in range(1, self.core.config.n_levels):
                d_l = self.core.config.level_dims[i]
                z_states.append(torch.zeros(B, L, d_l, device=self.device, dtype=self.dtype))

            total_loss = torch.tensor(0.0, device=self.device)
            all_breakdowns = []
            all_pred_errors = []
            K_max = self.core.config.K_max

            for seg in range(K_max):
                z_states, seg_loss, breakdown, (h_k, q_halt_logit, q_cont_logit) = \
                    self._forward_segment(z_states, labels)

                total_loss = total_loss + seg_loss
                all_breakdowns.append(breakdown)

                # Collect pred errors for amortisation loss
                seg_eps = {}
                if self.core.config.use_predictive_coding:
                    all_pred_errors.append(seg_eps)

                # Check halting
                halt = should_halt(
                    h_k,
                    threshold=self.core.config.halting_threshold,
                    exploration_prob=self.core.config.halting_exploration_prob,
                    training=True,
                )

                # CRITICAL: detach between segments
                z_states = [z.detach() for z in z_states]

                if halt:
                    break

        # Backward and optimizer step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.adapter.parameters()) + list(self.core.parameters()),
            self.config.training.gradient_clip,
        )
        self.optimizer.step()
        self.step += 1

        # Aggregate metrics
        metrics: Dict[str, float] = {"loss/total": total_loss.item()}
        if all_breakdowns:
            last_bd = all_breakdowns[-1]
            for k, v in last_bd.items():
                metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        metrics["train/num_segments"] = len(all_breakdowns)

        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        K_override: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run one evaluation step.

        Args:
            batch: Dict with "inputs" and "labels".
            K_override: If provided, force exactly K segments (for Pareto eval).

        Returns:
            Dict of scalar metrics.
        """
        self.adapter.eval()
        self.core.eval()

        inputs = batch["inputs"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            z1 = self.adapter.encode(inputs)
            B, L, _ = z1.shape
            z_states = [z1]
            for i in range(1, self.core.config.n_levels):
                d_l = self.core.config.level_dims[i]
                z_states.append(torch.zeros(B, L, d_l, device=self.device, dtype=self.dtype))

            K_max = K_override if K_override is not None else self.core.config.K_max
            num_segs = 0

            for seg in range(K_max):
                # Top-down predictions
                n_levels = self.core.config.n_levels
                predictions = [None] * n_levels
                error_signals = [None] * n_levels
                if self.core.config.use_predictive_coding:
                    for i in range(n_levels - 2, -1, -1):
                        predictions[i] = self.core.pc_modules[i].predict(z_states[i + 1])

                for i in range(n_levels):
                    cond = predictions[i]
                    if i > 0 and error_signals[i] is not None:
                        cond = cond + error_signals[i] if cond is not None else error_signals[i]
                    n_steps = self.core.level_steps[i]
                    z_states[i] = self.core._run_level(z_states[i], i, n_steps, conditioning=cond)
                    if self.core.config.use_predictive_coding and i < n_levels - 1:
                        _, _, _, _, xi_up = self.core.pc_modules[i](z_states[i], z_states[i + 1])
                        error_signals[i + 1] = xi_up

                h_k, _, _ = self.core.halting(z_states)
                num_segs = seg + 1
                z_states = [z.detach() for z in z_states]

                if K_override is None and should_halt(h_k, threshold=self.core.config.halting_threshold, training=False, exploration_prob=0.0):
                    break

        # Compute accuracy
        logits = self.adapter.decode(z_states[0])
        preds = logits.argmax(dim=-1)  # [B, L]

        mask = labels != -100
        correct_tokens = (preds == labels) & mask
        exact_correct = (correct_tokens.sum(-1) == mask.sum(-1))  # [B]

        exact_acc = exact_correct.float().mean().item()
        token_acc = (correct_tokens.sum().float() / mask.sum().float()).item() if mask.sum() > 0 else 0.0

        return {
            "eval/exact_accuracy": exact_acc,
            "eval/token_accuracy": token_acc,
            "eval/avg_halting_step": float(num_segs),
        }

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B and console."""
        step = step or self.step
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        log.info(f"Step {step}: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
