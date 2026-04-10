"""CORAL v4 — Training loop with deep supervision.

Training structure:
  For each batch:
    z1 = adapter.encode(inputs)
    output = core(z1, K_max=K_max, training=True, decode_fn=adapter.decode)

    For each segment in output.all_logits:
      seg_loss = loss_fn(logits, labels, pred_errors[i], precisions[i],
                         q_halt_logits[i], q_continue_logits[i])
      total_loss += seg_loss

    total_loss.backward()
    clip_grad_norm_(params, 1.0)
    optimizer.step()

CoralCore manages the segment loop, detach boundaries, halting, and
predictive coding internally. The trainer only handles the outer
train/eval loop, optimiser, and loss aggregation.
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig
from coral.model.coral_core import CoralCore
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

        dtype = torch.bfloat16 if config.training.precision == "bfloat16" else torch.float32
        self.dtype = dtype
        self.adapter = adapter.to(self.device).to(dtype)
        self.core = core.to(self.device).to(dtype)

        self.optimizer = build_optimizer(
            nn.ModuleList([adapter, core]),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=tuple(config.training.betas),
            optimizer_type=config.training.optimizer,
        )

        self.step = 0

        # Build attention masks once (static — depend only on grid shape).
        # Passed to core.forward() so the learned row/col/box bias scalars
        # are applied consistently during both training and evaluation.
        self._attention_masks: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if hasattr(adapter, "build_attention_masks") and config.model.use_local_attention_bias:
            self._attention_masks = adapter.build_attention_masks(device=self.device)

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
            z1_init = self.adapter.encode(inputs)

            output = self.core(
                z1_init,
                K_max=self.core.config.K_max,
                training=True,
                decode_fn=self.adapter.decode,
                attention_masks=self._attention_masks,
            )

            total_loss = torch.tensor(0.0, device=self.device)
            all_breakdowns = []

            num_segs = len(output.all_logits)
            weighting = getattr(self.config.training, "deep_supervision_weighting", "uniform")

            for i, logits in enumerate(output.all_logits):
                seg_loss, breakdown = self.loss_fn(
                    logits=logits,
                    labels=labels,
                    pred_errors=output.pred_errors[i] if output.pred_errors else None,
                    precisions=output.precisions[i] if output.precisions else None,
                    q_halt_logits=output.q_halt_logits[i] if output.q_halt_logits else None,
                    q_continue_logits=output.q_continue_logits[i] if output.q_continue_logits else None,
                    all_pred_errors=output.pred_errors if i == len(output.all_logits) - 1 else None,
                    commitment_loss=output.commit_losses[i] if output.commit_losses else None,
                    disentanglement_loss=output.dis_losses[i] if output.dis_losses else None,
                )
                if weighting == "linear":
                    # weight_i = (i+1) / num_segs, normalised so sum == num_segs
                    # => weight_i = 2*(i+1) / (num_segs+1)
                    weight = 2.0 * (i + 1) / (num_segs + 1)
                else:
                    weight = 1.0
                total_loss = total_loss + weight * seg_loss
                all_breakdowns.append(breakdown)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.adapter.parameters()) + list(self.core.parameters()),
            self.config.training.gradient_clip,
        )
        self.optimizer.step()
        self.step += 1

        metrics: Dict[str, float] = {}
        if all_breakdowns:
            for k, v in all_breakdowns[-1].items():
                if k == "loss/total":
                    # Rename to avoid overwriting the real training loss (sum across segments).
                    metrics["loss/last_seg_total"] = v.item() if isinstance(v, torch.Tensor) else v
                else:
                    metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        # loss/total = sum across ALL segments — this is what backward() received.
        # loss/per_segment_avg = normalised per-segment loss for monitoring.
        metrics["loss/total"] = total_loss.item()
        metrics["loss/per_segment_avg"] = total_loss.item() / max(output.num_segments, 1)
        metrics["train/num_segments"] = output.num_segments

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
            output = self.core(
                z1,
                K_max=K_override if K_override is not None else self.core.config.K_max,
                training=False,
                decode_fn=self.adapter.decode,
                attention_masks=self._attention_masks,
            )

        logits = output.all_logits[-1] if output.all_logits else self.adapter.decode(output.z_states[0])
        preds = logits.argmax(dim=-1)  # [B, L]

        mask = labels != -100
        correct_tokens = (preds == labels) & mask
        exact_correct = (correct_tokens.sum(-1) == mask.sum(-1))  # [B]

        exact_acc = exact_correct.float().mean().item()
        token_acc = (correct_tokens.sum().float() / mask.sum().float()).item() if mask.sum() > 0 else 0.0

        metrics: dict = {
            "eval/exact_accuracy": exact_acc,
            "eval/token_accuracy": token_acc,
            "eval/avg_halting_step": float(output.num_segments),
        }

        # Maze-specific metrics (gated on dataset name)
        if getattr(self.config.data, "dataset", "") == "maze_30x30_hard":
            # path_accuracy: accuracy on optimal-path cells (label == 5)
            path_mask = labels == 5
            if path_mask.sum() > 0:
                metrics["eval/path_accuracy"] = (
                    ((preds == labels) & path_mask).sum().float() / path_mask.sum().float()
                ).item()
            # wall_accuracy: accuracy on wall cells (label == 2)
            wall_mask = labels == 2
            if wall_mask.sum() > 0:
                metrics["eval/wall_accuracy"] = (
                    ((preds == labels) & wall_mask).sum().float() / wall_mask.sum().float()
                ).item()

        # ---- Crystallisation diagnostics (mode="full" only) ----
        use_crys = (
            self.config.model.use_crystallisation
            and hasattr(self.core, "crystallisation_manager")
            and self.core.crystallisation_manager is not None
        )
        if use_crys and output.crystal_stats:
            crys_mgr = self.core.crystallisation_manager
            crys_stats_overall = crys_mgr.get_stats()

            # Aggregate crystallisation rate from the last segment's stats
            last_stats = output.crystal_stats[-1]
            if "crystallisation_rate" in last_stats:
                metrics["crystal/rate_total"] = last_stats["crystallisation_rate"].item()
            if "per_head_rates" in last_stats:
                per_head = last_stats["per_head_rates"]
                for h, rate in enumerate(per_head):
                    metrics[f"crystal/rate_head_{h}"] = rate.item()

            # Codebook health
            perplexity = crys_stats_overall.get("perplexity")
            if perplexity is not None:
                for h, p in enumerate(perplexity):
                    metrics[f"codebook/perplexity_head_{h}"] = p.item()

            # Bypass accuracy: decode from crystallised (enforced) state
            z_full = output.z_states[0]
            z_crystal = crys_mgr.enforce_after_backbone(z_full)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                logits_crystal = self.adapter.decode(z_crystal)
            preds_crystal = logits_crystal.argmax(dim=-1)
            bypass_correct = (preds_crystal == labels) & mask
            bypass_exact = (bypass_correct.sum(-1) == mask.sum(-1))
            metrics["crystal/bypass_accuracy"] = bypass_exact.float().mean().item()

        return metrics

    @torch.no_grad()
    def compute_repr_diagnostics(
        self,
        eval_loader: Iterable,
        max_puzzles: int = 500,
        pca_sample: int = 5000,
    ) -> Dict[str, float]:
        """Compute representation diagnostics on a sample of eval puzzles.

        Runs the model at full K_max and analyses the final-segment states.
        Metrics:
          repr/inter_position_similarity — mean pairwise cosine sim among empty-cell states
          repr/same_digit_similarity     — mean within-digit cosine sim, averaged over digits
          repr/effective_rank            — PCA components explaining 90% of variance
          repr/state_norm_mean           — mean L2 norm of empty-cell states
          repr/state_norm_std            — std of L2 norms

        Args:
            eval_loader: Iterable yielding batches with "inputs" and "labels".
            max_puzzles: Maximum number of puzzles to process (for speed).
            pca_sample:  Number of states sampled for the SVD/PCA estimate.

        Returns:
            Dict of repr/* metrics (empty if not enough data).
        """
        # Sudoku-specific: "empty cells" (token==1) have no equivalent in maze.
        # Skip entirely for non-Sudoku datasets.
        if getattr(self.config.data, "dataset", "sudoku_extreme_1k") != "sudoku_extreme_1k":
            return {}

        self.adapter.eval()
        self.core.eval()

        all_states: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_empty: List[torch.Tensor] = []
        n_collected = 0

        # Temporarily disable halting so we always run full K_max segments
        orig_threshold = self.core.config.halting_threshold
        self.core.config.halting_threshold = 2.0  # sigmoid ∈ (0,1) → never triggers

        try:
            for batch in eval_loader:
                if n_collected >= max_puzzles:
                    break
                inputs = batch["inputs"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    z1 = self.adapter.encode(inputs)
                    output = self.core(
                        z1,
                        K_max=self.core.config.K_max,
                        training=False,
                        decode_fn=None,
                    )

                # z_states[0] = level-0 state after the final segment
                z_final = output.z_states[0].float().cpu()   # [B, L, d]
                empty_mask = (inputs == 1).cpu()             # [B, L] — empty cells are token 1
                labels_cpu = labels.cpu()                    # [B, L]

                all_states.append(z_final)
                all_labels.append(labels_cpu)
                all_empty.append(empty_mask)
                n_collected += inputs.shape[0]
        finally:
            self.core.config.halting_threshold = orig_threshold

        if not all_states:
            return {}

        states = torch.cat(all_states, dim=0)   # [N, L, d]
        labels_t = torch.cat(all_labels, dim=0)  # [N, L]
        empty_t = torch.cat(all_empty, dim=0)    # [N, L]

        N, L, d = states.shape
        states_flat = states.reshape(N * L, d)
        labels_flat = labels_t.reshape(N * L)
        empty_flat = empty_t.reshape(N * L)

        empty_states = states_flat[empty_flat]   # [E, d]
        empty_labels = labels_flat[empty_flat]   # [E]

        metrics: Dict[str, float] = {}
        E = empty_states.shape[0]
        if E < 2:
            return metrics

        # L2-normalise for cosine similarity
        norms = empty_states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = empty_states / norms

        # a) Inter-position similarity (random sample to limit O(S²) cost)
        S = min(500, E)
        idx = torch.randperm(E)[:S]
        sample = normed[idx]                             # [S, d]
        sim = sample @ sample.T                         # [S, S]
        off_diag = ~torch.eye(S, dtype=torch.bool)
        metrics["repr/inter_position_similarity"] = sim[off_diag].mean().item()

        # b) Same-digit similarity (mean across digits 1–9)
        digit_sims: List[float] = []
        for digit in range(1, 10):
            dmask = empty_labels == digit
            if dmask.sum() < 2:
                continue
            dstates = normed[dmask]
            Sd = min(200, dstates.shape[0])
            didx = torch.randperm(dstates.shape[0])[:Sd]
            dsample = dstates[didx]                      # [Sd, d]
            dsim = dsample @ dsample.T                   # [Sd, Sd]
            doff = ~torch.eye(Sd, dtype=torch.bool)
            digit_sims.append(dsim[doff].mean().item())
        if digit_sims:
            metrics["repr/same_digit_similarity"] = sum(digit_sims) / len(digit_sims)

        # c) Effective rank via SVD (90% variance threshold)
        pca_n = min(pca_sample, E)
        pidx = torch.randperm(E)[:pca_n]
        pca_states = empty_states[pidx].float()          # [pca_n, d]
        centered = pca_states - pca_states.mean(dim=0)
        try:
            _, S_vals, _ = torch.linalg.svd(centered, full_matrices=False)
            var_ratio = (S_vals ** 2).cumsum(0) / (S_vals ** 2).sum()
            effective_rank = int((var_ratio < 0.9).sum().item()) + 1
            metrics["repr/effective_rank"] = float(effective_rank)
        except Exception:
            pass

        # d) State norm statistics (all empty cells)
        state_norms = empty_states.norm(dim=-1)           # [E]
        metrics["repr/state_norm_mean"] = state_norms.mean().item()
        metrics["repr/state_norm_std"] = state_norms.std().item()

        return metrics

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B and console."""
        step = step or self.step
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        log.info(f"Step {step}: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
