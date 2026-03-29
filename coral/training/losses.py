"""CORAL v4 — Unified loss function.

Loss components:
  L_task:   Stablemax cross-entropy (float64) — always active
  L_pred:   Precision-weighted prediction error — when use_predictive_coding
  L_pi:     Precision regulariser (symmetric log-normal) — when use_predictive_coding
  L_halt:   Q-learning halting loss — always active
  L_amort:  Amortisation pressure (sum of ||eps||²) — Experiment 2+
  L_crystal: Crystallisation loss — Experiment 3+
  L_commit: Commitment loss — Experiment 3+

All components returned individually for W&B logging.

CRITICAL: L_pi uses (log π)² NOT -log π. The latter causes precision explosion.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig
from coral.model.predictive_coding import (
    precision_regulariser,
    precision_weighted_prediction_loss,
)

IGNORE_LABEL_ID: int = -100


# ---------------------------------------------------------------------------
# Stablemax cross-entropy (float64)
# ---------------------------------------------------------------------------


def _stablemax_s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """Stablemax transfer function: maps R → (0, ∞).

    For x < 0: s(x) = 1 / (1 - x + ε)
    For x ≥ 0: s(x) = x + 1
    """
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def _log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log stablemax — computed in float64 for numerical stability."""
    s_x = _stablemax_s(x)
    return torch.log(s_x / s_x.sum(dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    """Per-token stablemax cross-entropy (float64 computation).

    Args:
        logits: [B, L, vocab_size] — raw scores.
        labels: [B, L] — integer token ids; positions with ignore_index are masked.

    Returns:
        Per-token losses [B, L]; masked positions have loss=0.
    """
    # Cast to float64 to prevent NaN in stablemax denominator
    logprobs = _log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    prediction_logprobs = torch.gather(
        logprobs,
        index=safe_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)

    return -torch.where(
        valid_mask, prediction_logprobs, torch.zeros_like(prediction_logprobs)
    )


# ---------------------------------------------------------------------------
# Amortisation loss
# ---------------------------------------------------------------------------


def amortisation_loss(
    pred_errors: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Amortisation pressure: penalise large prediction errors late in training.

    Encourages the model to front-load reasoning (resolve errors quickly).
    L_amort = Σ_{k,l} ||ε_l^k||² (sum over segments k and levels l)

    Args:
        pred_errors: List of dicts per segment, each mapping level_key → eps tensor.

    Returns:
        Scalar loss.
    """
    total = torch.tensor(0.0)
    if not pred_errors:
        return total
    # Use the device of the first available tensor
    device = None
    for seg_errs in pred_errors:
        for eps in seg_errs.values():
            device = eps.device
            total = total.to(device)
            total = total + eps.pow(2).sum(dim=-1).mean()
    return total


# ---------------------------------------------------------------------------
# Unified CORAL loss
# ---------------------------------------------------------------------------


class CoralLoss(nn.Module):
    """Unified loss function for CORAL v4.

    Components are enabled/disabled based on training phase config.

    Args:
        config: Model configuration with lambda weights and flags.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pred_errors: Optional[Dict[str, torch.Tensor]] = None,
        precisions: Optional[Dict[str, torch.Tensor]] = None,
        q_halt_logits: Optional[torch.Tensor] = None,
        q_continue_logits: Optional[torch.Tensor] = None,
        all_pred_errors: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and breakdown.

        Args:
            logits:             [B, L, vocab_size] — task logits (for cross-entropy)
            labels:             [B, L] — integer labels
            pred_errors:        Dict level_key → [B, L, d_l] prediction errors
            precisions:         Dict level_key → [B, L, d_l] precision vectors
            q_halt_logits:      [B] Q-halt logits for halting loss
            q_continue_logits:  [B] Q-continue logits for halting loss
            all_pred_errors:    List of pred_error dicts across all segments
                                (for amortisation loss)

        Returns:
            (total_loss, breakdown_dict) where breakdown contains all
            individual loss components for W&B logging.
        """
        breakdown: Dict[str, torch.Tensor] = {}
        device = logits.device

        # ---- Task loss (always active, float64) ----
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1).clamp_min(1).float()  # [B]
        per_token = stablemax_cross_entropy(logits, labels)  # [B, L]
        # Per-sequence mean, then sum over batch
        L_task = (per_token.sum(-1).float() / loss_counts).mean()
        breakdown["loss/task"] = L_task.float()

        # ---- Predictive coding losses ----
        L_pred = torch.tensor(0.0, device=device)
        L_pi = torch.tensor(0.0, device=device)

        if self.config.use_predictive_coding and pred_errors and precisions:
            for key in pred_errors:
                if key in precisions:
                    eps = pred_errors[key]
                    pi = precisions[key]
                    L_pred = L_pred + precision_weighted_prediction_loss(eps, pi)
                    L_pi = L_pi + precision_regulariser(pi)

            L_pred = self.config.lambda_pred * L_pred
            L_pi = self.config.lambda_pi * L_pi

        breakdown["loss/prediction"] = L_pred
        breakdown["loss/precision_reg"] = L_pi

        # ---- Halting loss ----
        L_halt = torch.tensor(0.0, device=device)
        if q_halt_logits is not None:
            # Sequence is correct if argmax matches all non-ignored labels
            preds = logits.argmax(dim=-1)  # [B, L]
            correct_tokens = (preds == labels) & mask  # [B, L]
            seq_is_correct = correct_tokens.sum(-1) == mask.sum(-1)  # [B]

            halt_target = seq_is_correct.float()
            L_halt = F.binary_cross_entropy_with_logits(
                q_halt_logits, halt_target, reduction="mean"
            )
            if q_continue_logits is not None:
                # Bootstrap: use q_halt as proxy target for continue
                with torch.no_grad():
                    target_continue = torch.sigmoid(q_halt_logits.detach())
                L_halt = L_halt + F.binary_cross_entropy_with_logits(
                    q_continue_logits, target_continue, reduction="mean"
                )
            L_halt = 0.5 * L_halt

        breakdown["loss/halting"] = L_halt

        # ---- Amortisation loss (Experiment 2+) ----
        L_amort = torch.tensor(0.0, device=device)
        if self.config.use_amort and all_pred_errors is not None:
            L_amort = self.config.lambda_amort * amortisation_loss(all_pred_errors)
        breakdown["loss/amortisation"] = L_amort

        # ---- Crystallisation losses (Experiment 3+) ----
        L_crystal = torch.tensor(0.0, device=device)
        L_commit = torch.tensor(0.0, device=device)
        breakdown["loss/crystallisation"] = L_crystal
        breakdown["loss/commitment"] = L_commit

        # ---- Total ----
        total = L_task.float() + L_pred + L_pi + L_halt + L_amort + L_crystal + L_commit
        breakdown["loss/total"] = total

        return total, breakdown
