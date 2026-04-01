"""Optimizer selection for CORAL v4.

Priority order (from build plan):
1. Fused CUDA AdamATan2  — attempt compilation; use if successful
2. torch.optim.AdamW     — proven baseline, use if fused fails
3. Pure PyTorch AdamATan2 with torch.compile — last resort

For Experiment 1 we default to AdamW.
"""

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

log = logging.getLogger(__name__)


def _try_fused_adam_atan2(
    params, lr: float, weight_decay: float, betas: tuple
) -> Optional[Optimizer]:
    """Attempt to instantiate the fused CUDA AdamATan2 optimizer."""
    try:
        from coral.training.fused_adam_atan2 import FusedAdamATan2  # type: ignore
        opt = FusedAdamATan2(params, lr=lr, weight_decay=weight_decay, betas=betas)
        log.info("Using FusedAdamATan2 (CUDA kernel).")
        return opt
    except Exception as e:
        log.warning(f"FusedAdamATan2 not available ({e}). Falling back to AdamW.")
        return None


def build_optimizer(
    model: nn.Module,
    lr: float = 7e-5,
    weight_decay: float = 1.0,
    betas: tuple = (0.9, 0.95),
    optimizer_type: str = "adamw",
) -> Optimizer:
    """Build the optimizer.

    Args:
        model: The model whose parameters to optimise.
        lr: Learning rate.
        weight_decay: Weight decay applied to non-embedding, non-norm matrix weights.
        betas: Adam beta coefficients.
        optimizer_type: "adamw" | "fused_adam_atan2".

    Returns:
        Configured optimizer.

    Parameter group rules (matching GPT-2 / LLaMA / TRM practice):
        decay:    ndim >= 2  AND name does not contain 'emb', 'embedding', or 'norm'
        no_decay: ndim < 2  (scalars, biases)
                  OR name contains 'emb' / 'embedding' (token, row, col, level, timescale)
                  OR name contains 'norm' (RMSNorm, LayerNorm scale vectors)

    Rationale for excluding embeddings: with WD=1.0 over 20k cosine-schedule steps,
    embeddings decay to ~49% of their natural scale. The empty-cell embedding (token 1)
    is re-injected at every backbone step as the primary "solve me" signal; decaying it
    toward zero directly impairs the model's access to task constraints.
    """
    # _NO_DECAY_KEYWORDS: any param whose fully-qualified name (lower-cased) contains
    # one of these strings goes into the no-decay group regardless of ndim.
    _NO_DECAY_KEYWORDS = ("emb", "embedding", "norm")

    decay_params = []
    no_decay_params = []
    decay_numel = 0
    no_decay_numel = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if param.ndim < 2 or any(kw in name_lower for kw in _NO_DECAY_KEYWORDS):
            no_decay_params.append(param)
            no_decay_numel += param.numel()
        else:
            decay_params.append(param)
            decay_numel += param.numel()

    log.info(
        f"Optimizer param split — "
        f"Decay: {len(decay_params)} tensors ({decay_numel / 1e6:.2f}M params, WD={weight_decay}), "
        f"No-decay: {len(no_decay_params)} tensors ({no_decay_numel / 1e6:.2f}M params, WD=0.0)"
    )

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type == "fused_adam_atan2":
        opt = _try_fused_adam_atan2(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)
        if opt is not None:
            return opt
        # Fall through to AdamW

    log.info(f"Using AdamW (lr={lr}, weight_decay={weight_decay}, betas={betas}).")
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=True if torch.cuda.is_available() else False)


def build_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 500,
    scheduler_type: str = "cosine",
) -> object:
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer.
        total_steps: Total number of training steps.
        warmup_steps: Linear warmup steps.
        scheduler_type: "cosine" or "constant".

    Returns:
        A PyTorch LR scheduler.
    """
    if scheduler_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

        return LambdaLR(optimizer, lr_lambda)
    else:
        def lr_lambda_const(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0
        return LambdaLR(optimizer, lr_lambda_const)
