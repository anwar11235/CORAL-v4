"""CORAL v4 — Adaptive halting mechanism.

Q-learning based halting: the halting network learns to estimate whether
continuing computation is worth the cost, producing a halting probability
at each segment boundary.

During training: with probability epsilon_explore, continue regardless;
otherwise halt if h_k > halting_threshold.

During evaluation: halt when h_k > halting_threshold (deterministic).

The halting loss uses Q-learning:
    Q_halt:    predicted reward if halting now = correctness of current answer
    Q_continue: predicted reward if continuing = bootstrapped from next step

    L_halt = BCE(Q_halt_logit, seq_is_correct) + BCE(Q_continue_logit, target_Q)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig


class HaltingNetwork(nn.Module):
    """Q-learning adaptive halting network.

    Takes the concatenation of all level states (projected to a common dim)
    and produces:
        q_halt:     probability that halting now is correct
        q_continue: probability that continuing will improve the answer

    Architecture:
        - Per-level projection to halt_dim
        - Sum pooling over sequence dimension
        - 2-layer MLP with ReLU → 2 scalar outputs
    """

    def __init__(self, config: ModelConfig, halt_dim: int = 128) -> None:
        super().__init__()
        self.n_levels = config.n_levels
        self.halt_dim = halt_dim

        # Project each level's pooled state to halt_dim
        self.level_projs = nn.ModuleList([
            nn.Linear(config.level_dims[i], halt_dim, bias=False)
            for i in range(config.n_levels)
        ])

        # Q-value MLP
        self.mlp = nn.Sequential(
            nn.Linear(halt_dim, halt_dim, bias=True),
            nn.ReLU(),
            nn.Linear(halt_dim, 2, bias=True),  # [q_halt_logit, q_continue_logit]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.level_projs:
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self, level_states: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute halting probabilities from current level states.

        Args:
            level_states: List of [B, L, d_l] tensors, one per level.

        Returns:
            h:              [B] — halting probability (sigmoid of q_halt_logit)
            q_halt_logit:   [B] — raw logit for halt Q-value
            q_continue_logit: [B] — raw logit for continue Q-value
        """
        # Pool each level over sequence dimension, then project
        pooled = torch.zeros(
            level_states[0].shape[0],  # B
            self.halt_dim,
            device=level_states[0].device,
            dtype=level_states[0].dtype,
        )
        for i, z in enumerate(level_states):
            # Mean pool over L: [B, d_l] → [B, halt_dim]
            pooled = pooled + self.level_projs[i](z.mean(dim=1))

        q_out = self.mlp(pooled)  # [B, 2]
        q_halt_logit = q_out[:, 0]      # [B]
        q_continue_logit = q_out[:, 1]  # [B]
        h = torch.sigmoid(q_halt_logit)  # [B]

        return h, q_halt_logit, q_continue_logit


def halting_loss(
    q_halt_logits: torch.Tensor,
    q_continue_logits: torch.Tensor,
    seq_is_correct: torch.Tensor,
    target_q_continue: Optional[torch.Tensor] = None,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Q-learning halting loss.

    Args:
        q_halt_logits:     [B] — raw logit for halting Q-value
        q_continue_logits: [B] — raw logit for continuing Q-value
        seq_is_correct:    [B] bool — whether the current answer is fully correct
        target_q_continue: [B] float — bootstrapped Q-continue target (optional)
        gamma: discount factor for Q-learning

    Returns:
        Scalar loss.
    """
    # Q_halt target: current correctness
    q_halt_target = seq_is_correct.float()
    q_halt_loss = F.binary_cross_entropy_with_logits(
        q_halt_logits, q_halt_target, reduction="mean"
    )

    q_continue_loss = torch.tensor(0.0, device=q_halt_logits.device)
    if target_q_continue is not None:
        q_continue_loss = F.binary_cross_entropy_with_logits(
            q_continue_logits, target_q_continue, reduction="mean"
        )

    return 0.5 * (q_halt_loss + q_continue_loss)


def should_halt(
    h: torch.Tensor,
    threshold: float = 0.95,
    exploration_prob: float = 0.1,
    training: bool = True,
) -> bool:
    """Decide whether to halt computation for all sequences in batch.

    Args:
        h:                [B] halting probabilities
        threshold:        halt if h.mean() > threshold
        exploration_prob: probability of ignoring halting during training
        training:         if True, apply exploration

    Returns:
        True if we should halt this segment.
    """
    if training and torch.rand(1).item() < exploration_prob:
        return False
    return h.mean().item() > threshold
