"""CORAL v5 — Feature-selective conditioning gate.

Replaces v4's scalar cond_gate parameter with a context-dependent MLP that
produces a per-feature gate in [0, 1] via sigmoid.

Validated on Phase 1 synthetic selective-integration task (Model C):
  - reliable-dim MSE: 0.47  (scalar gate: 0.57; gate x precision: 0.98)
  - gate ratio (reliable / unreliable): 1.66x

Design choices:
  - 2-layer MLP with ReLU (same as Phase 1 validated Model C)
  - Hidden dim defaults to d_model (proportional parameter count per level)
  - Output bias initialised to -2.0: sigmoid(-2) ~ 0.12 — gate starts mostly
    closed; model must learn to open specific features
  - Sigmoid output: bounded in [0, 1] by construction
  - Precision is NOT in this path (Phase 1 showed gate x precision is harmful)

Biological analog: attentional gain control via prefrontal-to-sensory feedback
connections, operating feature-selectively. Distinct from precision, which
operates in the loss path as a learning-rate modulator analogous to
neuromodulatory systems.
"""

import torch
import torch.nn as nn

from coral.model.predictive_coding import rms_normalize  # noqa: F401 — re-exported for convenience


class ConditioningGate(nn.Module):
    """Context-dependent feature-selective gate for top-down conditioning.

    Takes the current state z as input. Produces a per-feature gate in [0, 1]
    via sigmoid. The gate modulates which features of a top-down prediction are
    integrated into the state.

    Args:
        d_model:    Dimensionality of the state (and gate output).
        hidden_dim: Hidden layer width. None / 0 defaults to d_model.
        init_bias:  Initial bias on the output layer. sigmoid(-2) ~ 0.12
                    means the gate starts mostly closed.
    """

    def __init__(self, d_model: int, hidden_dim: int = 0, init_bias: float = -2.0) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim and hidden_dim > 0 else d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )
        # Output weight initialised to zero so the gate starts at a uniform
        # sigmoid(init_bias) across all features, regardless of input.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-feature gate from state z.

        Args:
            z: [..., d_model] — current state tensor (any leading dims).

        Returns:
            gate: same shape as z, values in (0, 1).
        """
        return torch.sigmoid(self.net(z))
