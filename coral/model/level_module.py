"""CORAL v4 — Level module.

Each hierarchy level owns:
- Up-projection: d_l → d_backbone (512)
- Down-projection: d_backbone → d_l
- The level state z_l lives in d_l dimensions; the backbone always operates
  at d_backbone=512.

Level 1 (d=512): projections are identity (no-op Linear).
Level 2 (d=256): 256→512 up, 512→256 down.
Level 3 (d=128): 128→512 up, 512→128 down.
Level 4 (d=64):  64→512 up, 512→64 down.
"""

from typing import List

import torch
import torch.nn as nn

from coral.config import ModelConfig


class LevelModule(nn.Module):
    """Level module: manages up/down projections for one hierarchy level.

    Args:
        level_idx: 0-based index (level 0 = fastest/lowest, level N-1 = slowest/highest).
        dim: The dimension d_l of this level's state.
        backbone_dim: The backbone's operating dimension (always 512).
    """

    def __init__(self, level_idx: int, dim: int, backbone_dim: int = 512) -> None:
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.backbone_dim = backbone_dim

        if dim == backbone_dim:
            # Identity projections — register as None to skip computation
            self.up_proj: nn.Module = nn.Identity()
            self.down_proj: nn.Module = nn.Identity()
        else:
            self.up_proj = nn.Linear(dim, backbone_dim, bias=False)
            self.down_proj = nn.Linear(backbone_dim, dim, bias=False)
            # Initialise with scaled identity-like weights
            nn.init.normal_(self.up_proj.weight, std=(backbone_dim ** -0.5))
            nn.init.normal_(self.down_proj.weight, std=(backbone_dim ** -0.5))

    def project_up(self, z: torch.Tensor) -> torch.Tensor:
        """Project level state up to backbone dimension.

        Args:
            z: [B, L, d_l]

        Returns:
            [B, L, backbone_dim]
        """
        return self.up_proj(z)

    def project_down(self, h: torch.Tensor) -> torch.Tensor:
        """Project backbone output down to level dimension.

        Args:
            h: [B, L, backbone_dim]

        Returns:
            [B, L, d_l]
        """
        return self.down_proj(h)

    def init_state(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Initialise level state to zeros.

        Returns:
            [B, L, d_l] zero tensor.
        """
        return torch.zeros(batch_size, seq_len, self.dim, device=device, dtype=dtype)


class LevelStack(nn.Module):
    """Container for all LevelModules in the hierarchy.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_levels = config.n_levels
        self.level_dims = config.level_dims
        self.backbone_dim = config.backbone_dim

        assert len(config.level_dims) == config.n_levels, (
            f"len(level_dims)={len(config.level_dims)} != n_levels={config.n_levels}"
        )

        self.modules_list = nn.ModuleList([
            LevelModule(
                level_idx=i,
                dim=config.level_dims[i],
                backbone_dim=config.backbone_dim,
            )
            for i in range(config.n_levels)
        ])

    def __getitem__(self, idx: int) -> LevelModule:
        return self.modules_list[idx]  # type: ignore[return-value]

    def __len__(self) -> int:
        return self.n_levels

    def init_states(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Initialise all level states.

        Returns:
            List of [B, L, d_l] tensors, one per level.
        """
        return [
            m.init_state(batch_size, seq_len, device, dtype)
            for m in self.modules_list
        ]
