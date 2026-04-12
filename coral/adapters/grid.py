"""CORAL v4 — Grid adapter (Sudoku, ARC, Maze).

Encodes a 2D grid of discrete tokens to d_1=512 embeddings and decodes
back to per-cell token logits.

Encoding:
  - Token embedding: vocab_size → d_model
  - Full 2D positional embedding: one learned vector per (row, col) pair,
    pos_emb[H*W, d_model]. Rank ceiling = min(H*W, d_model), vs H+W for
    the old factorized (row+col) additive scheme.
  - Optional embed_scale: multiply by sqrt(d_model) before layer norm so the
    input signal maintains appropriate magnitude relative to the residual stream
    during deep recurrence (standard LLM practice; matches TRM's embed_scale).

Decoding:
  - Linear: d_1 → vocab_size

For Sudoku: seq_len=81, vocab_size=11, grid is 9×9.
For Maze:   seq_len=900, vocab_size=6, grid is 30×30.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from coral.adapters.base import BaseAdapter
from coral.config import CoralConfig


class GridAdapter(BaseAdapter):
    """Encoder/decoder for grid-based tasks (Sudoku, ARC, Maze).

    Args:
        config: Top-level CoralConfig.
        vocab_size: Number of token types (overrides config.model.vocab_size).
        grid_height: Number of rows (9 for Sudoku).
        grid_width: Number of columns (9 for Sudoku).
    """

    def __init__(
        self,
        config: CoralConfig,
        vocab_size: Optional[int] = None,
        grid_height: int = 9,
        grid_width: int = 9,
    ) -> None:
        super().__init__()
        self.d_model = config.model.backbone_dim
        self.vocab_size = vocab_size if vocab_size is not None else config.model.vocab_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.seq_len = grid_height * grid_width
        self.embed_scale = getattr(config.model, "embed_scale", True)

        # Token embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)

        # Full joint 2D positional embedding: one vector per (row, col) pair.
        # Rank ceiling = min(H*W, d_model) instead of H+W for factorized row+col.
        self.pos_emb = nn.Embedding(grid_height * grid_width, self.d_model)

        # Layer norm for stable input to core
        self.input_norm = nn.LayerNorm(self.d_model)

        # Decoder: d_1 → vocab_size
        self.decoder = nn.Linear(self.d_model, self.vocab_size, bias=True)

        self._init_weights()

        # Register position index buffers.
        # row_indices and col_indices are kept for build_attention_masks.
        # pos_indices = row * grid_width + col is the flat 2D lookup index.
        rows = torch.arange(grid_height).unsqueeze(1).expand(grid_height, grid_width).reshape(-1)
        cols = torch.arange(grid_width).unsqueeze(0).expand(grid_height, grid_width).reshape(-1)
        self.register_buffer("row_indices", rows, persistent=True)
        self.register_buffer("col_indices", cols, persistent=True)
        self.register_buffer("pos_indices", rows * grid_width + cols, persistent=True)

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)
        nn.init.normal_(self.decoder.weight, std=(self.d_model ** -0.5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode grid tokens to d_model embeddings.

        Args:
            x: [B, seq_len] integer token tensor (e.g., [B, 81] for Sudoku).

        Returns:
            [B, seq_len, d_model] float embedding tensor.
        """
        B = x.shape[0]

        # Token embedding
        tok = self.token_emb(x)  # [B, L, d_model]

        # Full joint 2D position embedding: one vector per (row, col) pair.
        pos = self.pos_emb(self.pos_indices).unsqueeze(0)  # [1, L, d_model]

        emb = tok + pos
        emb = self.input_norm(emb)
        if self.embed_scale:
            emb = emb * math.sqrt(self.d_model)
        return emb

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode d_model embeddings to per-cell token logits.

        Args:
            z: [B, seq_len, d_model]

        Returns:
            logits: [B, seq_len, vocab_size]
        """
        return self.decoder(z)

    def get_predictions(self, z: torch.Tensor) -> torch.Tensor:
        """Get argmax predictions from embeddings.

        Returns:
            [B, seq_len] integer predictions.
        """
        return self.decode(z).argmax(dim=-1)

    def build_attention_masks(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build binary structural adjacency masks for the grid.

        Returns three boolean-valued float tensors of shape [L, L] where
        L = grid_height * grid_width.  Each mask[i, j] is 1.0 if positions
        i and j share the corresponding structural property, 0.0 otherwise.

        These are used by the backbone to weight its 3 learnable attention
        bias scalars (row_bias, col_bias, box_bias).  The masks are static
        (depend only on grid shape) so they should be computed once per
        forward pass and reused across segments.

        Box size is grid_height // 3 × grid_width // 3 (i.e. 3×3 for Sudoku).

        Args:
            device: Target device.  Defaults to the device of the adapter's
                    registered buffers (self.row_indices.device).

        Returns:
            same_row: [L, L] — 1.0 where i and j are in the same row.
            same_col: [L, L] — 1.0 where i and j are in the same column.
            same_box: [L, L] — 1.0 where i and j are in the same 3×3 box.
        """
        if device is None:
            device = self.row_indices.device

        row = self.row_indices  # [L]
        col = self.col_indices  # [L]

        # Box index: (row // box_h) * n_box_cols + (col // box_w)
        # Designed for grids where height and width are divisible by 3.
        # For non-divisible grids, fall back to row-only grouping for the box mask
        # (same_box will equal same_row in that case).
        if self.grid_height % 3 == 0 and self.grid_width % 3 == 0:
            box_h = self.grid_height // 3
            box_w = self.grid_width // 3
            n_box_cols = self.grid_width // box_w
            box_idx = (row // box_h) * n_box_cols + (col // box_w)  # [L]
        else:
            # No sub-box structure: treat each row as its own "box"
            box_idx = row  # [L]

        same_row = (row.unsqueeze(1) == row.unsqueeze(0)).float().to(device)  # [L, L]
        same_col = (col.unsqueeze(1) == col.unsqueeze(0)).float().to(device)  # [L, L]
        same_box = (box_idx.unsqueeze(1) == box_idx.unsqueeze(0)).float().to(device)  # [L, L]

        return same_row, same_col, same_box
