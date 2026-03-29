"""CORAL v4 — Grid adapter (Sudoku, ARC, Maze).

Encodes a 2D grid of discrete tokens to d_1=512 embeddings and decodes
back to per-cell token logits.

Encoding:
  - Token embedding: vocab_size → d_model
  - 2D positional embedding: row_emb + col_emb (both learned)
  - Optional mask embedding for empty cells (Sudoku: digit 0 = empty)

Decoding:
  - Linear: d_1 → vocab_size

For Sudoku: seq_len=81, vocab_size=11, grid is 9×9.
"""

import math
from typing import Optional, Tuple

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

        # Token embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)

        # 2D positional embeddings (learned)
        self.row_emb = nn.Embedding(grid_height, self.d_model)
        self.col_emb = nn.Embedding(grid_width, self.d_model)

        # Layer norm for stable input to core
        self.input_norm = nn.LayerNorm(self.d_model)

        # Decoder: d_1 → vocab_size
        self.decoder = nn.Linear(self.d_model, self.vocab_size, bias=True)

        self._init_weights()

        # Register position index buffers
        rows = torch.arange(grid_height).unsqueeze(1).expand(grid_height, grid_width).reshape(-1)
        cols = torch.arange(grid_width).unsqueeze(0).expand(grid_height, grid_width).reshape(-1)
        self.register_buffer("row_indices", rows, persistent=True)
        self.register_buffer("col_indices", cols, persistent=True)

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        nn.init.normal_(self.col_emb.weight, std=0.02)
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

        # 2D position embedding
        row = self.row_emb(self.row_indices)  # [L, d_model]
        col = self.col_emb(self.col_indices)  # [L, d_model]
        pos = (row + col).unsqueeze(0)        # [1, L, d_model]

        emb = tok + pos
        return self.input_norm(emb)

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
