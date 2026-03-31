"""CORAL v4 — Shared 2-layer transformer backbone.

Weight-shared across all hierarchy levels and recursion steps.
Level-specific behaviour is induced by additive level embeddings passed in
at call time; the backbone itself has no notion of level.

Architecture:
  - 2 transformer layers
  - d_model=512, n_heads=8, d_k=64
  - SwiGLU FFN with 4× expansion
  - RMSNorm (pre-norm, no learnable parameters)
  - Rotary position encoding (RoPE)
  - PyTorch SDPA attention (no flash-attn dependency)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root mean square layer normalisation (no learnable shift/scale)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


# ---------------------------------------------------------------------------
# Rotary position encoding (RoPE)
# ---------------------------------------------------------------------------


def _precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute the complex frequency tensor for RoPE.

    Returns:
        freqs_cis: [max_seq_len, dim/2] complex tensor.
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)  # [max_seq_len, dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor.

    Args:
        x:          [B, L, n_heads, d_k] real tensor.
        freqs_cis:  [L, d_k/2] complex tensor.

    Returns:
        Rotated tensor of same shape as x.
    """
    B, L, H, D = x.shape
    x_r = x.float().reshape(B, L, H, D // 2, 2)
    x_c = torch.view_as_complex(x_r)  # [B, L, H, D/2]
    freqs = freqs_cis[:L].unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
    x_out = torch.view_as_real(x_c * freqs)  # [B, L, H, D/2, 2]
    return x_out.reshape(B, L, H, D).to(x.dtype)


# ---------------------------------------------------------------------------
# Self-attention with RoPE and SDPA
# ---------------------------------------------------------------------------


class RotaryAttention(nn.Module):
    """Multi-head self-attention with RoPE and PyTorch SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention — no flash-attn
    package required.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        freqs_cis = _precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, L, d_model]
            mask: Optional attention mask [B, 1, L, L] or [L, L]; True=keep.

        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape
        H, dk = self.n_heads, self.d_k

        q = self.q_proj(x).reshape(B, L, H, dk)
        k = self.k_proj(x).reshape(B, L, H, dk)
        v = self.v_proj(x).reshape(B, L, H, dk)

        # Apply RoPE to q and k
        freqs = self.freqs_cis[:L]
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        # SDPA expects [B, H, L, dk]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Cast mask to match query dtype — required when running bfloat16 forward
        # and the bias was built from float32 binary masks.
        if mask is not None:
            mask = mask.to(q.dtype)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        # [B, H, L, dk] → [B, L, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    Architecture:
        gate = SiLU(W_gate @ x)
        up   = W_up @ x
        out  = W_down @ (gate * up)

    Hidden dimension is (expansion * d_model), but SwiGLU uses two parallel
    projections of that size so the total parameter count is 3× d_model × d_ff.
    """

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        d_ff = d_model * expansion
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """Single transformer layer: attention + FFN, pre-norm.

    Pre-norm allows the residual stream to grow freely across recurrence steps
    while only normalising sublayer inputs. This is critical for iterative
    architectures where the same block is applied many times (21+ steps).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.attn = RotaryAttention(d_model, n_heads)
        self.ffn = SwiGLUFFN(d_model, ffn_expansion)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class CoralBackbone(nn.Module):
    """Shared 2-layer transformer backbone.

    Weight-shared across all hierarchy levels and recursion steps.
    Level-specific behaviour is induced by additive inputs (level_emb,
    timescale_emb, prediction/error signals) passed in at call time.

    v4.2: 3 learnable scalar attention bias parameters (row_bias, col_bias,
    box_bias) weight the structural adjacency masks provided by the adapter.
    When attention_bias is None (default) the backbone behaves identically to v4.1.

    Input/output shape: [B, L, d_model] where d_model=512.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.backbone_dim
        self.n_layers = config.backbone_layers

        self.layers = nn.ModuleList([
            TransformerLayer(config.backbone_dim, config.n_heads, config.ffn_expansion)
            for _ in range(config.backbone_layers)
        ])

        # v4.2: learnable scalars for local structure attention bias.
        # Only registered when use_local_attention_bias=True so that configs that
        # do not opt-in are not given unused parameters (which would break gradient
        # checks in existing tests).  Initialised to 0.0 so that at the start of
        # training the backbone is identical to the no-bias baseline.
        self.use_local_attention_bias = getattr(config, "use_local_attention_bias", False)
        if self.use_local_attention_bias:
            self.row_bias = nn.Parameter(torch.zeros(1))
            self.col_bias = nn.Parameter(torch.zeros(1))
            self.box_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, L, d_model] — input (already includes level_emb +
                            timescale_emb as additive signals from the caller).
            attention_bias: Optional [L, L] float tensor that is added to the
                            raw attention logits before softmax inside every
                            transformer layer.  Broadcastable across batch and
                            head dimensions by PyTorch SDPA.  Pass None (default)
                            to skip entirely — backward compatible with v4.1.

        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask=attention_bias)
        return x


# ---------------------------------------------------------------------------
# Level and timescale embeddings (owned by CoralCore but defined here)
# ---------------------------------------------------------------------------


class LevelEmbedding(nn.Module):
    """Learned additive level embedding.

    One embedding vector per hierarchy level, shape [d_model].
    Added to the backbone input to signal which level is being processed.
    """

    def __init__(self, n_levels: int, d_model: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(n_levels, d_model)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, level_idx: int, device: torch.device) -> torch.Tensor:
        """Return the embedding for the given level (0-indexed).

        Returns:
            [d_model] tensor.
        """
        idx = torch.tensor(level_idx, dtype=torch.long, device=device)
        return self.embeddings(idx)


class TimescaleEmbedding(nn.Module):
    """Sinusoidal timescale embedding for the step index within a level.

    Encodes the position of the current inner step (t=0..T_max-1) so the
    backbone can distinguish early vs late steps within a recursion segment.
    """

    def __init__(self, d_model: int, max_steps: int = 64) -> None:
        super().__init__()
        self.d_model = d_model

        # Precompute sinusoidal embeddings: [max_steps, d_model]
        pe = torch.zeros(max_steps, d_model)
        position = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, step_idx: int) -> torch.Tensor:
        """Return the sinusoidal embedding for the given step index.

        Returns:
            [d_model] tensor.
        """
        return self.pe[step_idx]  # type: ignore[index]
