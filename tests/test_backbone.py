"""Unit tests for CoralBackbone.

Verification criteria (from build plan):
  - Forward pass produces correct output shape
  - Gradient flows through all parameters
  - Level embedding changes output (backbone is not ignoring it)
  - Smoke test: overfit on 1 puzzle (loss → 0 within 100 steps)
"""

import pytest
import torch
import torch.nn as nn

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.backbone import (
    CoralBackbone,
    LevelEmbedding,
    RMSNorm,
    RotaryAttention,
    SwiGLUFFN,
    TimescaleEmbedding,
)


@pytest.fixture
def default_config():
    return ModelConfig(
        n_levels=2,
        level_dims=[512, 256],
        backbone_layers=2,
        backbone_dim=512,
        n_heads=8,
        ffn_expansion=4,
    )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def test_rmsnorm_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalises():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64) * 100  # large inputs
    out = norm(x)
    # RMSNorm should bring values to ~O(1)
    assert out.abs().mean().item() < 10.0


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

def test_swiglu_shape():
    ffn = SwiGLUFFN(512, 4)
    x = torch.randn(2, 10, 512)
    out = ffn(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Rotary attention
# ---------------------------------------------------------------------------

def test_rotary_attention_shape():
    attn = RotaryAttention(512, 8)
    x = torch.randn(2, 81, 512)
    out = attn(x)
    assert out.shape == (2, 81, 512)


def test_rotary_attention_gradient():
    attn = RotaryAttention(512, 8)
    x = torch.randn(2, 10, 512, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# CoralBackbone
# ---------------------------------------------------------------------------

def test_backbone_output_shape(default_config):
    backbone = CoralBackbone(default_config)
    x = torch.randn(4, 81, 512)  # B=4, L=81, d=512
    out = backbone(x)
    assert out.shape == (4, 81, 512)


def test_backbone_gradient_flow(default_config):
    backbone = CoralBackbone(default_config)
    x = torch.randn(2, 10, 512)
    out = backbone(x)
    loss = out.sum()
    loss.backward()

    for name, param in backbone.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_level_embedding_changes_output(default_config):
    """Backbone should produce different outputs when given different level embeddings."""
    backbone = CoralBackbone(default_config)
    level_emb = LevelEmbedding(2, 512)

    x = torch.randn(2, 10, 512)
    le0 = level_emb(0, x.device)
    le1 = level_emb(1, x.device)

    # Add level embeddings
    x0 = x + le0.unsqueeze(0).unsqueeze(0)
    x1 = x + le1.unsqueeze(0).unsqueeze(0)

    out0 = backbone(x0)
    out1 = backbone(x1)

    # Outputs should differ (level embeddings are different)
    assert not torch.allclose(out0, out1, atol=1e-5), (
        "Level embedding has no effect on backbone output"
    )


def test_timescale_embedding_shape(default_config):
    te = TimescaleEmbedding(512, max_steps=64)
    emb = te(0)
    assert emb.shape == (512,)
    emb5 = te(5)
    assert not torch.allclose(emb, emb5), "Timescale embeddings should differ by step"


def test_level_embedding_shape(default_config):
    le = LevelEmbedding(4, 512)
    emb = le(0, torch.device("cpu"))
    assert emb.shape == (512,)


# ---------------------------------------------------------------------------
# Local attention bias (v4.2)
# ---------------------------------------------------------------------------

def test_backbone_attention_bias_changes_output():
    """Applying a non-zero attention bias should change the backbone output."""
    config = ModelConfig(
        backbone_dim=64, n_heads=4, backbone_layers=2, ffn_expansion=4,
        n_levels=1, level_dims=[64],
        use_local_attention_bias=True,
    )
    backbone = CoralBackbone(config)
    # Manually set bias params to a non-trivial value
    with torch.no_grad():
        backbone.row_bias.fill_(1.0)

    x = torch.randn(2, 9, 64)
    L = 9
    # Use a non-uniform bias: strongly encourage self-attention (diagonal=10, off-diag=-10)
    # This is deliberately asymmetric so softmax outputs differ meaningfully.
    bias = torch.full((L, L), -10.0)
    bias.fill_diagonal_(10.0)

    out_no_bias = backbone(x, attention_bias=None)
    out_with_bias = backbone(x, attention_bias=bias)

    assert not torch.allclose(out_no_bias, out_with_bias, atol=1e-3), (
        "Non-uniform attention bias should change backbone output"
    )


def test_backbone_none_bias_identical_to_no_bias():
    """Passing attention_bias=None should produce the same output as omitting it."""
    config = ModelConfig(
        backbone_dim=64, n_heads=4, backbone_layers=2, ffn_expansion=4,
        n_levels=1, level_dims=[64],
        use_local_attention_bias=False,
    )
    backbone = CoralBackbone(config)
    backbone.eval()

    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        out_none = backbone(x, attention_bias=None)
        out_default = backbone(x)

    assert torch.allclose(out_none, out_default, atol=1e-6), (
        "attention_bias=None should produce identical output to calling without the arg"
    )


def test_grid_attention_masks_structure_9x9():
    """GridAdapter produces structurally correct masks for a 9×9 Sudoku grid."""
    config = ModelConfig(n_levels=1, level_dims=[64], backbone_dim=64, vocab_size=11)
    full_config = CoralConfig(model=config)
    adapter = GridAdapter(full_config, vocab_size=11, grid_height=9, grid_width=9)

    row, col, box = adapter.build_attention_masks()
    L = 81

    assert row.shape == (L, L), f"row mask shape: {row.shape}"
    assert col.shape == (L, L), f"col mask shape: {col.shape}"
    assert box.shape == (L, L), f"box mask shape: {box.shape}"

    # All values must be 0 or 1
    for mask, name in ((row, "row"), (col, "col"), (box, "box")):
        assert set(mask.unique().tolist()).issubset({0.0, 1.0}), (
            f"{name} mask must be binary"
        )

    # Diagonal must be all-ones (each cell is in its own row/col/box)
    assert row.diagonal().all(), "row mask diagonal must be all-ones"
    assert col.diagonal().all(), "col mask diagonal must be all-ones"
    assert box.diagonal().all(), "box mask diagonal must be all-ones"

    # Symmetry: if i shares a row with j, then j shares a row with i
    assert torch.allclose(row, row.T), "row mask must be symmetric"
    assert torch.allclose(col, col.T), "col mask must be symmetric"
    assert torch.allclose(box, box.T), "box mask must be symmetric"

    # Row 0 (cells 0–8) should all share the same row
    assert row[0, :9].sum() == 9, "First 9 cells should all share row 0"
    # Cell 0 and cell 9 should NOT share a row (different rows)
    assert row[0, 9].item() == 0.0, "Cell 0 and cell 9 are in different rows"


# ---------------------------------------------------------------------------
# Smoke test: overfit on single sample
# ---------------------------------------------------------------------------

def test_backbone_smoke_overfit():
    """Backbone should be able to overfit on a single (x, y) pair within 100 steps."""
    config = ModelConfig(backbone_dim=64, n_heads=4, backbone_layers=2, ffn_expansion=4,
                         n_levels=2, level_dims=[64, 32])
    backbone = CoralBackbone(config)
    target = torch.randn(1, 5, 64)
    x = torch.randn(1, 5, 64)
    opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(100):
        out = backbone(x)
        loss = (out - target).pow(2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, (
        f"Backbone failed to overfit: initial_loss={initial_loss:.4f}, "
        f"final_loss={final_loss:.4f}"
    )
