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

from coral.config import ModelConfig
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
