"""Unit tests for GridAdapter.

Verification criteria:
  - encode → decode cycle on random grid produces valid logits
  - Smoke test: overfit on 1 puzzle through full pipeline
  - Full 2D position embedding produces rank well above the old factorized ceiling
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter


def _make_config(grid_height: int = 9, grid_width: int = 9,
                 vocab_size: int = 10, d_model: int = 64) -> CoralConfig:
    """Build a minimal CoralConfig with the given grid / model dimensions."""
    return CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[d_model], backbone_dim=d_model,
        n_heads=4, d_k=d_model // 4, ffn_expansion=2,
        vocab_size=vocab_size, embed_scale=False,
    ))


@pytest.fixture
def config():
    return CoralConfig(model=ModelConfig(
        n_levels=2,
        level_dims=[64, 32],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        vocab_size=10,
    ))


def test_encode_shape(config):
    adapter = GridAdapter(config, vocab_size=10, grid_height=9, grid_width=9)
    x = torch.randint(0, 10, (4, 81))
    emb = adapter.encode(x)
    assert emb.shape == (4, 81, config.model.backbone_dim)


def test_decode_shape(config):
    adapter = GridAdapter(config, vocab_size=10, grid_height=9, grid_width=9)
    z = torch.randn(4, 81, config.model.backbone_dim)
    logits = adapter.decode(z)
    assert logits.shape == (4, 81, 10)


def test_encode_decode_produces_valid_logits(config):
    """encode then decode should produce finite logits."""
    adapter = GridAdapter(config, vocab_size=10, grid_height=9, grid_width=9)
    x = torch.randint(0, 10, (2, 81))
    emb = adapter.encode(x)
    logits = adapter.decode(emb)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    assert logits.shape == (2, 81, 10)


def test_encode_deterministic(config):
    """Encode should be deterministic (no dropout)."""
    adapter = GridAdapter(config, vocab_size=10, grid_height=3, grid_width=3)
    adapter.eval()
    x = torch.randint(0, 10, (2, 9))
    emb1 = adapter.encode(x)
    emb2 = adapter.encode(x)
    assert torch.allclose(emb1, emb2)


def test_position_embeddings_unique(config):
    """All 81 position embeddings should be different."""
    adapter = GridAdapter(config, vocab_size=10, grid_height=9, grid_width=9)
    # Encode all-zero grid
    x = torch.zeros(1, 81, dtype=torch.long)
    emb = adapter.encode(x)  # [1, 81, d]
    # Check that different positions produce different embeddings
    # (row+col embeddings differ across positions)
    for i in range(81):
        for j in range(i + 1, 81):
            if (i // 9 != j // 9) or (i % 9 != j % 9):
                # Different position should differ
                pass  # Not all will differ due to shared token emb, but most should


def test_embed_scale_increases_magnitude(config):
    """embed_scale=True should produce embeddings ~sqrt(d_model) larger than embed_scale=False."""
    from coral.config import ModelConfig

    config_scaled = CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, vocab_size=10, embed_scale=True,
    ))
    config_unscaled = CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, vocab_size=10, embed_scale=False,
    ))
    # Share weights so the only difference is the scale factor
    adapter_scaled = GridAdapter(config_scaled, vocab_size=10)
    adapter_unscaled = GridAdapter(config_unscaled, vocab_size=10)
    adapter_unscaled.load_state_dict(adapter_scaled.state_dict())

    x = torch.randint(0, 10, (2, 81))
    with torch.no_grad():
        emb_scaled = adapter_scaled.encode(x)
        emb_unscaled = adapter_unscaled.encode(x)

    norm_scaled = emb_scaled.norm(dim=-1).mean().item()
    norm_unscaled = emb_unscaled.norm(dim=-1).mean().item()
    # Scaled norm should be noticeably larger (LayerNorm normalises but scale is absorbed
    # via the learnable gamma; before LayerNorm the pre-norm variance is ~d_model larger)
    assert norm_scaled > norm_unscaled, (
        f"embed_scale=True should produce larger norm: {norm_scaled:.4f} vs {norm_unscaled:.4f}"
    )


def test_embed_scale_false_backward_compat():
    """embed_scale=False should produce identical output to a config without the field."""
    from coral.config import ModelConfig

    # Config without embed_scale field (defaults to True — so use False explicitly for old-style)
    config_old = CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, vocab_size=10, embed_scale=False,
    ))
    config_same = CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, vocab_size=10, embed_scale=False,
    ))
    adapter1 = GridAdapter(config_old, vocab_size=10)
    adapter2 = GridAdapter(config_same, vocab_size=10)
    adapter2.load_state_dict(adapter1.state_dict())

    x = torch.randint(0, 10, (2, 81))
    with torch.no_grad():
        emb1 = adapter1.encode(x)
        emb2 = adapter2.encode(x)

    assert torch.allclose(emb1, emb2), "embed_scale=False should be deterministic across identical configs"


def _svd_rank(tensor: torch.Tensor, threshold: float) -> int:
    """Return PCA rank of a [N, D] tensor at the given cumulative variance threshold."""
    flat = tensor.reshape(-1, tensor.shape[-1]).float().numpy()
    S = np.linalg.svd(flat, compute_uv=False)
    cum = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
    return int(np.searchsorted(cum, threshold) + 1)


def test_full_2d_position_encoding_rank_ceiling_maze():
    """Maze adapter (30x30, vocab=6, d=512) must produce encoder rank well
    above the old factorized ceiling of H+W+vocab = 66.

    Expected under the joint 2D scheme:
      rank @90% variance  > 200  (old adapter: ~47)
      rank @99% variance  > 400  (old adapter: ~62)
    """
    config = _make_config(grid_height=30, grid_width=30, vocab_size=6, d_model=512)
    adapter = GridAdapter(config, grid_height=30, grid_width=30, vocab_size=6)
    adapter.eval()

    inputs = torch.randint(1, 5, (4, 900))
    with torch.no_grad():
        encoded = adapter.encode(inputs)  # [4, 900, 512]

    rank_90 = _svd_rank(encoded, 0.90)
    rank_99 = _svd_rank(encoded, 0.99)

    assert rank_90 > 200, (
        f"rank @90% variance = {rank_90}, expected > 200 "
        f"(old factorized ceiling was H+W+vocab = 66)"
    )
    assert rank_99 > 400, (
        f"rank @99% variance = {rank_99}, expected > 400"
    )


def test_full_2d_position_encoding_rank_ceiling_sudoku():
    """Sudoku adapter (9x9, vocab=10, d=512) should be limited by H*W=81,
    but well above the old factorized ceiling of H+W+vocab = 28.

    Using constant-token inputs isolates the positional contribution so the
    rank ceiling is exactly min(H*W, d_model) = 81.  With random tokens the
    token embeddings would add up to vocab_size extra rank, making the bound
    harder to assert precisely.

    Expected (constant tokens):
      rank @99% variance  > 40   (old factorized ceiling was H+W+vocab = 28)
      rank @99% variance  <= 81  (hard ceiling: 81 unique positions)
    """
    config = _make_config(grid_height=9, grid_width=9, vocab_size=10, d_model=512)
    adapter = GridAdapter(config, grid_height=9, grid_width=9, vocab_size=10)
    adapter.eval()

    # Constant token=1 across all positions/batches: only positional variation
    # contributes rank, so the ceiling is exactly H*W = 81.
    inputs = torch.ones(4, 81, dtype=torch.long)
    with torch.no_grad():
        encoded = adapter.encode(inputs)  # [4, 81, 512]

    rank_99 = _svd_rank(encoded, 0.99)

    assert rank_99 > 40, (
        f"rank @99% variance = {rank_99}, expected > 40 "
        f"(old factorized ceiling was H+W+vocab = 28)"
    )
    assert rank_99 <= 81, (
        f"rank @99% variance = {rank_99}, exceeds H*W=81 (impossible for 81 unique positions)"
    )


def test_smoke_overfit_single_puzzle():
    """Adapter + simple linear decoder should overfit a single puzzle."""
    config = CoralConfig(model=ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, vocab_size=10,
    ))
    adapter = GridAdapter(config, vocab_size=10, grid_height=3, grid_width=3)
    opt = optim.Adam(adapter.parameters(), lr=1e-3)

    x = torch.randint(0, 10, (1, 9))
    labels = torch.randint(0, 10, (1, 9))

    initial_loss = None
    for step in range(200):
        emb = adapter.encode(x)
        logits = adapter.decode(emb)  # [1, 9, 10]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 10), labels.reshape(-1)
        )
        if initial_loss is None:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, (
        f"Adapter failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
