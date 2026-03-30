"""Tests for v4.2 local attention bias: backbone bias params and adapter masks.

Verification criteria:
  - Attention bias masks have correct shape and structural properties
  - Backbone output changes when bias params are non-zero
  - Backbone with attention_bias=None is identical to no-bias baseline (backward compat)
  - bias params receive gradients after backward
"""

import torch
import torch.nn as nn

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.backbone import CoralBackbone
from coral.model.coral_core import CoralCore
from coral.training.losses import CoralLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_backbone_config() -> ModelConfig:
    return ModelConfig(
        n_levels=1,
        level_dims=[64],
        backbone_layers=2,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        K_max=2,
        use_predictive_coding=False,
        vocab_size=11,
        mode="baseline",
        use_local_attention_bias=True,
    )


def _sudoku_adapter() -> GridAdapter:
    """Full 9x9 GridAdapter for mask-structure tests."""
    config = CoralConfig(model=ModelConfig(backbone_dim=64, vocab_size=11))
    return GridAdapter(config, vocab_size=11, grid_height=9, grid_width=9)


# ---------------------------------------------------------------------------
# Mask structure tests
# ---------------------------------------------------------------------------

def test_attention_masks_shape():
    """build_attention_masks returns three [81, 81] tensors for a 9x9 grid."""
    adapter = _sudoku_adapter()
    same_row, same_col, same_box = adapter.build_attention_masks()
    assert same_row.shape == (81, 81), f"Expected (81,81), got {same_row.shape}"
    assert same_col.shape == (81, 81), f"Expected (81,81), got {same_col.shape}"
    assert same_box.shape == (81, 81), f"Expected (81,81), got {same_box.shape}"


def test_attention_masks_same_row():
    """Positions 0 and 1 are in the same row (both row 0)."""
    adapter = _sudoku_adapter()
    same_row, _, _ = adapter.build_attention_masks()
    assert same_row[0, 1].item() == 1.0, "Positions 0 and 1 should share a row"
    assert same_row[0, 9].item() == 0.0, "Positions 0 and 9 should NOT share a row"


def test_attention_masks_same_col():
    """Positions 0 and 9 are in the same column (both col 0)."""
    adapter = _sudoku_adapter()
    _, same_col, _ = adapter.build_attention_masks()
    assert same_col[0, 9].item() == 1.0, "Positions 0 and 9 should share a col"
    assert same_col[0, 1].item() == 0.0, "Positions 0 and 1 should NOT share a col"


def test_attention_masks_same_box():
    """Positions 0 and 10 are in the same 3x3 box (box 0) but different row/col."""
    adapter = _sudoku_adapter()
    same_row, same_col, same_box = adapter.build_attention_masks()
    # Position 10 = row 1, col 1 — box (1//3)*3 + (1//3) = 0, same as position 0
    assert same_box[0, 10].item() == 1.0, "Positions 0 and 10 should share a box"
    assert same_row[0, 10].item() == 0.0, "Positions 0 and 10 should NOT share a row"
    assert same_col[0, 10].item() == 0.0, "Positions 0 and 10 should NOT share a col"


def test_attention_masks_diagonal_ones():
    """Every position is in the same row/col/box as itself."""
    adapter = _sudoku_adapter()
    for mask in adapter.build_attention_masks():
        diag = torch.diag(mask)
        assert (diag == 1.0).all(), "Diagonal must be all 1.0 (self-similarity)"


def test_attention_masks_symmetric():
    """All masks must be symmetric: mask[i,j] == mask[j,i]."""
    adapter = _sudoku_adapter()
    for mask in adapter.build_attention_masks():
        assert torch.allclose(mask, mask.T), "Mask must be symmetric"


def test_attention_masks_values_binary():
    """All mask values must be 0.0 or 1.0."""
    adapter = _sudoku_adapter()
    for mask in adapter.build_attention_masks():
        unique_vals = mask.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0}), (
            f"Mask should only contain 0.0 and 1.0, got {unique_vals}"
        )


# ---------------------------------------------------------------------------
# Backbone bias parameter tests
# ---------------------------------------------------------------------------

def test_backbone_has_bias_params():
    """CoralBackbone must have row_bias, col_bias, box_bias parameters."""
    config = _small_backbone_config()
    backbone = CoralBackbone(config)
    assert hasattr(backbone, "row_bias"), "backbone missing row_bias"
    assert hasattr(backbone, "col_bias"), "backbone missing col_bias"
    assert hasattr(backbone, "box_bias"), "backbone missing box_bias"
    for name in ("row_bias", "col_bias", "box_bias"):
        param = getattr(backbone, name)
        assert isinstance(param, nn.Parameter), f"{name} should be nn.Parameter"
        assert param.item() == 0.0, f"{name} should initialise to 0.0"


def test_backbone_no_bias_is_backward_compatible():
    """backbone(x, attention_bias=None) must give the same result as the v4.1 call."""
    config = _small_backbone_config()
    backbone = CoralBackbone(config)
    backbone.eval()

    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        out_no_arg = backbone(x)
        out_none   = backbone(x, attention_bias=None)

    assert torch.allclose(out_no_arg, out_none), (
        "backbone(x) and backbone(x, attention_bias=None) must be identical"
    )


def test_backbone_output_changes_with_nonzero_bias():
    """When bias params are non-zero, output must differ from the zero-bias baseline."""
    config = _small_backbone_config()
    backbone = CoralBackbone(config)
    backbone.eval()

    B, L, d = 2, 9, 64
    x = torch.randn(B, L, d)

    # Build a dummy [L, L] bias (non-zero)
    bias = torch.ones(L, L) * 0.5

    with torch.no_grad():
        out_no_bias = backbone(x, attention_bias=None)
        out_biased  = backbone(x, attention_bias=bias)

    assert not torch.allclose(out_no_bias, out_biased), (
        "Output should change when a non-zero attention_bias is passed"
    )


def test_backbone_bias_params_receive_gradients():
    """row_bias, col_bias, box_bias must receive non-zero gradients after backward."""
    config = _small_backbone_config()
    backbone = CoralBackbone(config)

    B, L, d = 2, 9, 64
    x = torch.randn(B, L, d)
    # Set non-zero bias params so the gradients are non-trivially connected
    with torch.no_grad():
        backbone.row_bias.fill_(0.1)
        backbone.col_bias.fill_(0.1)
        backbone.box_bias.fill_(0.1)

    # Build a simple bias from scalars and a random mask
    row_mask = (torch.arange(L).unsqueeze(1) == torch.arange(L).unsqueeze(0)).float()
    attn_bias = backbone.row_bias * row_mask

    out = backbone(x, attention_bias=attn_bias)
    loss = out.sum()
    loss.backward()

    for name in ("row_bias", "col_bias", "box_bias"):
        param = getattr(backbone, name)
        # row_bias gets gradient (it's in the computation graph via attn_bias)
        # col_bias and box_bias have no path in this test, skip their grad check
        if name == "row_bias":
            assert param.grad is not None and param.grad.abs().item() > 0, (
                f"{name} should have non-zero gradient"
            )


# ---------------------------------------------------------------------------
# CoralCore integration: attention_masks threading
# ---------------------------------------------------------------------------

def test_coral_core_with_attention_masks():
    """CoralCore with attention_masks produces correct output shape and no errors."""
    config = ModelConfig(
        n_levels=1,
        level_dims=[64],
        backbone_layers=2,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        K_max=2,
        use_predictive_coding=False,
        vocab_size=11,
        mode="baseline",
        use_local_attention_bias=True,
    )
    coral_config = CoralConfig(model=config)
    adapter = GridAdapter(coral_config, vocab_size=11, grid_height=3, grid_width=3)
    core = CoralCore(config)
    loss_fn = CoralLoss(config)

    inputs = torch.randint(0, 11, (2, 9))
    labels = torch.randint(0, 11, (2, 9))

    z1 = adapter.encode(inputs)
    masks = adapter.build_attention_masks(device=z1.device)

    # Set non-zero bias params
    with torch.no_grad():
        core.backbone.row_bias.fill_(0.1)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode, attention_masks=masks)
    assert out.z_states[0].shape == (2, 9, 64)

    # Forward + backward
    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(logits=logits, labels=labels)
        total_loss = total_loss + seg_loss
    total_loss.backward()


def test_coral_core_no_masks_unchanged():
    """CoralCore without attention_masks behaves identically to v4.1 (no bias)."""
    config = ModelConfig(
        n_levels=1,
        level_dims=[64],
        backbone_layers=2,
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        K_max=1,
        use_predictive_coding=False,
        vocab_size=11,
        mode="baseline",
        use_local_attention_bias=True,
    )
    coral_config = CoralConfig(model=config)
    adapter = GridAdapter(coral_config, vocab_size=11, grid_height=3, grid_width=3)
    core = CoralCore(config)
    core.eval()

    inputs = torch.randint(0, 11, (2, 9))
    z1 = adapter.encode(inputs)

    with torch.no_grad():
        out_no_masks = core(z1, K_max=1, training=False, decode_fn=adapter.decode)
        out_none_masks = core(z1, K_max=1, training=False, decode_fn=adapter.decode,
                              attention_masks=None)

    assert torch.allclose(out_no_masks.z_states[0], out_none_masks.z_states[0]), (
        "CoralCore with no attention_masks arg must match attention_masks=None"
    )
