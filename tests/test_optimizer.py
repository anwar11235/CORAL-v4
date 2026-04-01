"""Tests for coral/training/optimizer.py — parameter group split.

Verifies that build_optimizer correctly separates:
  - Embedding weights (token_emb, row_emb, col_emb, level, timescale) → no_decay (WD=0)
  - Bias parameters (1-d tensors) → no_decay (WD=0)
  - Scalar learnable params (cond_gate, row_bias, col_bias, box_bias) → no_decay (WD=0)
  - RMSNorm / LayerNorm parameters → no_decay (WD=0)
  - Backbone linear weights (q_proj, k_proj, etc.) → decay (WD=weight_decay)
  - Decoder linear weight → decay (WD=weight_decay)
"""

import torch
import torch.nn as nn

from coral.config import CoralConfig, ModelConfig
from coral.adapters.grid import GridAdapter
from coral.model.coral_core import CoralCore
from coral.training.optimizer import build_optimizer


def _make_model():
    """Build a small but realistic adapter + core matching phase1 config."""
    model_cfg = ModelConfig(
        n_levels=1, level_dims=[64], backbone_dim=64, n_heads=4, d_k=16,
        ffn_expansion=2, timescale_base=3, K_max=2,
        use_predictive_coding=False, use_crystallisation=False,
        epsilon_min=0.01, lambda_pred=0.001, vocab_size=11, mode="baseline",
        use_local_attention_bias=True,   # enables row_bias, col_bias, box_bias
    )
    full_cfg = CoralConfig(model=model_cfg, device="cpu")
    adapter = GridAdapter(full_cfg, vocab_size=11, grid_height=9, grid_width=9)
    core = CoralCore(model_cfg)
    return nn.ModuleList([adapter, core])


def _get_param_groups(weight_decay: float = 1.0):
    """Return (decay_group_params_set, no_decay_group_params_set, optimizer)."""
    model = _make_model()
    optimizer = build_optimizer(model, lr=7e-5, weight_decay=weight_decay)

    # Map id → (name, param) for lookup
    id_to_name = {id(p): n for n, p in model.named_parameters() if p.requires_grad}

    decay_names = set()
    no_decay_names = set()
    for group in optimizer.param_groups:
        wd = group["weight_decay"]
        for p in group["params"]:
            name = id_to_name.get(id(p), "<unknown>")
            if wd > 0:
                decay_names.add(name)
            else:
                no_decay_names.add(name)

    return decay_names, no_decay_names, optimizer


# ---------------------------------------------------------------------------
# No-decay: embeddings
# ---------------------------------------------------------------------------

def test_token_embedding_in_no_decay():
    """adapter.token_emb.weight must be in the no-decay group."""
    _, no_decay, _ = _get_param_groups()
    matches = [n for n in no_decay if "token_emb" in n]
    assert len(matches) > 0, (
        f"token_emb not found in no_decay group. no_decay={sorted(no_decay)}"
    )


def test_row_embedding_in_no_decay():
    """adapter.row_emb.weight must be in the no-decay group."""
    _, no_decay, _ = _get_param_groups()
    matches = [n for n in no_decay if "row_emb" in n]
    assert len(matches) > 0, f"row_emb not in no_decay. no_decay={sorted(no_decay)}"


def test_col_embedding_in_no_decay():
    """adapter.col_emb.weight must be in the no-decay group."""
    _, no_decay, _ = _get_param_groups()
    matches = [n for n in no_decay if "col_emb" in n]
    assert len(matches) > 0, f"col_emb not in no_decay. no_decay={sorted(no_decay)}"


def test_level_embedding_in_no_decay():
    """core.level_emb.embeddings.weight must be in the no-decay group."""
    _, no_decay, _ = _get_param_groups()
    matches = [n for n in no_decay if "level_emb" in n]
    assert len(matches) > 0, f"level_emb not in no_decay. no_decay={sorted(no_decay)}"


# ---------------------------------------------------------------------------
# No-decay: biases, scalars, norms
# ---------------------------------------------------------------------------

def test_bias_parameters_in_no_decay():
    """All bias parameters (ndim=1, name ends with .bias) must be in no_decay."""
    _, no_decay, _ = _get_param_groups()
    bias_names = [n for n in no_decay if n.endswith(".bias")]
    assert len(bias_names) > 0, "No bias parameters found in no_decay group"


def test_row_col_box_bias_scalars_in_no_decay():
    """backbone.row_bias, col_bias, box_bias (shape [1], ndim=1) must be no_decay."""
    _, no_decay, _ = _get_param_groups()
    for scalar_name in ("row_bias", "col_bias", "box_bias"):
        matches = [n for n in no_decay if scalar_name in n]
        assert len(matches) > 0, (
            f"{scalar_name} not found in no_decay group. no_decay={sorted(no_decay)}"
        )


def test_cond_gate_in_no_decay():
    """core.cond_gate (shape [1], ndim=1) must be in the no-decay group."""
    _, no_decay, _ = _get_param_groups()
    matches = [n for n in no_decay if "cond_gate" in n]
    assert len(matches) > 0, (
        f"cond_gate not found in no_decay group. no_decay={sorted(no_decay)}"
    )


def test_norm_parameters_in_no_decay():
    """RMSNorm / LayerNorm parameters (name contains 'norm') must be no_decay."""
    _, no_decay, _ = _get_param_groups()
    norm_names = [n for n in no_decay if "norm" in n.lower()]
    # input_norm in adapter, norm1/norm2 in each TransformerLayer
    assert len(norm_names) > 0, (
        f"No norm parameters found in no_decay group. no_decay={sorted(no_decay)}"
    )


# ---------------------------------------------------------------------------
# Decay: backbone linear weights, decoder
# ---------------------------------------------------------------------------

def test_backbone_linear_weights_in_decay():
    """Backbone q_proj, k_proj, v_proj, out_proj weights must be in decay group."""
    decay, _, _ = _get_param_groups()
    proj_names = [n for n in decay if "proj" in n and "weight" in n]
    assert len(proj_names) > 0, (
        f"No projection weights found in decay group. decay={sorted(decay)}"
    )


def test_decoder_weight_in_decay():
    """adapter.decoder.weight (linear) must be in the decay group."""
    decay, _, _ = _get_param_groups()
    matches = [n for n in decay if "decoder" in n and "weight" in n]
    assert len(matches) > 0, (
        f"decoder.weight not found in decay group. decay={sorted(decay)}"
    )


def test_ffn_weights_in_decay():
    """SwiGLU gate_proj, up_proj, down_proj weights must be in decay group."""
    decay, _, _ = _get_param_groups()
    ffn_names = [n for n in decay if any(k in n for k in ("gate_proj", "up_proj", "down_proj"))]
    assert len(ffn_names) > 0, (
        f"FFN weights not found in decay group. decay={sorted(decay)}"
    )


# ---------------------------------------------------------------------------
# Mutual exclusion and coverage
# ---------------------------------------------------------------------------

def test_no_parameter_in_both_groups():
    """No parameter should appear in both decay and no_decay groups."""
    decay, no_decay, _ = _get_param_groups()
    overlap = decay & no_decay
    assert len(overlap) == 0, f"Parameters in both groups: {overlap}"


def test_all_params_covered():
    """Every requires_grad parameter must appear in exactly one group."""
    model = _make_model()
    all_names = {n for n, p in model.named_parameters() if p.requires_grad}
    decay, no_decay, _ = _get_param_groups()
    covered = decay | no_decay
    missing = all_names - covered
    assert len(missing) == 0, f"Parameters not in any group: {missing}"


# ---------------------------------------------------------------------------
# Weight-decay values are correctly set
# ---------------------------------------------------------------------------

def test_decay_group_has_correct_wd():
    """The decay group must have weight_decay == weight_decay argument."""
    model = _make_model()
    optimizer = build_optimizer(model, lr=7e-5, weight_decay=1.0)
    wd_values = [g["weight_decay"] for g in optimizer.param_groups if g["params"]]
    assert 1.0 in wd_values, f"Expected WD=1.0 in some group. Got: {wd_values}"


def test_no_decay_group_has_zero_wd():
    """The no-decay group must have weight_decay == 0.0."""
    model = _make_model()
    optimizer = build_optimizer(model, lr=7e-5, weight_decay=1.0)
    wd_values = [g["weight_decay"] for g in optimizer.param_groups if g["params"]]
    assert 0.0 in wd_values, f"Expected WD=0.0 in some group. Got: {wd_values}"
