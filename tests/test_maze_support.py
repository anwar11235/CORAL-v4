"""Tests for Maze-Hard support (Session A).

Verifies:
  - GridAdapter works with 30x30 grid and vocab_size=6
  - Maze config loads correctly with seq_len=900 and vocab_size=6
  - CoralCore forward pass works with maze dimensions
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.model.coral_core import CoralCore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _make_maze_model_config(**overrides) -> ModelConfig:
    """Minimal ModelConfig for a 30x30 maze smoke test."""
    defaults = dict(
        n_levels=1,
        level_dims=[512],
        backbone_dim=512,
        backbone_layers=2,
        n_heads=8,
        d_k=64,
        ffn_expansion=4,
        timescale_base=3,
        use_predictive_coding=False,
        use_crystallisation=False,
        use_amort=False,
        use_local_attention_bias=False,
        K_max=2,
        inner_steps_override=3,
        grad_inner_steps=None,
        vocab_size=6,
        embed_scale=True,
        use_consolidation_step=False,
        mode="baseline",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_coral_config(model_cfg: ModelConfig) -> CoralConfig:
    return CoralConfig(
        model=model_cfg,
        training=TrainingConfig(),
        data=DataConfig(
            dataset="maze_30x30_hard",
            seq_len=900,
            grid_height=30,
            grid_width=30,
        ),
        wandb=WandbConfig(),
    )


# ---------------------------------------------------------------------------
# Test 1: GridAdapter 30x30 encode
# ---------------------------------------------------------------------------

def test_grid_adapter_30x30():
    """GridAdapter encode produces correct shape for 30x30 maze input."""
    model_cfg = _make_maze_model_config()
    config = _make_coral_config(model_cfg)

    adapter = GridAdapter(config, vocab_size=6, grid_height=30, grid_width=30)
    adapter.eval()

    B = 2
    x = torch.randint(0, 6, (B, 900))  # [B, 900] tokens in [0,5]
    with torch.no_grad():
        z = adapter.encode(x)

    assert z.shape == (B, 900, 512), f"Expected (2, 900, 512), got {z.shape}"
    assert z.isfinite().all(), "encode output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 2: GridAdapter 30x30 decode
# ---------------------------------------------------------------------------

def test_grid_adapter_decode_30x30():
    """GridAdapter decode produces correct shape for 30x30 maze embeddings."""
    model_cfg = _make_maze_model_config()
    config = _make_coral_config(model_cfg)

    adapter = GridAdapter(config, vocab_size=6, grid_height=30, grid_width=30)
    adapter.eval()

    B = 2
    z = torch.randn(B, 900, 512)
    with torch.no_grad():
        logits = adapter.decode(z)

    assert logits.shape == (B, 900, 6), f"Expected (2, 900, 6), got {logits.shape}"
    assert logits.isfinite().all(), "decode output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3: Maze config loads correctly
# ---------------------------------------------------------------------------

def test_maze_config_loads():
    """phase1_maze_baseline.yaml loads into a valid CoralConfig."""
    omegaconf = pytest.importorskip("omegaconf", reason="omegaconf not installed")
    OmegaConf = omegaconf.OmegaConf

    path = os.path.join(_CONFIGS_DIR, "phase1_maze_baseline.yaml")
    assert os.path.exists(path), f"Config file not found: {path}"

    raw = OmegaConf.load(path)
    model_cfg = ModelConfig(**dict(raw.model))
    train_cfg = TrainingConfig(**dict(raw.training))
    data_cfg = DataConfig(**dict(raw.data))
    wandb_cfg = WandbConfig(**dict(raw.wandb))
    config = CoralConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        wandb=wandb_cfg,
        seed=raw.seed,
        device=raw.device,
        experiment_name=raw.experiment_name,
    )

    assert config.data.seq_len == 900
    assert config.model.vocab_size == 6
    assert config.data.grid_height == 30
    assert config.data.grid_width == 30
    assert config.data.dataset == "maze_30x30_hard"
    assert config.model.use_local_attention_bias is False
    assert config.model.mode == "baseline"


# ---------------------------------------------------------------------------
# Test 4: CoralCore forward pass with maze dimensions
# ---------------------------------------------------------------------------

def test_coral_core_forward_maze_shapes():
    """CoralCore forward pass works with maze dimensions (vocab=6, seq_len=900)."""
    model_cfg = _make_maze_model_config(K_max=2, inner_steps_override=3)
    config = _make_coral_config(model_cfg)

    core = CoralCore(model_cfg)
    core.eval()

    B = 2
    z1 = torch.randn(B, 900, 512)

    with torch.no_grad():
        output = core(z1, K_max=2, training=False, decode_fn=None)

    assert output.z_states[0].shape == (B, 900, 512), (
        f"Expected z_states[0] shape (2, 900, 512), got {output.z_states[0].shape}"
    )
    assert output.z_states[0].isfinite().all(), "CoralCore output contains NaN or Inf"
