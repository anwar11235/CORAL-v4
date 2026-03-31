"""Tests for Session 7: config loading, model output shapes, annealing.

Verification criteria:
  - All 5 phase config YAML files parse into valid CoralConfig without error
  - mode="full" with use_crystallisation=True produces correct output shapes
  - get_effective_lambda_amort returns correct values at boundary points
"""

import os
import sys

import pytest
import torch

from coral.adapters.grid import GridAdapter
from coral.config import CoralConfig, ModelConfig, TrainingConfig, DataConfig, WandbConfig
from coral.model.coral_core import CoralCore
from coral.training.annealing import get_effective_lambda_amort

# Path to scripts directory
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Try importing Hydra/OmegaConf for YAML loading (skip if not installed)
# ---------------------------------------------------------------------------

omegaconf = pytest.importorskip("omegaconf", reason="omegaconf not installed")
OmegaConf = omegaconf.OmegaConf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> CoralConfig:
    """Load a YAML config file and return a CoralConfig dataclass."""
    path = os.path.join(_CONFIGS_DIR, filename)
    raw = OmegaConf.load(path)
    model_cfg = ModelConfig(**dict(raw.model))
    train_cfg = TrainingConfig(**dict(raw.training))
    data_cfg = DataConfig(**dict(raw.data))
    wandb_cfg = WandbConfig(**dict(raw.wandb))
    return CoralConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        wandb=wandb_cfg,
        seed=raw.seed,
        device=raw.device,
        experiment_name=raw.experiment_name,
    )




# ---------------------------------------------------------------------------
# Test 1: YAML config loading
# ---------------------------------------------------------------------------

_PHASE_CONFIGS = [
    "phase3a_crystal_simple.yaml",
    "phase3b_crystal_multihead.yaml",
    "phase3c_crystal_decrystal.yaml",
    "phase4_amort_with_crystal.yaml",
    "phase4_amort_no_crystal.yaml",
]


@pytest.mark.parametrize("filename", _PHASE_CONFIGS)
def test_config_loads_without_error(filename):
    """Each phase config should parse into a valid CoralConfig."""
    config = _load_yaml(filename)
    assert isinstance(config, CoralConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)


@pytest.mark.parametrize("filename", _PHASE_CONFIGS)
def test_config_has_new_training_fields(filename):
    """All configs must expose the new TrainingConfig fields from Session 7."""
    config = _load_yaml(filename)
    # codebook_init_from: None or a str path
    assert config.training.codebook_init_from is None or isinstance(
        config.training.codebook_init_from, str
    )
    # amort annealing: ints
    assert isinstance(config.training.amort_anneal_start, int)
    assert isinstance(config.training.amort_anneal_end, int)


def test_phase3a_monolithic_codebook():
    """phase3a should have codebook_heads=1, entries=256, lambda_dis=0."""
    config = _load_yaml("phase3a_crystal_simple.yaml")
    assert config.model.codebook_heads == 1
    assert config.model.codebook_entries_per_head == 256
    assert config.model.lambda_dis == 0.0
    assert config.model.use_crystallisation is True
    assert config.model.mode == "full"


def test_phase3b_multihead_codebook():
    """phase3b should have codebook_heads=8, entries=32, lambda_dis=0.01."""
    config = _load_yaml("phase3b_crystal_multihead.yaml")
    assert config.model.codebook_heads == 8
    assert config.model.codebook_entries_per_head == 32
    assert config.model.lambda_dis == pytest.approx(0.01)
    assert config.model.use_crystallisation is True
    assert config.model.mode == "full"


def test_phase4_amort_with_crystal():
    """phase4_amort_with_crystal should have use_amort=True and annealing steps."""
    config = _load_yaml("phase4_amort_with_crystal.yaml")
    assert config.model.use_amort is True
    assert config.model.lambda_amort == pytest.approx(0.01)
    assert config.model.use_crystallisation is True
    assert config.training.amort_anneal_start == 2000
    assert config.training.amort_anneal_end == 8000


def test_phase4_amort_no_crystal():
    """phase4_amort_no_crystal should have use_amort=True but use_crystallisation=False."""
    config = _load_yaml("phase4_amort_no_crystal.yaml")
    assert config.model.use_amort is True
    assert config.model.lambda_amort == pytest.approx(0.01)
    assert config.model.use_crystallisation is False
    assert config.training.amort_anneal_start == 2000
    assert config.training.amort_anneal_end == 8000


# ---------------------------------------------------------------------------
# Test 2: mode="full" model output shapes
# ---------------------------------------------------------------------------

def _make_full_mode_config():
    """Minimal config: mode='full', n_levels=1, crystallisation ON."""
    return ModelConfig(
        n_levels=1,
        level_dims=[64],
        backbone_dim=64,
        n_heads=4,
        d_k=16,
        ffn_expansion=2,
        timescale_base=2,
        K_max=4,
        use_predictive_coding=False,
        use_crystallisation=True,
        codebook_heads=4,
        codebook_entries_per_head=8,
        epsilon_min=0.01,
        lambda_pred=0.001,
        vocab_size=10,
        mode="full",
    )


def test_full_mode_output_shape():
    """CoralCore in mode='full' should return z_states[0] with correct shape."""
    config = _make_full_mode_config()
    core = CoralCore(config)
    core.eval()

    B, L, d = 2, 9, 64
    z1 = torch.randn(B, L, d)
    with torch.no_grad():
        out = core(z1, K_max=4, training=False, decode_fn=None)

    assert out.z_states[0].shape == (B, L, d)
    assert out.num_segments > 0


def test_full_mode_crystal_stats_populated():
    """In mode='full' with use_crystallisation=True, crystal_stats should be non-empty."""
    config = _make_full_mode_config()
    core = CoralCore(config)
    core.eval()

    B, L, d = 2, 9, 64
    z1 = torch.randn(B, L, d)
    with torch.no_grad():
        out = core(z1, K_max=4, training=False, decode_fn=None)

    # crystal_stats is a list (one entry per segment)
    assert isinstance(out.crystal_stats, list)
    assert len(out.crystal_stats) == out.num_segments


def test_full_mode_backward():
    """mode='full' forward + backward should not raise."""
    config = _make_full_mode_config()
    full_config = CoralConfig(model=config, device="cpu")
    full_config.training.precision = "float32"

    adapter = GridAdapter(full_config, vocab_size=10, grid_height=3, grid_width=3)
    core = CoralCore(config)

    adapter.train()
    core.train()

    inputs = torch.randint(0, 10, (2, 9))
    z1 = adapter.encode(inputs)

    from coral.training.losses import CoralLoss
    loss_fn = CoralLoss(config)

    out = core(z1, K_max=2, training=True, decode_fn=adapter.decode)
    labels = torch.randint(1, 10, (2, 9))

    total_loss = torch.tensor(0.0)
    for i, logits in enumerate(out.all_logits):
        seg_loss, _ = loss_fn(
            logits=logits,
            labels=labels,
            commitment_loss=out.commit_losses[i] if out.commit_losses else None,
            disentanglement_loss=out.dis_losses[i] if out.dis_losses else None,
        )
        total_loss = total_loss + seg_loss

    total_loss.backward()


# ---------------------------------------------------------------------------
# Test 3: get_effective_lambda_amort
# ---------------------------------------------------------------------------

def test_annealing_before_start():
    """Before anneal_start, effective lambda should be 0."""
    assert get_effective_lambda_amort(step=0, base=0.01, anneal_start=2000, anneal_end=8000) == pytest.approx(0.0)
    assert get_effective_lambda_amort(step=1999, base=0.01, anneal_start=2000, anneal_end=8000) == pytest.approx(0.0)


def test_annealing_at_midpoint():
    """At midpoint of annealing range, effective lambda should be base/2."""
    mid = (2000 + 8000) // 2  # 5000
    result = get_effective_lambda_amort(step=mid, base=0.01, anneal_start=2000, anneal_end=8000)
    assert result == pytest.approx(0.005, abs=1e-6)


def test_annealing_at_end():
    """At or after anneal_end, effective lambda should equal base."""
    assert get_effective_lambda_amort(step=8000, base=0.01, anneal_start=2000, anneal_end=8000) == pytest.approx(0.01)
    assert get_effective_lambda_amort(step=10000, base=0.01, anneal_start=2000, anneal_end=8000) == pytest.approx(0.01)


def test_annealing_disabled_when_anneal_end_zero():
    """When anneal_end=0, no annealing — returns base immediately."""
    assert get_effective_lambda_amort(step=0, base=0.01, anneal_start=0, anneal_end=0) == pytest.approx(0.01)
    assert get_effective_lambda_amort(step=1000, base=0.01, anneal_start=0, anneal_end=0) == pytest.approx(0.01)


def test_annealing_zero_base():
    """When base=0.0, always returns 0 regardless of step."""
    assert get_effective_lambda_amort(step=5000, base=0.0, anneal_start=0, anneal_end=10000) == pytest.approx(0.0)
