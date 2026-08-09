"""Unit tests for the pure-logic helpers in scripts/train/train_grpo.py.

Covers lora_movement -- the live "is the policy actually changing?" metric fed into
TRL's own _metrics dict (and so into wandb + trainer_state.json). GPU/model-free: a
handful of nn.Parameters named the way PEFT names them is enough.

The property that matters is the Gram-trick identity
    ||B @ A||_F == sqrt(sum((B^T B) * (A A^T)))
which lets the callback run every step without materializing dW (a full out x in
matrix per module -- ~134 MB fp32 for one Gemma MLP projection).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "gemma"))

import train_grpo as t  # noqa: E402

torch = pytest.importorskip("torch")


class FakeLoraModel:
    """Just enough of an nn.Module: PEFT-shaped parameter names."""

    def __init__(self, mods, r=4, out=8, inp=6, b_scale=1.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self._params = []
        for m in mods:
            a = torch.randn(r, inp, generator=g)
            b = torch.randn(out, r, generator=g) * b_scale
            self._params.append((f"base_model.model.{m}.lora_A.default.weight", a))
            self._params.append((f"base_model.model.{m}.lora_B.default.weight", b))

    def named_parameters(self):
        return iter(self._params)


def test_returns_empty_without_lora_params():
    class Bare:
        def named_parameters(self):
            return iter([("model.layers.0.mlp.weight", torch.randn(4, 4))])

    assert t.lora_movement(Bare(), scaling=2.0) == {}


def test_b_norm_is_zero_at_peft_init():
    """PEFT initializes lora_B to exactly zero -- the property the metric relies on."""
    m = FakeLoraModel(["l0.q_proj", "l1.q_proj"], b_scale=0.0)
    out = t.lora_movement(m, scaling=2.0)
    assert out["lora/b_norm_median"] == 0.0
    assert out["lora/b_norm_max"] == 0.0
    assert out["lora/delta_w_norm_median"] == 0.0
    assert out["lora/b_norm_vs_sft"] == 0.0


def test_delta_w_matches_explicit_product():
    """The Gram trick must equal scaling * ||B @ A||_F computed the naive way."""
    m = FakeLoraModel(["l0.q_proj"], seed=7)
    scaling = 2.0
    out = t.lora_movement(m, scaling=scaling)
    params = dict(m.named_parameters())
    a = params["base_model.model.l0.q_proj.lora_A.default.weight"]
    b = params["base_model.model.l0.q_proj.lora_B.default.weight"]
    expected = scaling * (b @ a).norm().item()
    assert out["lora/delta_w_norm_median"] == pytest.approx(expected, rel=1e-5)


def test_b_norm_median_and_max_over_modules():
    m = FakeLoraModel(["l0.q_proj", "l1.q_proj", "l2.q_proj"], seed=3)
    out = t.lora_movement(m, scaling=2.0)
    params = dict(m.named_parameters())
    norms = sorted(v.norm().item() for k, v in params.items() if ".lora_B" in k)
    assert out["lora/b_norm_median"] == pytest.approx(norms[1], rel=1e-6)
    assert out["lora/b_norm_max"] == pytest.approx(norms[-1], rel=1e-6)


def test_vs_sft_ratio_uses_the_measured_reference():
    m = FakeLoraModel(["l0.q_proj"], seed=1)
    out = t.lora_movement(m, scaling=2.0)
    assert t.SFT_REFERENCE_B_NORM == 0.41
    assert out["lora/b_norm_vs_sft"] == pytest.approx(
        out["lora/b_norm_median"] / t.SFT_REFERENCE_B_NORM, rel=1e-9
    )


def test_unpaired_b_without_its_a_is_skipped():
    """A B with no matching A cannot contribute a dW term; it must not crash."""
    m = FakeLoraModel(["l0.q_proj"], seed=5)
    m._params = [p for p in m._params if ".lora_A" not in p[0]]
    assert t.lora_movement(m, scaling=2.0) == {}
