import cubbyllm.core.config as mod
from cubbyllm.core.config import SEED, CubbyConfig
from cubbyllm.core.protocols import Wiring


def test_config_and_d_vsa():
    cfg = CubbyConfig(d_model=512, n_layers=8, ctx_dim=64, vocab_core=131072)
    assert cfg.d_vsa == cfg.block_k * cfg.block_l == 10240
    assert cfg.vocab_dynamic is False


def test_seed_is_canonical():
    assert SEED == 0xC0DEB00C


def test_wiring_declared():
    assert mod.__wiring__ in (Wiring.WIRED, Wiring.STANDALONE)
