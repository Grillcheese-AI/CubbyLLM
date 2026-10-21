"""The fused MinGRU kernels (cubbyllm/model/backbone/mingru_triton.py) against the recurrence.

Pinned, to float32 precision: the forward is ``h_t = a_t h_{t-1} + sigmoid(g_t) tanh(v_t)``
with ``a_t = 0.001 + 0.998 sigmoid(d_t)``, across chunk and channel-block edges (S and D not
multiples of the tile); the three input gradients are autograd's through the sequential
recurrence; bf16 inputs give the float32 recurrence of the rounded inputs; the torch path the
mixer takes without the flag agrees; and the op passes ``torch.library.opcheck`` (its fake
implementation and its autograd registration). On a CPU the kernels run under
``TRITON_INTERPRET=1``; without triton (Windows) the file skips.
"""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")         # before triton sees the kernels
pytest.importorskip("triton")

from cubbyllm.model.backbone import mingru_triton as mt  # noqa: E402
from cubbyllm.model.backbone.mingru import _log_domain_scan, _sequential_scan  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("no GPU and the kernels were compiled before TRITON_INTERPRET was set", allow_module_level=True)


@pytest.fixture(autouse=True)
def small_tiles(monkeypatch):
    monkeypatch.setattr(mt, "T_CHUNK", 16)
    monkeypatch.setattr(mt, "C_BLOCK", 16)


def _inputs(B=2, S=37, D=40, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, S, D, generator=gen, dtype=torch.float64) * 2).to(DEV, dtype) for _ in range(3)]


def _reference(g, v, d):
    return _sequential_scan(torch.sigmoid(g) * torch.tanh(v), 0.001 + 0.998 * torch.sigmoid(d))


def test_the_forward_is_the_recurrence_across_tile_edges():
    g, v, d = _inputs()
    h = mt.mingru_scan(g, v, d)
    assert h.dtype == torch.float32 and h.shape == g.shape
    assert torch.allclose(h.double(), _reference(g.double(), v.double(), d.double()), atol=2e-6)


def test_the_gradients_are_autograds_through_the_recurrence():
    g, v, d = _inputs(seed=1)
    w = torch.randn(g.shape, generator=torch.Generator().manual_seed(9), dtype=torch.float64).to(DEV)
    ins = [t.clone().requires_grad_() for t in (g, v, d)]
    ref = [t.double().clone().requires_grad_() for t in (g, v, d)]
    got = torch.autograd.grad((mt.mingru_scan(*ins).double() * w).sum(), ins)
    want = torch.autograd.grad((_reference(*ref) * w).sum(), ref)
    for a, b in zip(got, want):
        assert a.dtype == torch.float32
        assert torch.allclose(a.double(), b, atol=5e-6, rtol=1e-5)


def test_bf16_inputs_give_the_float32_recurrence_of_the_rounded_inputs():
    g, v, d = _inputs(seed=2, dtype=torch.bfloat16)
    h = mt.mingru_scan(g, v, d)
    assert h.dtype == torch.float32
    assert torch.allclose(h.double(), _reference(g.double(), v.double(), d.double()), atol=2e-6)
    ins = [t.clone().requires_grad_() for t in (g, v, d)]
    grads = torch.autograd.grad(mt.mingru_scan(*ins).sum(), ins)
    assert all(x.dtype == torch.bfloat16 and torch.isfinite(x.float()).all() for x in grads)


def test_the_torch_path_agrees():
    """The mixer's own path (log-domain parallel scan) and the kernel: one function."""
    g, v, d = _inputs(seed=3)
    torch_path = _log_domain_scan(torch.sigmoid(g) * torch.tanh(v), 0.001 + 0.998 * torch.sigmoid(d))
    assert torch.allclose(mt.mingru_scan(g, v, d), torch_path, atol=1e-4, rtol=1e-4)


def test_opcheck():
    g, v, d = _inputs(B=1, S=20, D=18, seed=4)
    torch.library.opcheck(mt.mingru_scan, tuple(t.requires_grad_() for t in (g, v, d)),
                          test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))


@pytest.mark.skipif(DEV != "cuda", reason="the mixer takes the kernel on CUDA only")
def test_the_mixer_takes_the_kernel_under_the_flag(monkeypatch):
    from cubbyllm.model.backbone import mingru
    torch.manual_seed(0)
    m = mingru._MinGRUMixer(64).to(DEV)
    x = torch.randn(3, 50, 64, device=DEV, requires_grad=True)
    ref = m(x)
    gref = torch.autograd.grad(ref.square().sum(), [x] + list(m.parameters()))
    monkeypatch.setattr(mingru, "TRITON", True)
    out = m(x)
    got = torch.autograd.grad(out.square().sum(), [x] + list(m.parameters()))
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    for a, b in zip(got, gref):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
