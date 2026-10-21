"""The banded flash-attention kernels (cubbyllm/model/backbone/window_attn_triton.py) against the
dense reference (``banded.dense_attention``: SDPA under the S x S causal-window mask).

Pinned: values and the three gradients, with the sequence a multiple of no tile and windows
narrower than a tile, wider than one, and wider than the sequence; bf16 inputs within bf16
error; ``opcheck`` (schema, fake, autograd); and on CUDA the hybrid's attention mixer under
the flag. On a CPU the kernels run under ``TRITON_INTERPRET=1``; without triton the file skips.
"""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")         # before triton sees the kernels
pytest.importorskip("triton")

from cubbyllm.model.backbone import window_attn_triton as wa  # noqa: E402
from cubbyllm.model.backbone.banded import dense_attention  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("no GPU and the kernels were compiled before TRITON_INTERPRET was set", allow_module_level=True)
DT = torch.float32 if DEV == "cpu" else torch.bfloat16     # on a GPU fp32 dots would be tf32
TOL = 2e-5 if DEV == "cpu" else 3e-2


def _qkv(B=2, h=2, S=45, D=32, seed=0, dtype=DT):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, h, S, D, generator=g).to(DEV, dtype) for _ in range(3)]


@pytest.mark.parametrize("window", [5, 20, 64])
def test_values_and_gradients_are_the_dense_references(window):
    q, k, v = _qkv(seed=window)
    w = torch.randn(q.shape, generator=torch.Generator().manual_seed(7)).to(DEV)
    a = [t.clone().requires_grad_() for t in (q, k, v)]
    r = [t.float().clone().requires_grad_() for t in (q, k, v)]
    out = wa.window_attention(*a, window)
    ref = dense_attention(*r, window)
    assert out.dtype == q.dtype and out.shape == q.shape
    assert torch.allclose(out.float(), ref, atol=TOL, rtol=TOL)
    got = torch.autograd.grad((out.float() * w).sum(), a)
    want = torch.autograd.grad((ref * w).sum(), r)
    for x, y in zip(got, want):
        assert x.dtype == q.dtype
        assert torch.allclose(x.float(), y, atol=TOL * 4, rtol=TOL * 4), (x.float() - y).abs().max()


def test_opcheck():
    q, k, v = _qkv(B=1, h=1, S=20, D=16, seed=3)
    torch.library.opcheck(wa.window_attention_fwd, (q.requires_grad_(), k.requires_grad_(), v.requires_grad_(), 8),
                          test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))


def test_transposed_inputs_as_the_mixer_passes_them():
    """The mixer hands (B, S, h, D) views transposed to (B, h, S, D): the op's outputs are contiguous,
    and its fake (what torch.compile plans with) must say so -- it said otherwise once, and only the
    compiled G4 run caught it (``assert_size_stride`` on the run's first step, 2026-10-21)."""
    g = torch.Generator().manual_seed(5)
    q, k, v = [torch.randn(2, 30, 2, 16, generator=g).to(DEV, DT).transpose(1, 2) for _ in range(3)]
    assert not q.is_contiguous()
    torch.library.opcheck(wa.window_attention_fwd, (q, k, v, 8), test_utils=("test_faketensor",))
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        fq, fk, fv = (mode.from_tensor(t) for t in (q, k, v))
        o, _ = wa.window_attention_fwd(fq, fk, fv, 8)
    real, _ = wa.window_attention_fwd(q, k, v, 8)
    assert o.stride() == real.stride() and real.is_contiguous()
    assert torch.allclose(real.float(), dense_attention(q.float(), k.float(), v.float(), 8), atol=TOL, rtol=TOL)


@pytest.mark.skipif(DEV != "cuda", reason="the mixer takes the kernel on CUDA only")
def test_the_mixer_takes_the_kernel_under_the_flag(monkeypatch):
    from cubbyllm.model.backbone import hybrid
    torch.manual_seed(0)
    m = hybrid._WindowedAttnMixer(256, heads=2, window=48).to(DEV)
    x = torch.randn(2, 150, 256, device=DEV, requires_grad=True)
    monkeypatch.setenv("CB_NO_FLEX", "1")
    ref = m(x)
    gref = torch.autograd.grad(ref.square().sum(), [x] + list(m.parameters()))
    monkeypatch.setattr(hybrid, "ATTN_TRITON", True)
    out = m(x)
    got = torch.autograd.grad(out.square().sum(), [x] + list(m.parameters()))
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    for a, b in zip(got, gref):
        assert torch.allclose(a, b, atol=2e-2, rtol=2e-2)
