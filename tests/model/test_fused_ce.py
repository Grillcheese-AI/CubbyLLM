"""Order 0050 stage 2a (``cubbyllm/model/vocab/fused_ce.py``): the chunked cross-entropy equals the full path -- loss and
both gradients -- within 1e-5 (relative) in fp32 and 1e-3 in bf16, with ignored targets and a ragged last chunk; and
it never holds a (positions x vocabulary) tensor beyond one chunk. A CUDA run of the same file is the pod's kernels
check (``CB_FUSED_CE``)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from cubbyllm.model.vocab.fused_ce import chunked_cross_entropy  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _case(dtype, n=300, d=32, v=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g).to(DEV, dtype).requires_grad_()
    w = (torch.randn(v, d, generator=g) * 0.1).to(DEV, dtype).requires_grad_()
    y = torch.randint(0, v, (n,), generator=g).to(DEV)
    y[::7] = -100                                                     # ignored positions
    return h, w, y


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-3)])
def test_the_chunked_loss_and_gradients_equal_the_full_path(dtype, tol):
    h, w, y = _case(dtype)
    full = F.cross_entropy((h @ w.t()).float(), y, ignore_index=-100)
    full.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    loss = chunked_cross_entropy(h, w, y, chunk=64)                   # 300 = 4 x 64 + a ragged 44
    loss.backward()
    assert _rel(loss, full) <= tol
    if dtype == torch.float32:
        assert _rel(h.grad, gh) <= tol and _rel(w.grad, gw) <= tol
    else:                                                              # bf16: the full path's own grads are rounded
        assert _rel(h.grad, gh) <= 2e-2 and _rel(w.grad, gw) <= 2e-2
        h32, w32 = h.detach().float().requires_grad_(), w.detach().float().requires_grad_()
        F.cross_entropy(h32 @ w32.t(), y, ignore_index=-100).backward()
        assert _rel(h.grad, h32.grad) <= 2e-2 and _rel(w.grad, w32.grad) <= 2e-2


def test_a_batch_shape_and_the_upstream_gradient_carry_through():
    h, w, y = _case(torch.float32, n=2 * 50)
    hb, yb = h.detach().reshape(2, 50, -1).requires_grad_(), y.reshape(2, 50)
    (3.0 * chunked_cross_entropy(hb, w, yb, chunk=16)).backward()
    ref = h.detach().clone().requires_grad_()
    (3.0 * F.cross_entropy(ref @ w.detach().t(), y, ignore_index=-100)).backward()
    assert hb.grad.shape == hb.shape and _rel(hb.grad.reshape(-1, hb.shape[-1]), ref.grad) <= 1e-5


def test_nothing_the_size_of_the_logits_is_saved_for_backward():
    h, w, y = _case(torch.float32, n=512, v=4000)
    loss = chunked_cross_entropy(h, w, y, chunk=64)
    saved = loss.grad_fn.saved_tensors
    assert max(t.numel() for t in saved) < 512 * 4000 // 4               # h, w, y, lse: never (positions x vocab)


def test_a_bf16_state_against_fp32_rows_as_autocast_trains_it():
    h, w, y = _case(torch.float32)
    hb = h.detach().to(torch.bfloat16).requires_grad_()
    loss = chunked_cross_entropy(hb, w, y, chunk=64)
    loss.backward()
    assert hb.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.float32
    ref_h = hb.detach().float().requires_grad_()
    ref_w = w.detach().to(torch.bfloat16).float().requires_grad_()
    F.cross_entropy(ref_h @ ref_w.t(), y, ignore_index=-100).backward()
    assert _rel(w.grad, ref_w.grad) <= 2e-2 and _rel(hb.grad, ref_h.grad) <= 2e-2


def test_the_matmuls_run_in_the_asked_dtype_as_autocast_runs_the_full_head():
    """The callers compute h under autocast and call outside it: h is fp32 there, and dtype=bf16 puts the matmuls
    where the full head runs them (the A100's first read ran them in fp32: +460 ms a step)."""
    h, w, y = _case(torch.float32)
    loss = chunked_cross_entropy(h, w, y, chunk=64, dtype=torch.bfloat16)
    loss.backward()
    assert h.grad.dtype == torch.float32 and w.grad.dtype == torch.float32
    hb, wb = h.detach().to(torch.bfloat16).float().requires_grad_(), w.detach().to(torch.bfloat16).float().requires_grad_()
    ref = F.cross_entropy(hb @ wb.t(), y, ignore_index=-100)
    ref.backward()
    assert _rel(loss, ref) <= 1e-2 and _rel(h.grad, hb.grad) <= 2e-2 and _rel(w.grad, wb.grad) <= 2e-2
    with torch.no_grad():                                                  # eval: the loss alone, nothing saved
        assert _rel(chunked_cross_entropy(h, w, y, chunk=64, dtype=torch.bfloat16), loss) <= 1e-6
