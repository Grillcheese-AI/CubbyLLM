import numpy as np

import cubbyllm.model.binding.base as mod
from cubbyllm.core.protocols import Wiring
from cubbyllm.model.binding import BindingHead, DirectUnbinder, Unbinder
from cubbyllm.ops import BlockCodeVSA

K, L = 8, 16


def test_binding_head_uses_ops_seam():
    head = BindingHead()
    assert head.vsa.backend() in {"grilly_bridge", "grilly_python", "numpy"}


def test_role_filler_roundtrip_via_direct_unbinder():
    """Bind a role to a filler, recover the filler from the role (H-B3 unbind)."""
    vsa = BlockCodeVSA(k=K, l=L)
    head = BindingHead(vsa)
    unbinder = DirectUnbinder(vsa)
    cb = vsa.codebook(6)
    role, filler = cb[0], cb[3]
    bound = head.bind(role, filler)
    recovered = unbinder.unbind(bound, role)
    sims = vsa.similarity_batch(recovered, cb)
    assert int(np.argmax(sims)) == 3  # recovers the right filler


def test_bind_pairs_bundles_role_filler_structure():
    # Role/filler binding needs INDEPENDENT random atoms (orthogonal=False).
    # The orthogonal=True codebook builds entries by successive binding
    # (cb[i]=bind(cb[i-1],cb[1])), so it is algebraically structured and unbinding
    # produces systematic crosstalk onto a neighbor — that set is for resonator
    # factorization slots, not role/filler. Real block dims (k=80,l=128), the
    # regime H-B5 validated at 100%.
    vsa = BlockCodeVSA(k=80, l=128)
    head = BindingHead(vsa)
    unbinder = DirectUnbinder(vsa)
    cb = vsa.codebook(10, orthogonal=False)
    roles = [cb[0], cb[1], cb[2]]
    fillers = [cb[5], cb[6], cb[7]]
    struct = head.bind_pairs(roles, fillers)
    # unbind role[1] -> should point at filler[1] (cb index 6)
    rec = unbinder.unbind(struct, roles[1])
    assert int(np.argmax(vsa.similarity_batch(rec, cb))) == 6


def test_unbinder_satisfies_protocol():
    assert isinstance(DirectUnbinder(), Unbinder)


def test_wiring_is_wired():
    assert mod.__wiring__ is Wiring.WIRED


def test_binding_aux_loss_penalizes_collapse():
    """The contrastive binding loss must make representation COLLAPSE the
    high-loss state (2026-08-05 fix — the old own-cosine-only loss gave collapse
    a free ride). Collapse = every hidden state on one direction -> fillers are
    indistinguishable -> loss saturates at chance (log(N)/log(N) = 1.0). A
    healthy, distinct representation is retrievable -> low loss.
    """
    import torch

    from cubbyllm.model.binding.torch_ops import binding_aux_loss, make_roles

    torch.manual_seed(0)
    B, S, d, n = 4, 64, 256, 8
    roles = make_roles(n, d, device="cpu")

    # healthy: distinct random hidden states -> separable fillers -> low loss
    loss_healthy = binding_aux_loss(torch.randn(B, S, d), roles).item()

    # collapsed: every hidden state identical -> indistinguishable -> ~chance
    v = torch.randn(1, 1, d)
    loss_collapsed = binding_aux_loss(v.expand(B, S, d).contiguous(), roles).item()

    assert loss_collapsed > 0.95, f"collapse should score ~chance (1.0), got {loss_collapsed}"
    assert loss_healthy < 0.6, f"healthy should be low, got {loss_healthy}"
    assert loss_collapsed > loss_healthy + 0.3, (
        f"collapse ({loss_collapsed}) must be clearly worse than healthy ({loss_healthy})"
    )


def test_binding_aux_loss_backprops():
    """Loss is differentiable end-to-end into the trunk's hidden states."""
    import torch

    from cubbyllm.model.binding.torch_ops import binding_aux_loss, make_roles

    torch.manual_seed(1)
    h = torch.randn(2, 48, 128, requires_grad=True)
    loss = binding_aux_loss(h, make_roles(8, 128, device="cpu"))
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
