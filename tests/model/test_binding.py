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
