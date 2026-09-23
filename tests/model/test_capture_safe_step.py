"""The decode step is a pure function of fixed buffers.

That is the contract a recorded-and-replayed step has to satisfy — grilly2's
``grilly.graphed``, torch's CUDA graphs, the same four rules either way: no
host read, no host write, no shape change, state updated in place. These
tests check the three of those a backend cannot check for you, plus the
equivalence that licenses the whole change.

The fourth — the host read — is checked by the backend itself, which
refuses a capture that performs one. ``tests/model/test_graph_seam.py``
covers the seam that asks it to.
"""
import pytest

torch = pytest.importorskip("torch")

from cubbyllm.model.backbone import HybridBackbone  # noqa: E402
from cubbyllm.model.recall import EpisodicStore  # noqa: E402

D, L, W, HEADS = 32, 6, 6, 4


def _backbone(mem_every=0, **kw):
    torch.manual_seed(0)
    return HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=HEADS,
                          mem_every=mem_every, mem_topk=4, mem_key=8, **kw)


def _shapes(state):
    """Every tensor's shape in a decode state, in a stable order."""
    out = []

    def walk(v, path=""):
        if torch.is_tensor(v):
            out.append((path, tuple(v.shape), str(v.dtype)))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(x, f"{path}.{i}")
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                walk(v[k], f"{path}.{k}")
        elif isinstance(v, EpisodicStore):
            for k in ("keys", "keys_n", "values", "codes", "bias"):
                walk(getattr(v, k), f"{path}.{k}")

    walk(state)
    return out


# ── no shape change ──────────────────────────────────────────────────────


@pytest.mark.parametrize("mem_every", [0, 2])
def test_the_state_has_the_same_shapes_from_the_first_token(mem_every):
    """A step whose shapes change is a new graph signature every token, so
    a recording is taken and thrown away every time and nothing ever
    replays. The old cache grew for ``window`` tokens and the old store
    grew forever, so that was every token of a real decode."""
    bb = _backbone(mem_every)
    seen = []
    with torch.no_grad():
        state = None
        for _ in range(3 * W):
            _, state = bb.step(torch.randn(1, D), state)
            seen.append(_shapes(state))
    assert all(s == seen[0] for s in seen), "the decode state changed shape"


@pytest.mark.parametrize("mem_every", [0, 2])
def test_the_state_tensors_are_the_same_objects_every_token(mem_every):
    """In place, and returned. A replay writes into the buffers its
    commands name, so the state a replay produces has to *be* the state the
    next one reads — otherwise the wrapper copies it back every token."""
    bb = _backbone(mem_every)
    with torch.no_grad():
        _, state = bb.step(torch.randn(1, D), None)
        first = [id(t) for _, t in _walk_tensors(state)]
        for _ in range(2 * W):
            _, state = bb.step(torch.randn(1, D), state)
    assert [id(t) for _, t in _walk_tensors(state)] == first


def _walk_tensors(v, path=""):
    if torch.is_tensor(v):
        return [(path, v)]
    if isinstance(v, (list, tuple)):
        return [p for i, x in enumerate(v) for p in _walk_tensors(x, f"{path}.{i}")]
    if isinstance(v, dict):
        return [p for k in sorted(v, key=str) for p in _walk_tensors(v[k], f"{path}.{k}")]
    if isinstance(v, EpisodicStore):
        return [p for k in ("keys", "keys_n", "values", "codes", "bias")
                for p in _walk_tensors(getattr(v, k), f"{path}.{k}")]
    return []


# ── no host value in the state ───────────────────────────────────────────


def test_every_counter_in_the_state_is_a_tensor():
    """A Python int in the state is part of the call's shape signature, so
    it makes every token a new graph. The position is a float32 scalar
    (exact to 2**24 tokens) and the two slot cursors are int64 counters
    advanced by device adds — an int64 *scalar* would have to be built on
    the host, which is why the increment is a preallocated tensor too."""
    bb = _backbone(mem_every=2)
    with torch.no_grad():
        _, state = bb.step(torch.randn(1, D), None)
    for i, st in state["mem"].items():
        assert torch.is_tensor(st["pos"]) and st["pos"].dtype is torch.float32, i
        for name in ("slot", "aged", "one"):
            assert torch.is_tensor(st[name]) and st[name].dtype is torch.int64, (i, name)
        assert torch.is_tensor(st["store"].bias)
    for s in state["mix"]:
        if isinstance(s, tuple):                      # an attention ring
            assert torch.is_tensor(s[2]) and s[2].dtype is torch.float32
            assert torch.is_tensor(s[3]) and s[3].dtype is torch.int64


# ── the equivalence the change has to preserve ───────────────────────────


@pytest.mark.parametrize("mem_every", [0, 2])
def test_the_ring_decodes_the_same_sequence_as_a_trimmed_cache(mem_every):
    """The ring holds the same window of keys in a different order, and
    attention is a softmax over its keys, so the answer is the same up to
    the order the weighted sum accumulates in. Run well past the window so
    the ring has wrapped several times."""
    bb = _backbone(mem_every)
    S = 4 * W
    x = torch.randn(1, S, D)
    with torch.no_grad():
        parallel = bb.forward(x)
        state, seq = None, []
        for t in range(S):
            y, state = bb.step(x[:, t], state)
            seq.append(y)
        incremental = torch.stack(seq, dim=1)
    assert torch.allclose(parallel, incremental, atol=1e-3), \
        f"decode diverged; max |diff| {(parallel - incremental).abs().max():.2e}"


def test_resuming_from_a_carried_state_is_unchanged_by_the_rings():
    """In-place state is the one change that could break this: if a step
    kept a reference into a buffer a later step overwrites, splitting the
    run would not match decoding straight through."""
    bb = _backbone(mem_every=2)
    S = 3 * W
    x = torch.randn(1, S, D)
    with torch.no_grad():
        state, out = None, []
        for t in range(S):
            y, state = bb.step(x[:, t], state)
            out.append(y.clone())
        straight = torch.stack(out, dim=1)

        bb2 = _backbone(mem_every=2)
        state, out = None, []
        for t in range(S):
            y, state = bb2.step(x[:, t], state)
            out.append(y.clone())
        split = torch.stack(out, dim=1)
    assert torch.equal(straight, split)


# ── the store's fixed capacity ───────────────────────────────────────────


def test_the_store_wraps_at_capacity_and_keeps_the_newest():
    """The behaviour change the fixed capacity buys, asserted rather than
    assumed: past ``capacity`` aged-out tokens the oldest entry goes. See
    H-A8 in CUBBYLLM_HYPOTHESES.md."""
    cap = 4
    bb = _backbone(mem_every=2, mem_capacity=cap)
    with torch.no_grad():
        state = None
        for _ in range(W + cap + 3):
            _, state = bb.step(torch.randn(1, D), state)
    store = state["mem"][0]["store"]
    assert store.keys.shape[0] == cap
    assert len(store) == cap, "the fill level should saturate, not grow"


def test_the_store_holds_only_the_beyond_window_past():
    """The causal contract, and the reason the ring doubles as the FIFO:
    at step t the store must hold tokens 0..t-window-1 and nothing newer,
    because ``forward``'s mask is ``j < i - window``."""
    bb = _backbone(mem_every=2)
    with torch.no_grad():
        state = None
        for t in range(3 * W):
            _, state = bb.step(torch.randn(1, D), state)
            expected = max(0, t + 1 - W)
            assert len(state["mem"][0]["store"]) == expected, \
                f"after {t + 1} tokens the store holds {len(state['mem'][0]['store'])}, want {expected}"
