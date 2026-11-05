"""The spatial link's decode port (``cubbyllm/model/spatial_decode.py``, order 0052 A), against torch.

Pinned: the DFT matrices are numpy's rfft / irfft; the torch trunk stepped with the numpy link where grilly2 calls it
(the input bound, the scene and the ring written, the read at layer ``at``, the copy on the logits) gives torch's own
step's logits at every token -- roles, a document restart, the copy's gate open -- within 1e-4 relative (the order's
parity line); with nothing placed the logits are bit-identical; the export writes the link beside the trunk and never
into its files, and refuses the variants the port has not taken."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from cubbyllm.model.spatial_decode import SpatialDecode, dft
from test_spatial import S
from test_spatial_copy import _decode, _linked, _open, _roles

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "validation"))


def test_the_dft_matrices_are_rfft_and_irfft():
    d, x = 16, np.random.default_rng(0).normal(size=(3, 16))
    Fr, Fi, Ir, Ii = dft(d)
    f = np.fft.rfft(x)
    assert np.allclose(x @ Fr, f.real) and np.allclose(x @ Fi, f.imag)
    z = np.random.default_rng(1).normal(size=(3, 9)) + 1j * np.random.default_rng(2).normal(size=(3, 9))
    assert np.allclose(z.real @ Ir + z.imag @ Ii, np.fft.irfft(z, n=d))


def _port(m):
    from export_base import spatial_export
    sp = {"scale": m.spatial.scale, "seed": m.spatial.seed, "at": m.spatial.at, "query": "placed", "bind": "writes",
          "copy": "ring" if m.spatial.copy_path is not None else None}
    cfg, t = spatial_export({"spatial": sp}, m)
    return SpatialDecode(cfg, {k: v.numpy() for k, v in t.items()})


def _hybrid(m, dec, x, pos, placed, writes, asks, starts):
    """CubbyModel.step with the link's four calls made by the numpy port (what grilly2's trunk does)."""
    st, state, out = dec.state(x.shape[0]), m.init_state(x.shape[0]), []
    table = m.embedding.core.weight.detach().numpy()
    with torch.no_grad():
        for t in range(S):
            tok = x[:, t]
            e = m.embedding.embed(tok.unsqueeze(1), ctx=None)[:, 0]
            state["ctx_sum"].add_(e)
            state["ctx_n"].add_(1.0)
            ctx = m.context_source.infer(state["ctx_sum"] / state["ctx_n"])
            xi = m.embedding.embed(tok.unsqueeze(1), ctx)[:, 0]
            xb = dec.bind_input(xi.numpy(), pos[:, t].numpy(), writes[:, t].numpy())
            dec.write(st, xb, tok.numpy(), writes[:, t].numpy(), starts[:, t].numpy())
            p, a = pos[:, t].numpy(), asks[:, t].numpy()
            read = (dec.at, lambda h: torch.from_numpy(dec.read(st, h.numpy(), p, a)).to(h.dtype))
            y, state["bb"] = m.backbone.step(torch.from_numpy(np.asarray(xb)).float(), state["bb"], read=read)
            y = m.memory.forward_generated(y.unsqueeze(1), ctx)[:, 0]
            lg = m.head.logits(y.unsqueeze(1), m.retrieval_k)[:, 0]
            out.append(torch.from_numpy(dec.add_copy(st, lg.numpy(), y.numpy(), p, a, table)))
    return torch.stack(out, 1)


@pytest.mark.parametrize("copy", [None, "ring"])
def test_the_port_steps_as_torch_steps(copy):
    m = _linked(copy, ring=3)
    if copy:
        _open(m, bias=2.0, tau=1.0)
    roles = _roles()
    with torch.no_grad():
        ref, _ = _decode(m, *roles)
    got = _hybrid(m, _port(m), *roles)
    rel = ((got - ref).norm(dim=-1) / ref.norm(dim=-1)).max()
    assert rel <= 1e-4, float(rel)
    if copy:                                          # the copy acted: asking rows are log-probabilities
        asks = roles[4]
        assert torch.allclose(got[asks].exp().sum(-1), torch.ones(int(asks.sum())), atol=1e-4)


def test_with_nothing_placed_the_logits_are_bit_identical():
    m = _linked("ring")
    _open(m, bias=2.0)
    x, pos, placed, writes, asks, starts = _roles()
    none = torch.zeros_like(writes)
    got = _hybrid(m, _port(m), x, pos, none, none, none, starts)
    m.spatial = None                                   # the trunk alone
    with torch.no_grad():
        state, ref = None, []
        for t in range(S):
            lg, state = m.step(x[:, t], state)
            ref.append(lg)
    assert torch.equal(got, torch.stack(ref, 1))


def test_the_export_writes_the_link_beside_the_trunk_and_refuses_what_is_not_ported(tmp_path):
    from export_base import spatial_export
    m = _linked("ring")
    cfg, t = spatial_export({"spatial": {"copy": "ring", "query": "placed", "bind": "writes"}}, m)
    assert {"phases", "q.weight", "o.weight", "copy.tau"} <= set(t) and cfg["d"] == m.spatial.d
    assert not any(k.startswith("model.") for k in t)                 # never a trunk name
    for bad in ({"key": "text"}, {"copy_refuse": {"theta": 1}}, {"query": "pointer"}, {"copy": "table"}):
        with pytest.raises(SystemExit):
            spatial_export({"spatial": {"copy": "ring", **bad}}, m)
    from safetensors.numpy import save_file
    import json
    json.dump(cfg, open(tmp_path / "spatial.json", "w"))
    save_file({k: v.numpy() for k, v in t.items()}, str(tmp_path / "spatial.safetensors"))
    assert SpatialDecode.load(str(tmp_path)).at == m.spatial.at
    assert SpatialDecode.load(str(tmp_path / "nothing")) is None      # an export without the link


def _hybrid_forward(m, dec, x, pos, placed, writes, asks, starts):
    """CubbyModel.forward with the link's calls made by one port ``Call`` over the whole sequence (a prefill)."""
    st, table = dec.state(x.shape[0]), m.embedding.core.weight.detach().numpy()
    with torch.no_grad():
        ctx = m.infer_context(x)
        xi = m.embedding.embed(x, ctx)
        call = dec.call(st, xi.numpy(), x.numpy(), pos.numpy(), writes.numpy(), asks.numpy(), starts.numpy())

        def fn(h):
            r = call.read(h.numpy())
            return torch.zeros_like(h) if r is None else torch.from_numpy(r).to(h.dtype)
        h = m.backbone.forward(torch.from_numpy(call.xb).float(), read=(dec.at, fn))
        h = m.memory.forward_generated(h, ctx)
        lg = m.logits_from(h)
        return torch.from_numpy(call.copy(lg.numpy(), h.numpy(), table)), st


@pytest.mark.parametrize("copy", [None, "ring"])
def test_one_call_over_a_prompt_is_the_torch_forward_and_leaves_the_steps_state(copy):
    m = _linked(copy, ring=3)
    if copy:
        _open(m, bias=2.0, tau=1.0)
    x, pos, placed, writes, asks, starts = _roles()
    with torch.no_grad():
        ref = m.forward(x, pos, placed, writes, asks, starts)
    dec = _port(m)
    got, st = _hybrid_forward(m, dec, x, pos, placed, writes, asks, starts)
    assert float(((got - ref).norm(dim=-1) / ref.norm(dim=-1)).max()) <= 1e-4
    st2 = dec.state(x.shape[0])                                  # the prompt's end state = the steps' end state
    with torch.no_grad():
        xi = m.embedding.embed(x, m.infer_context(x)).numpy()
    for t in range(S):
        dec.write(st2, dec.bind_input(xi[:, t], pos[:, t].numpy(), writes[:, t].numpy()),
                  x[:, t].numpy(), writes[:, t].numpy(), starts[:, t].numpy())
    assert np.allclose(st["scene"], st2["scene"]) and all(np.array_equal(st[k], st2[k]) for k in st if k != "scene")
