"""The spatial link on grilly2 (order 0052 A): a linked base exported (the trunk's files as before, the link beside
them) and run by grilly2's ``CubbyForCausalLM`` with ``GrillyLink`` gives torch's logits -- over a prompt (the forward)
and one token at a time after it (the step path) -- within 1e-4 relative (the order's parity line), roles, a document
restart and the copy open; with nothing placed grilly2's logits are its own, bit for bit. Skipped where grilly2 is not
on the path (``uv run --project <grilly2>``)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("grilly.infer.models")
import grilly  # noqa: E402
import inspect  # noqa: E402
from grilly.infer.models import CubbyForCausalLM  # noqa: E402

if "spatial" not in inspect.signature(CubbyForCausalLM.forward).parameters:
    pytest.skip("this grilly2 has no spatial hook (its branch cubby-spatial-link, under review)", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "validation"))

from cubbyllm.ops.grilly_spatial import GrillyLink  # noqa: E402
from export_base import grilly_config, named_tensors, spatial_export  # noqa: E402
from test_spatial import S  # noqa: E402
from test_spatial_copy import _decode, _linked, _open, _roles  # noqa: E402


def _export(m, out):
    from safetensors.torch import save_file
    from test_grow import META
    meta = dict(META, vocab_real=META["vocab"])
    cfg = grilly_config(meta, m)
    save_file(named_tensors(m), os.path.join(out, "model.safetensors"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
    sp = {"query": "placed", "bind": "writes", "copy": "ring" if m.spatial.copy_path is not None else None}
    lc, lt = spatial_export({"spatial": sp}, m)
    json.dump(lc, open(os.path.join(out, "spatial.json"), "w"))
    save_file(lt, os.path.join(out, "spatial.safetensors"))
    from cubbyllm.ops.grilly_seats import load_export
    return load_export(out)[0]


def _rel(a, b):
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


@pytest.mark.parametrize("copy", [None, "ring"])
def test_grilly2_with_the_link_gives_torchs_logits_over_a_prompt_and_its_steps(tmp_path, copy):
    m = _linked(copy, ring=3)
    if copy:
        _open(m, bias=2.0, tau=1.0)
    x, pos, placed, writes, asks, starts = _roles()
    with torch.no_grad():
        ref, _ = _decode(m, x, pos, placed, writes, asks, starts)          # torch's own step path, every token
    g = _export(m, str(tmp_path))
    k = 6                                                                  # a prompt of 6, then 5 steps
    with grilly.no_grad():
        link = GrillyLink.load(str(tmp_path), g, batch=2)
        state = g.make_cache(2, 0)
        cut = lambda v, a, b: v[:, a:b].numpy()                            # noqa: E731
        out = g(grilly.from_numpy(x[:, :k].numpy()), past_key_values=state, spatial=link.roles(
            cut(x, 0, k), cut(pos, 0, k), cut(writes, 0, k), cut(asks, 0, k), cut(starts, 0, k)))
        got, state = [torch.from_numpy(out.logits.numpy())], out.past_key_values
        for t in range(k, S):
            out = g(grilly.from_numpy(x[:, t:t + 1].numpy()), past_key_values=state, spatial=link.roles(
                cut(x, t, t + 1), cut(pos, t, t + 1), cut(writes, t, t + 1), cut(asks, t, t + 1), cut(starts, t, t + 1)))
            got.append(torch.from_numpy(out.logits.numpy()))
            state = out.past_key_values
    assert _rel(torch.cat(got, 1), ref) <= 1e-4


def test_with_nothing_placed_grilly2s_logits_are_its_own(tmp_path):
    m = _linked("ring")
    _open(m, bias=2.0)
    x, pos, placed, writes, asks, starts = _roles()
    none = np.zeros(writes.shape, dtype=bool)
    g = _export(m, str(tmp_path))
    with grilly.no_grad():
        alone = g(grilly.from_numpy(x.numpy())).logits.numpy()
        link = GrillyLink.load(str(tmp_path), g, batch=2)
        linked = g(grilly.from_numpy(x.numpy()), spatial=link.roles(x.numpy(), pos.numpy(), none, none,
                                                                    starts.numpy())).logits.numpy()
    assert np.array_equal(alone, linked)
