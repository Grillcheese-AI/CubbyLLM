"""The export contract, end to end: a base exported by ``validation/export_base.py``
loads into grilly2's ``CubbyForCausalLM`` and gives the torch model's logits.

Over a prompt (the training forward) and one token at a time (the step path,
``CubbyModel.step``), with and without the episodic memory. With it, the
window is 4 and the store 6 rows, so twenty tokens take the step path through
the first read (position 5), a full store (position 10) and its wrap: the two
decode paths have to age the same token into the same row at the same step.
Skipped where grilly2 is not on the path (``PYTHONPATH=<grilly2>/python``).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

grilly = pytest.importorskip("grilly")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "validation"))

from export_base import grilly_config, named_tensors  # noqa: E402
from test_grow import META, _build  # noqa: E402

S, CAPACITY = 20, 6


def _torch_model(memory: bool):
    meta = dict(META, vocab_real=META["vocab"], mem_every=2 if memory else 0, mem_topk=3, mem_key=4)
    m = _build(meta)
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.context_source.parameters()):
            p.copy_(torch.randn_like(p) * 0.3)
    m.backbone.mem_capacity = CAPACITY                 # read by init_state (the store) and by the export
    return m, meta


def _export(model, meta, out):
    from safetensors.torch import save_file
    cfg = grilly_config(meta, model)
    cfg["model_type"] = "cubby"
    save_file(named_tensors(model), os.path.join(out, "model.safetensors"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return cfg


def _load(out):
    from grilly.infer.models import CubbyConfig, CubbyForCausalLM
    from grilly.infer.safetensors import load_model
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = CubbyForCausalLM(CubbyConfig(**{k: v for k, v in cfg.items() if k != "model_type"}))
    missing, unexpected = load_model(model, os.path.join(out, "model.safetensors"))
    assert not missing and not unexpected, (missing, unexpected)
    return model


def _close(a, b, what):
    tol = 1e-4 * max(1.0, float(np.abs(a).max()))
    assert np.abs(a - b).max() <= tol, f"{what}: max |dlogit| {np.abs(a - b).max():.2e} > {tol:.2e}"


@pytest.mark.parametrize("memory", [False, True], ids=["plain", "memory"])
def test_the_export_gives_the_torch_models_logits(tmp_path, memory):
    tm, meta = _torch_model(memory)
    cfg = _export(tm, meta, str(tmp_path))
    assert cfg["mem_every"] == (2 if memory else 0)
    if memory:
        assert (cfg["mem_topk"], cfg["mem_key"], cfg["mem_capacity"]) == (3, 4, CAPACITY)
    gm = _load(str(tmp_path))
    x = torch.randint(0, META["vocab"], (2, S), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        want = tm.forward(x).numpy()
    with grilly.no_grad():
        got = gm(grilly.from_numpy(x.numpy().astype(np.int64))).logits.numpy()
    _close(want, got, "prompt")

    # one token at a time: torch's step (one stream, as its memory store is) against grilly's
    with torch.no_grad():
        state, want = None, []
        for s in range(S):
            logits, state = tm.step(x[:1, s], state)
            want.append(logits.numpy())
    with grilly.no_grad():
        cache, got = gm.make_cache(1, 0), []
        for s in range(S):
            tok = grilly.from_numpy(x[:1, s:s + 1].numpy().astype(np.int64))
            got.append(gm(tok, past_key_values=cache).logits.numpy()[:, -1])
    _close(np.stack(want, 1), np.stack(got, 1), "decode")
    if memory:
        assert cache.capacity == CAPACITY and cache.position == S
