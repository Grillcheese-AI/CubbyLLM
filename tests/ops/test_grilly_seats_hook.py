"""Order 0071, part B: grilly_seats.generate hands grilly2's logits hook to the model only when given; without it, the
call the seats make is the one they made before (no new argument reaches the model). With grilly2 importable and the
hook present, an allowed list holds on a tiny Cubby model through the seam."""
from __future__ import annotations

import contextlib

import pytest

from cubbyllm.ops import grilly_seats as gs


class _Out:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class _G:
    no_grad = staticmethod(contextlib.nullcontext)
    tensor = staticmethod(lambda rows: rows)

    def manual_seed(self, s):
        self.seed = s


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, ids, **kw):
        self.calls.append(kw)
        return _Out([r + [0] for r in ids])


def test_without_the_hook_the_call_is_the_one_before():
    m = _Model()
    gs.generate(m, [[1, 2]], 4, eos=9, g=_G())
    gs.generate(m, [[1, 2]], 4, eos=9, temperature=0.7, top_k=5, seed=3, g=_G())
    assert m.calls[0] == {"max_new_tokens": 4, "eos_token_id": 9}
    assert m.calls[1] == {"max_new_tokens": 4, "eos_token_id": 9, "do_sample": True, "temperature": 0.7, "top_k": 5,
                          "top_p": 1.0}


def test_the_hook_is_handed_through_when_given():
    m, proc, fn = _Model(), [lambda i, s: s], (lambda b, s: [1])
    gs.generate(m, [[1]], 2, eos=9, g=_G(), logits_processor=proc)
    gs.generate(m, [[1]], 2, eos=9, temperature=0.5, g=_G(), prefix_allowed_tokens_fn=fn)
    assert m.calls[0]["logits_processor"] is proc and "prefix_allowed_tokens_fn" not in m.calls[0]
    assert m.calls[1]["prefix_allowed_tokens_fn"] is fn and "logits_processor" not in m.calls[1]


def test_an_allowed_list_holds_through_the_seam_on_grilly2():
    lp = pytest.importorskip("grilly.infer.logits_process", reason="grilly2 with the logits hook (0071) not installed")
    from grilly.infer.models import CubbyConfig, CubbyForCausalLM
    try:
        model = CubbyForCausalLM(CubbyConfig(vocab_size=96, hidden_size=32, num_hidden_layers=3, num_attention_heads=2, window=16))
    except TypeError as e:                                # the config's field names moved
        pytest.skip(f"CubbyConfig: {e}")
    allowed = {4, 9, 33, 70}
    out = gs.generate(model, [[1, 2, 3]], 12, eos=95, logits_processor=[lp.AllowedTokensLogitsProcessor(allowed)])
    assert set(out[0][3:]) <= allowed
