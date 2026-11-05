"""Order 0049's GPU harness (``validation/selfimprove/``), pinned on CPU with a toy recurrent model:
- the batched sampler gives each prompt what it gets alone (no padding enters a row's state);
- unlikelihood reaches only the wrong program's tokens from its first divergence;
- the marker's verdicts;
- the keys split wd from plan, and an arithmetic level from the others;
- a probe with too few right programs gives no gain (a shuffled family is never flagged)."""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from selfimprove.trunk import first_divergence, sample_many  # noqa: E402

V, D, EOS = 11, 8, 0


class _LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, 2, bias=False)
        self.lora_B = nn.Linear(2, base.out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


class _Toy(nn.Module):
    """A tiny recurrent LM with the trunk's decode and training surface."""
    lora_targets = ("out",)

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = nn.Parameter(torch.randn(V, D, generator=g))
        self.rec = nn.Parameter(torch.randn(D, D, generator=g) * 0.5)
        self.out = _LoRA(nn.Linear(D, V))

    def lora_find(self, t):
        return self.out

    def init_state(self, batch=1, device=None):
        return torch.zeros(batch, D, device=device)

    def step(self, tok, state):
        h = torch.tanh(state @ self.rec + self.emb[tok])
        return self.out(h), h

    def features(self, x):
        h, hs = torch.zeros(x.shape[0], D), []
        for t in range(x.shape[1]):
            h = torch.tanh(h @ self.rec + self.emb[x[:, t]])
            hs.append(h)
        return torch.stack(hs, 1)

    def logits_from(self, h):
        return self.out(h)


def test_a_batch_of_prompts_samples_what_each_prompt_samples_alone():
    m = _Toy()
    prompts, caps = [[3, 4, 5, 6, 7], [2], [9, 8, 7]], [6, 4, 5]
    together = sample_many(m, prompts, 1, 0.0, caps, EOS, "cpu")
    alone = [sample_many(m, [p], 1, 0.0, [c], EOS, "cpu")[0] for p, c in zip(prompts, caps)]
    assert together == alone and all(len(x[0]) <= c for x, c in zip(together, caps))
    k3 = sample_many(m, prompts, 3, 0.0, caps, EOS, "cpu")
    assert all(rows == [alone[i][0]] * 3 for i, rows in enumerate(k3))             # greedy: k copies of the same


def test_the_first_divergence():
    assert first_divergence([1, 2, 3, 0], [1, 2, 4, 0]) == 2
    assert first_divergence([1, 2, 0], [1, 2, 3, 0]) == 2                          # stopped early: its </s>
    assert first_divergence([5, 0], [1, 0]) == 0


def _trunk():
    from selfimprove.trunk import Trunk
    tk = types.SimpleNamespace(token_to_id=lambda s: EOS,
                               encode=lambda s: types.SimpleNamespace(ids=[1 + (ord(c) % (V - 1)) for c in s]))
    return Trunk("", "", "", dev="cpu", model=_Toy(), tk=tk)


def test_unlikelihood_reaches_only_the_tokens_from_the_divergence(monkeypatch):
    t = _trunk()
    row = t.negative("ab", "cdef", "cdxy")                    # shared "cd", then the wrong program's own tokens
    assert row["d"] == 2
    seen = []
    real = t.model.logits_from
    monkeypatch.setattr(t.model, "logits_from", lambda h: seen.append(h.shape[1]) or real(h))
    t._ul([row])
    assert seen == [len(row["t"]) - row["d"]]                  # "ef", "\n" and </s>: never "c", "d"
    assert t.negative("ab", "cd", "cd") is None                # no divergence: nothing to push down


def test_training_moves_the_adapter_and_never_the_trunk():
    t = _trunk()
    frozen = [p.detach().clone() for n, p in t.model.named_parameters() if "lora" not in n]
    for n, p in t.model.named_parameters():
        p.requires_grad_("lora" in n)
    t.train([("ab", "cd")], [("ab", "ce", "cd")], steps=3, batch=4)
    assert any(not torch.equal(a, b) for a, b in zip(t.base, t.state()))
    assert all(torch.equal(a, p) for a, (n, p) in zip(frozen, [(n, p) for n, p in t.model.named_parameters()
                                                               if "lora" not in n]))


class _Judge:
    def __init__(self, outcome="correct", why="", runs=(True, "42")):
        self.o, self.why, self.runs = outcome, why, runs

    def attempt_step(self, row, text):
        return {"outcome": self.o, "why": self.why}

    def _run(self, program):
        return self.runs


class _Aliases:
    _pid_of = {"place of birth": "P19", "date of birth": "P569", "residence": "P551"}

    def relations(self, w):
        return {"born in": ["place of birth"], "born": ["place of birth", "date of birth"],
                "home": ["date of birth", "residence"], "born on": ["date of birth"]}.get(w, [])


def _wd(word):
    return {"id": "wd-1", "task": "wd", "gold": "P19", "question": "where was X born", "spans": [],
            "reference": ""}, f'use vsa;\n bind frame, HOP1, "{word}";\n'


def test_the_marker_says_right_wrong_refused_or_caught():
    from selfimprove.mark import Marker
    mk = Marker([], judge=_Judge(), aliases=_Aliases())
    assert mk(*_wd("born in")) == {"verdict": "right", "gold_right": True, "why": ""}
    assert mk(*_wd("born"))["gold_right"] is True                                  # the gold among two (the item decides)
    assert mk(*_wd("home"))["verdict"] == "refused"                                  # two, neither the gold
    assert mk(*_wd("born on"))["gold_right"] is False                                # one, not the gold: wrong
    assert mk(*_wd("lives"))["verdict"] == "refused"                                 # none
    step = {"id": "a#0", "task": "arithmetic", "stats": {"k": 0, "K": 2}}
    assert Marker([], judge=_Judge("wrong"), aliases=_Aliases())(step, "x")["gold_right"] is False
    assert Marker([], judge=_Judge("refused", "does not run"), aliases=_Aliases())(step, "x")["verdict"] == "caught"
    assert Marker([], judge=_Judge("refused", "another step's value"), aliases=_Aliases())(step, "x")["verdict"] == "refused"
    rec, prog = _wd("born in")
    assert Marker([], judge=_Judge(runs=(False, None)), aliases=_Aliases())(rec, prog)["verdict"] == "caught"


def test_the_keys_split_wd_from_plan_and_one_level_from_another():
    from selfimprove.loop import key_of
    prog = 'use vsa;\n bind frame, SEED, "$E1";\n bind frame, HOP1, "x";\n'
    wd, plan = {"task": "wd", "lang": "en", "reference": prog}, {"task": "plan", "reference": prog}
    assert key_of(wd) != key_of(plan) and key_of({**wd, "lang": "fr"}) != key_of(wd)
    step = "        create s0 : quantity;\n        assign s0 = $N1;\n        add s0, $N2;\n"
    a4 = {"task": "arithmetic", "reference": step, "stats": {"k": 0, "K": 4}}
    assert key_of(a4) != key_of({**a4, "stats": {"k": 0, "K": 5}})


def test_a_probe_with_too_few_right_programs_gives_no_gain():
    from cubbyllm.selfimprove.gaps import Rule
    from selfimprove.loop import Loop
    lp = Loop.__new__(Loop)
    lp.a = types.SimpleNamespace(probe_min_right=10)
    lp.failed = lambda key, n: [{"id": f"p{i}", "prompt": "q"} for i in range(60)]
    lp.practice = lambda probs, key, w: None                    # practice found nothing right (shuffled golds)
    lp.rights = lambda rows: []
    lp.t = types.SimpleNamespace(base=None)
    lp.event = lambda **kw: None
    assert lp.probe("k", Rule().probe_train, Rule().probe_held) is None
