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


def _probe_loop(right_programs: int, base_ok: bool, probe_ok: bool):
    """A Loop with its GPU parts stubbed: 60 failed problems and 120 others of one key."""
    from cubbyllm.selfimprove.ledger import Ledger
    from selfimprove.loop import Loop
    lp = Loop.__new__(Loop)
    lp.a = types.SimpleNamespace(probe_min_right=10, probe_steps=10, batch=16, lr=1e-4)
    lp.led = Ledger(window=2000)
    lp.by_id = {}
    for i in range(180):
        lp.by_id[f"p{i}"] = {"id": f"p{i}", "prompt": "q"}
        lp.led.add("k", "refused" if i < 60 else "right", None, request=f"p{i}")
    lp.practice = lambda probs, key, w: None
    lp.rights = lambda rows: [("q", "prog")] * right_programs
    trained = []
    lp.t = types.SimpleNamespace(base="BASE", use=lambda w: None, state=lambda: "PROBE",
                                 train=lambda *a, **k: trained.append(k["steps"]))
    lp.read = lambda rows, w: [{"verdict": "right", "gold_right": base_ok}] * len(rows)
    lp.rate = lambda rows, w: float(probe_ok)
    lp.base_ok, lp.event = {}, (lambda **kw: None)
    return lp, trained


def test_a_probe_with_too_few_right_programs_gives_no_gain():
    from cubbyllm.selfimprove.gaps import Rule
    lp, trained = _probe_loop(right_programs=3, base_ok=False, probe_ok=True)   # shuffled golds: practice finds ~none
    assert lp.probe("k", Rule().probe_train, 100) is None and trained == []


def test_the_probe_reads_the_held_right_rate_as_the_gate_does():
    lp, trained = _probe_loop(right_programs=40, base_ok=False, probe_ok=True)
    assert lp.probe("k", 100, 100) == 1.0 and trained == [10]                   # one pass over 40 rows of 16: 10 steps
    lp, _ = _probe_loop(right_programs=40, base_ok=True, probe_ok=True)
    assert lp.probe("k", 100, 100) == 0.0                                       # no rise, no gap


def test_practice_tries_more_only_where_eight_find_too_few_and_counts_each_problems_tries():
    from cubbyllm.selfimprove.mistakes import MistakeMemory
    from selfimprove.loop import Loop
    lp = Loop.__new__(Loop)
    lp.a = types.SimpleNamespace(tries=8, bootstrap_min=10, temperature=1.0)
    lp.mm, lp.result, lp.event = MistakeMemory(32), {"tries": {}}, (lambda **kw: None)
    asked = []

    def programs(prompts, caps, k, temperature, rows):
        asked.append(k)
        last = "R" if k > 8 else "w-last"                                       # right only past the 8 first tries
        return [[f"w{j}" for j in range(k - 1)] + [last] for _ in prompts]
    lp.t = types.SimpleNamespace(use=lambda w: None, programs=programs)
    lp.mark = lambda r, p: {"verdict": "right", "gold_right": p == "R", "why": ""}
    probs = [{"id": f"p{i}", "prompt": "q", "task": "wd"} for i in range(5)]
    lp.practice(probs, "k", None)
    assert asked == [8, 24] and lp.result["tries"]["k"][0]["right_after_max"] == 5   # 8 found 5 < 10: on to 32
    lp.practice(probs, "k", None)
    assert asked == [8, 24]                                                     # solved: nothing more
    assert all(lp.mm.problems[p["id"]].attempts == 32 for p in probs)                 # 8, then up to 32 in all


def test_the_noise_copy_has_no_stop_row_and_no_neutral_value():
    from selfimprove import stream as S
    from selfimprove.mark import Marker
    recs = []
    for fam, task, lang, sub in (("wd/en", "wd", "en", ""), ("wd/fr", "wd", "fr", ""), ("a4", "arithmetic", None, "steps=4"),
                                 ("a2", "arithmetic", None, "steps=2")):
        for i in range(40):
            recs.append({"id": f"{fam}-{i}#0", "task": task, "lang": lang, "subtype": sub, "split": "train",
                         "gold": str(i), "prompt": "q", "program": "return $S1;" if i % 2 else "add s0, $N1;",
                         "stats": {"k": 0, "K": 2}})
    _, st, _ = S.build(recs, 4, "arithmetic/steps=2", 10, 5, 30, 0)
    copy = [r for r in st if r["shuffled"]]
    assert copy and not any(r["program"].startswith("return") for r in copy)
    seen = []

    class J:
        def attempt_step(self, row, text):
            seen.append(row["plan_values"])
            return {"outcome": "correct", "why": ""}
    mk = Marker(recs, judge=J(), aliases=_Aliases())
    mk(copy[0], "x")
    mk({**copy[0], "shuffled": False, "id": "a2-0#0"}, "x")
    assert seen[0] == [] and seen[1] != []
