"""Order 0087: a plan must use the names the question holds (``CB_COVER_NAMES=1``, off by default).

Pinned: 0083's seven "Ghana" rows (the slotted seat seeds "citizenship" for "Which country does <person> hold
citizenship in?"), as written and lowercased, are spoken with the flag off, as today, and refused with it
(``seed_is_relation``) before any fetch, whether the store or only the source's property table holds `country of
citizenship`; "who are you?" is refused (``seed_is_pronoun``) and no pronoun is a span of the slot table with the flag
on; planted cases: an unused name span is refused (``name_left_over``), an aside about the seed, a name that carries
its own parenthesis and a plan that uses every span are not. Off, the checks never run."""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cubbyllm.reasoning import cover_names as C  # noqa: E402
from cubbyllm.reasoning.planner import QuestionPlan, normalize  # noqa: E402
from cubbyllm.reasoning.slots import NameIndex, SlotEmitter, extract  # noqa: E402
from test_pipeline_plan_refusal import faithful_vm  # noqa: E402
from test_search_learn import DictSource, LookupStore  # noqa: E402

from cubbyllm.reasoning.ask import AskLoop  # noqa: E402

GHANA = ["Which country does Julius Riyadi Cardinal Darmaatmadja hold citizenship in?",       # 0083's other-words
         "Which country does Jane Nossette Jarvis hold citizenship in?",                       # rows 67, 100, 125,
         "Which country does Diana Sokolowska hold citizenship in?",                           # 127, 132, 142, 149
         "Which country does gerry mays hold citizenship in?",
         "Which country does Mr. Swirl hold citizenship in?",
         "Which country does Esaiah Pello Benson hold citizenship in?",
         "Which country does kenny rayborn hold citizenship in?"]
NAMES = ["country", "citizenship", "you", "france"]          # the served index holds the first three as names


class Seat:
    """The programs seat: writes SEED as the slot the annotated prompt gives the seed's text, then the hops."""

    def __init__(self, plans):
        self.plans, self.calls = plans, 0

    def emit(self, prompt, max_new_tokens=300, **kw):
        self.calls += 1
        ids = {t.lower(): i for i, t in re.findall(r"\[(E\d+): ([^\]]+)\]", prompt)}
        plain = re.sub(r"\[E\d+: ([^\]]+)\]", r"\1", prompt)
        if normalize(plain) not in self.plans:
            return "no plan\n"
        seed, rels = self.plans[normalize(plain)]
        sid = f"${ids[seed]}" if seed in ids else seed
        lines = [f'    bind frame, SEED, "{sid}";'] + [f'    bind frame, HOP{i + 1}, "{r}";' for i, r in enumerate(rels)]
        return "function plan(): str {\n" + "\n".join(lines) + "\n    return recover(frame, SEED);\n}\n"


class Table:
    """The source's property table (``WikidataAliases.relations``): a label or an alias -> the labels."""

    def relations(self, text):
        return ["country of citizenship"] if normalize(text) in ("country of citizenship", "citizenship") else []


class Source(DictSource):
    def __init__(self, facts, aliases=None):
        super().__init__(facts)
        self.aliases = aliases


def ghana_loop(store_holds_label=True, aliases=None, facts=None):
    store = ["ghana is the country of citizenship"] + (["france is the country of citizenship of jean"]
                                                       if store_holds_label else [])
    plans = {normalize(q): ("citizenship", ["country"]) for q in GHANA}
    loop = AskLoop(SlotEmitter(Seat(plans), index=NameIndex(NAMES)), world=LookupStore(store),
                   source=Source(facts or {}, aliases), lexicon=False, run_fn=faithful_vm([]))
    vm = faithful_vm([])

    def one_hop(src, fn):                                     # one hop's tau is 1.0; the control still recovers nothing
        out = vm(src, fn)
        return {**out, "similarity": 1.0} if out.get("similarity") else out
    loop._vm = one_hop
    return loop


@pytest.mark.parametrize("lower", [False, True])
def test_the_seven_ghana_rows_are_spoken_without_the_flag_and_refused_with_it(monkeypatch, lower):
    monkeypatch.delenv("CB_COVER_NAMES", raising=False)
    off = ghana_loop()
    for q in GHANA:
        rec = off.ask(q.lower() if lower else q)
        assert rec["verified"] and normalize(rec["answer"]) == "ghana" and rec["seed"] == "citizenship", q
    monkeypatch.setenv("CB_COVER_NAMES", "1")
    on = ghana_loop()
    for q in GHANA:
        rec = on.ask(q.lower() if lower else q)
        assert not rec["verified"] and rec["answer"] is None and rec["reason"] == "seed_is_relation", q
        assert rec["refused"]["names"]["seed"] == "citizenship"
    assert on.source.calls == []                              # refused before any fetch


def test_the_property_table_alone_names_the_relation(monkeypatch):
    """The store has fetched nothing under `country of citizenship`: the source's table still names it."""
    monkeypatch.setenv("CB_COVER_NAMES", "1")
    bare = ghana_loop(store_holds_label=False)
    assert bare.ask(GHANA[3])["verified"]                     # without the table, the store's vocabulary alone
    loop = ghana_loop(store_holds_label=False, aliases=Table())
    assert all(loop.ask(q)["reason"] == "seed_is_relation" for q in GHANA)


def test_a_pronoun_is_never_a_span_or_a_seed(monkeypatch):
    idx = NameIndex(NAMES)
    monkeypatch.delenv("CB_COVER_NAMES", raising=False)
    assert [s.text for s in extract("who are you?", idx).spans if s.kind == "E"] == ["you"]      # today
    monkeypatch.setenv("CB_COVER_NAMES", "1")
    for q in ("who are you?", "what are you?", "Who are You?"):
        assert not [s for s in extract(q, idx).spans if s.kind == "E"], q
    loop = ghana_loop(facts={"you": ["japan is the country of citizenship of you"]})
    rec = loop.ask("who are you?")                            # the host's profile shape seeds "you"
    assert rec["reason"] == "seed_is_pronoun" and rec["answer"] is None and loop.source.calls == []
    assert loop.ask("what are you?")["reason"] == "self"     # the host's self shape: about Cubby, no seed (as today)


class Known:
    def __init__(self, rels):
        self.rels = {normalize(r) for r in rels}

    def __contains__(self, r):
        return normalize(r) in self.rels

    def match(self, r):
        return None


def plan(rels, seed):
    return QuestionPlan(relations=[None] + rels[1:], tail=f"{rels[0]} of {seed}", n_hop=len(rels))


KNOWN = Known(["capital", "country of citizenship", "place of birth", "performer"])


def check(q, rels, seed, names):
    return C.refusal(q, plan(rels, seed), KNOWN, None, (extract(q, NameIndex(names)), None, None))


def test_planted_spans():
    q = "What is the capital of France near Germany?"
    assert check(q, ["capital"], "france", ["france", "germany"]) == ("name_left_over", {"seed": "france",
                                                                                         "spans": ["Germany"]})
    aside = "Where was Dina Nath Walli (an Indian watercolor artist from Srinagar) born?"
    assert check(aside, ["place of birth"], "dina nath walli", ["dina nath walli", "srinagar"]) is None
    assert check(aside, ["place of birth"], "dina nath walli", ["srinagar"]) is None        # the capitalised run
    assert check("Who is the performer of Tombo (album)?", ["performer"], "tombo (album)", ["tombo (album)"]) is None
    every = "What is the capital of the country of citizenship of Marie Curie?"
    assert check(every, ["country of citizenship", "capital"], "marie curie", ["marie curie", "capital"]) is None
    assert C.refusal(q, plan(["capital"], "it"), KNOWN) == ("seed_is_pronoun", {"seed": "it"})


def test_off_the_checks_never_run(monkeypatch):
    monkeypatch.delenv("CB_COVER_NAMES", raising=False)
    monkeypatch.setattr(C, "refusal", lambda *a, **k: pytest.fail("cover_names ran with the flag off"))
    monkeypatch.setattr(C, "slots", lambda *a, **k: pytest.fail("cover_names ran with the flag off"))
    rec = ghana_loop().ask(GHANA[1])
    assert rec["verified"] and normalize(rec["answer"]) == "ghana"
