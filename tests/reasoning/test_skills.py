"""The skill library (2026-09-24): composition rules mined from episodes, adopted only with zero
counterexamples across the whole record, applied over every bracketing and every path, and certified in
the VM with their facts. Fake VM, no subprocess."""
import json
import re

from cubbyllm.reasoning.index import TripleIndex
from cubbyllm.reasoning.skills import Episode, Library, Rule, derive, mine, rel, relate, reverify


def ep(i, rels, concl):
    return Episode(str(i), rels.split(), concl)


def echo_vm(reject=(), sim=0.93, calls=None):
    """Recovers the object each function binds; `reject` objects come back below tau."""
    def run(source, fn):
        if calls is not None:
            calls.append(fn)
        body = source.split(f"function {fn}(", 1)[1].split("}", 1)[0]
        role = re.search(r"recover\(frame, (\w+)\)", body).group(1)
        if role == "ABSENT_CTRL":
            return {"ok": True, "result": None, "similarity": None}
        obj = re.search(rf"bind frame, {role}, \"([^\"]*)\"", body).group(1)
        return {"ok": True, "result": obj, "similarity": 0.1 if obj in reject else sim}
    return run


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────────
def test_a_rule_gets_in_when_every_instance_agrees_and_enough_of_them_do():
    lib = Library()
    rep = mine([ep(1, "father sister", "aunt"), ep(2, "father sister", "aunt")], lib, "n1")
    assert Rule("father", "sister", "aunt") in lib and len(lib) == 1
    assert rep["adopted"][0]["support"] == 2 and rep["adopted"][0]["evidence"] == ["1", "2"]


def test_one_instance_is_not_enough():
    lib = Library()
    rep = mine([ep(1, "father sister", "aunt")], lib, "n1")
    assert len(lib) == 0 and rep["rejected"]["father then sister"]["why"] == "support"


def test_a_premise_whose_instances_disagree_never_gets_in():
    lib = Library()
    rep = mine([ep(1, "father sister", "aunt"), ep(2, "father sister", "aunt"), ep(3, "father sister", "mother")],
               lib, "n1")
    assert len(lib) == 0
    assert rep["rejected"]["father then sister"] == {"why": "conflict", "conclusions": {"aunt": 2, "mother": 1}}


def test_closure_learns_a_rule_seen_only_inside_longer_chains():
    """(grandfather, sister) is never stated on its own; it shows once (father, father) is known."""
    eps = [ep(1, "father father", "grandfather"), ep(2, "father father", "grandfather"),
           ep(3, "father father sister", "grandaunt"), ep(4, "father father sister", "grandaunt")]
    lib = Library()
    rep = mine(eps, lib, "n1")
    assert Rule("grandfather", "sister", "grandaunt") in lib and rep["rounds"] >= 2
    assert derive(["father", "father", "sister"], lib).conclusion == "grandaunt"


def test_a_rule_that_would_derive_a_past_episode_wrongly_is_kept_out():
    """The gate is the whole record, not the premise's own instances. Before (a, b) is known, nothing
    states (c, d) but episodes 3 and 4: -> y. Re-deriving every episode with it catches episode 5 (a b d),
    which would derive y where it states z. Once (a, b) -> c is in, episode 5 IS an instance of (c, d),
    so the next round records the premise as a conflict -- the same episode, now seen directly."""
    eps = [ep(1, "a b", "c"), ep(2, "a b", "c"), ep(3, "c d", "y"), ep(4, "c d", "y"), ep(5, "a b d", "z")]
    lib = Library()
    one = mine(eps, lib, "n1", max_rounds=1)
    assert Rule("a", "b", "c") in lib and Rule("c", "d", "y") not in lib
    assert one["rejected"]["c then d"] == {"why": "counterexample", "conclusion": "y", "episode": "5",
                                           "states": "z", "derived": "y"}
    rep = mine(eps, lib, "n1")
    assert Rule("c", "d", "y") not in lib
    assert rep["rejected"]["c then d"] == {"why": "conflict", "conclusions": {"y": 2, "z": 1}}
    assert reverify(eps, lib) == []


def test_a_new_counterexample_retires_the_weakest_rule_it_used_and_nothing_is_deleted(tmp_path):
    path = tmp_path / "skills.jsonl"
    night1 = [ep(1, "a b", "c"), ep(2, "a b", "c"), ep(3, "a b", "c"), ep(4, "c d", "y"), ep(5, "c d", "y")]
    lib = Library(path)
    mine(night1, lib, "n1")
    assert Rule("a", "b", "c") in lib and Rule("c", "d", "y") in lib
    rep = mine(night1 + [ep(6, "a b d", "z")], lib, "n2")
    assert Rule("c", "d", "y") not in lib and Rule("a", "b", "c") in lib      # support 2 goes before support 3
    assert [r["action"] for r in rep["retired"]] == ["retire"] and "episode 6" in rep["retired"][0]["reason"]
    again = Library(path)                                                    # replayed from the ledger
    assert Rule("c", "d", "y") not in again and ("c", "d", "y") in again.retired
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["action"] for x in lines] == ["adopt", "adopt", "retire"]      # the adoption is still on record
    mine(night1 + [ep(6, "a b d", "z")], again, "n3")
    assert Rule("c", "d", "y") not in again                                  # a retired rule does not come back


def test_mining_the_same_record_twice_writes_nothing(tmp_path):
    path = tmp_path / "skills.jsonl"
    eps = [ep(1, "father sister", "aunt"), ep(2, "father sister", "aunt")]
    mine(eps, Library(path), "n1")
    before = path.read_text(encoding="utf-8")
    rep = mine(eps, Library(path), "n1")
    assert rep["adopted"] == [] and path.read_text(encoding="utf-8") == before


# ── composing ────────────────────────────────────────────────────────────────────────────────────────
def lib_of(*rules):
    lib = Library()
    for r in rules:
        lib.adopt(Rule(*r), support=2, evidence=[], night="t")
    return lib


def test_derive_tries_every_bracketing():
    """Only a (b c) first bracketing completes; a left fold would call this unknown."""
    lib = lib_of(("b", "c", "x"), ("a", "x", "y"))
    d = derive(["a", "b", "c"], lib)
    assert d.status == "derived" and d.conclusion == "y" and [str(s) for s in d.steps] == [
        "the c of the b is the x", "the x of the a is the y"]


def test_bracketings_that_disagree_are_a_split_not_a_choice():
    lib = lib_of(("a", "b", "p"), ("p", "c", "q"), ("b", "c", "r"), ("a", "r", "s"))
    d = derive(["a", "b", "c"], lib)
    assert d.status == "split" and sorted(d.conclusions) == ["q", "s"] and d.conclusion is None


def test_a_single_relation_is_itself_and_nothing_composes_to_no_rule():
    assert derive(["father"], Library()).conclusion == "father"
    assert derive(["father", "sister"], Library()).status == "no_rule"


# ── relating two people ──────────────────────────────────────────────────────────────────────────────
F1 = "michael is the father of donald"
F2 = "dorothy is the sister of michael"
AUNT = ("father", "sister", "aunt")


def paths(facts, a="donald", b="dorothy", **kw):
    return TripleIndex(facts).paths(a, b, **kw)


def test_a_composed_relation_is_spoken_after_the_vm_recovers_the_facts_and_the_rule_step():
    calls = []
    ps, over = paths([F1, F2])
    r = relate(ps, lib_of(AUNT), echo_vm(calls=calls), overflow=over)
    assert r["verified"] and r["answer"] == "aunt" and r["reason"] is None
    assert r["paths"][0]["steps"] == ["the sister of the father is the aunt"] and r["paths"][0]["certified"]
    assert calls == ["solve", "hop_2", "hop_3", "control"]                  # two facts, one step, the control


def test_a_step_the_vm_does_not_recover_is_not_spoken():
    ps, _ = paths([F1, F2])
    r = relate(ps, lib_of(AUNT), echo_vm(reject={"aunt"}))
    assert not r["verified"] and r["answer"] is None and r["reason"] == "vm_verify_failed"


def test_no_rule_and_no_path_are_refusals_with_their_own_reasons():
    ps, _ = paths([F1, F2])
    assert relate(ps, Library(), echo_vm())["reason"] == "no_rule"
    ps, _ = paths([F1])
    assert relate(ps, lib_of(AUNT), echo_vm())["reason"] == "no_path"


def test_paths_that_agree_speak_and_paths_that_disagree_refuse():
    both = [F1, F2, "jane is the mother of donald", "dorothy is the sister in law of jane"]
    ps, _ = paths(both)
    assert len(ps) == 2
    agree = relate(ps, lib_of(AUNT, ("mother", "sister in law", "aunt")), echo_vm())
    assert agree["verified"] and agree["answer"] == "aunt"
    split = relate(ps, lib_of(AUNT, ("mother", "sister in law", "cousin")), echo_vm())
    assert not split["verified"] and split["reason"] == "rule_split"


def test_a_path_the_library_cannot_compose_does_not_block_one_it_can():
    ps, _ = paths([F1, F2, "jane is the mother of donald", "dorothy is the sister in law of jane"])
    r = relate(ps, lib_of(AUNT), echo_vm())
    assert r["verified"] and [p["status"] for p in r["paths"]] == ["derived", "no_rule"]


def test_a_stated_fact_is_its_own_answer_and_too_many_paths_are_refused():
    ps, _ = paths(["dorothy is the aunt of donald"])
    r = relate(ps, Library(), echo_vm(sim=1.0))                  # one binding in the frame: the 1-hop bar, 1.0
    assert r["verified"] and r["answer"] == "aunt" and r["paths"][0]["steps"] == []
    ps, over = paths([F1, F2, "jane is the mother of donald", "dorothy is the sister in law of jane"], limit=1)
    assert over and relate(ps, lib_of(AUNT), echo_vm(), overflow=over)["reason"] == "relation_paths_overflow"


def test_paths_are_simple_and_shortest_first():
    facts = ["b is the r of a", "c is the r of b", "a is the r of c", "c is the s of a"]
    ps, over = TripleIndex(facts).paths("a", "c")
    assert not over and [[t.rel for t in p] for p in ps] == [["s"], ["r", "r"]]


# ── the night ────────────────────────────────────────────────────────────────────────────────────────
def test_a_spoken_relation_gold_disagrees_with_is_a_defect_and_never_an_episode_in_the_hippocampus(tmp_path):
    from cubbyllm.reasoning import sleep as S
    rec = {"question": "How is Dorothy related to Donald?", "kind": "relation", "verified": True, "answer": "uncle",
           "reason": None, "plan": ["father", "sister"], "seed": "donald", "gold": "aunt",
           "trace": ["michael is the father of donald", "dorothy is the sister of michael"],
           "paths": [["father", "sister"]]}
    day = tmp_path / "d.jsonl"
    day.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    rep = S.sleep([day], tmp_path / "s", night="n1", trusted={"wikidata"})
    assert rep["audit"]["wrong"] == 1 and rep["replay"]["episodes_written"] == 0
    assert rep["skills"]["episodes_total"] == 1                  # gold's relation is the episode, not the answer


def test_a_premise_kept_out_is_shown_to_the_host_most_one_sided_first(tmp_path):
    """114 against 2 is a different question from 50 against 50: the night lists what it kept out, with
    the counts, so the host can decide whether the minority is label noise. The loop decides nothing."""
    from cubbyllm.reasoning import sleep as S
    rows = ([{"kind": "relation", "question": f"q{i}", "verified": False, "reason": "no_rule",
              "paths": [["husband", "father"]], "gold": "father in law"} for i in range(5)]
            + [{"kind": "relation", "question": "odd", "verified": False, "reason": "no_rule",
                "paths": [["husband", "father"]], "gold": "father"}]
            + [{"kind": "relation", "question": f"s{i}", "verified": False, "reason": "no_rule",
                "paths": [["son", "grandfather"]], "gold": g} for i, g in enumerate(["father", "father in law"])])
    day = tmp_path / "d.jsonl"
    day.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = S.sleep([day], tmp_path / "s", night="n1", trusted={"wikidata"})
    assert rep["skills"]["rules"] == 0
    seen = [json.loads(x) for x in (tmp_path / "s" / "skills_contested.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(c["premise"], c["majority"], c["share"]) for c in seen] == [
        ("husband then father", "father in law", 0.833), ("son then grandfather", "father", 0.5)]
    assert "contested, kept out: `husband then father`" in (tmp_path / "s" / "nights" / "n1" / "report.md").read_text(
        encoding="utf-8")


def test_the_host_can_adopt_a_contested_rule_by_discounting_its_minority(tmp_path):
    """The host's word, on the ledger: the rule goes in, the minority stays on record as discounted, and
    the next night neither retires the rule nor counts the minority again."""
    from cubbyllm.reasoning.skills import host_adopt
    path = tmp_path / "skills.jsonl"
    eps = [ep(i, "husband father", "father in law") for i in range(5)] + [ep(9, "husband father", "father")]
    lib = Library(path)
    mine(eps, lib, "n1")
    assert len(lib) == 0
    out = host_adopt(eps, lib, ("husband", "father"), "father-in-law", "a CLUTRR label error", "n1")
    assert out["discounted"] == ["9"] and Rule("husband", "father", "father in law") in lib
    again = Library(path)
    rep = mine(eps, again, "n2")
    assert Rule("husband", "father", "father in law") in again and rep["retired"] == [] and "9" in again.discounted
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [(x["action"], x.get("host")) for x in lines] == [("discount", None), ("adopt", True)]


def test_the_host_overrides_a_minority_never_the_record():
    """An episode the rule would derive wrongly through a LONGER chain is not a minority of the premise;
    the host cannot discount it by adopting the rule, and the adoption is refused."""
    from cubbyllm.reasoning.skills import host_adopt
    eps = [ep(1, "a b", "c"), ep(2, "a b", "c"), ep(3, "y e", "r"), ep(4, "y e", "r"),
           ep(5, "c d", "y"), ep(6, "c d", "y"), ep(7, "c d", "y"), ep(8, "c d", "z"), ep(10, "a b d e", "q")]
    lib = Library()
    mine(eps, lib, "n1")
    assert Rule("a", "b", "c") in lib and Rule("y", "e", "r") in lib and len(lib) == 2
    out = host_adopt(eps, lib, ("c", "d"), "y", "noise", "n1")
    assert "refused" in out and "episode 10" in out["refused"]
    assert len(lib) == 2 and lib.discounted == {}


# ── terms at different grain ─────────────────────────────────────────────────────────────────────────
TAX = {"father's brother": ["uncle", "relative"], "mother's brother": ["uncle", "relative"], "uncle": ["relative"],
       "paternal grandfather": ["grandfather", "relative"], "grandfather": ["relative"], "stepson": ["relative"],
       "son": ["relative"]}


def test_a_finer_term_never_contradicts_a_coarser_one_and_the_rule_concludes_what_was_stated():
    """'uncle' three times and "father's brother" once: comparable, so no conflict; the rule concludes the
    finest term enough instances CONFIRM (uncle) -- never 'relative', which nobody stated."""
    eps = [ep(i, "father brother", "uncle") for i in range(3)] + [ep(9, "father brother", "father's brother")]
    lib = Library(entails=TAX)
    rep = mine(eps, lib, "n1")
    assert Rule("father", "brother", "uncle") in lib and rep["adopted"][0]["support"] == 4
    eps += [ep(10, "father brother", "father's brother")]
    lib2 = Library(entails=TAX)
    mine(eps, lib2, "n1")
    assert lib2.conclusion("father", "brother") == rel("father's brother")   # two now confirm the finer term


def test_incomparable_terms_are_still_a_conflict():
    eps = [ep(1, "wife son", "son"), ep(2, "wife son", "son"), ep(3, "wife son", "stepson")]
    lib = Library(entails=TAX)
    rep = mine(eps, lib, "n1")
    assert len(lib) == 0 and rep["rejected"]["wife then son"]["why"] == "conflict"


def test_comparable_conclusions_speak_the_finest_and_incomparable_ones_split():
    lib = Library(entails=TAX)
    for r in (("father", "father", "grandfather"), ("dad", "father", "paternal grandfather"),
              ("mother", "brother", "uncle"), ("mom", "brother", "mother's brother")):
        lib.adopt(Rule(*r), support=2, evidence=[], night="t")
    ix = TripleIndex(["b is the father of a", "c is the father of b", "b is the dad of a"])
    ps, _ = ix.paths("a", "c")
    r = relate(ps, lib, echo_vm())
    assert r["verified"] and r["answer"] == "paternal grandfather" and len(ps) == 2
    assert derive(["mother", "brother"], lib).conclusion == "uncle"


# ── the other way round ──────────────────────────────────────────────────────────────────────────────
def test_an_inverse_is_adopted_on_agreeing_sources_and_a_conflict_stays_out():
    from cubbyllm.reasoning.skills import InverseEpisode, mine_inverses
    lib = Library(entails=TAX)
    eps = [InverseEpisode(str(i), "father", "male", "son") for i in range(3)] + [
        InverseEpisode("u1", "uncle", "male", "nephew"), InverseEpisode("u2", "uncle", "male", "grandson")]
    rep = mine_inverses(eps, lib, "n1")
    assert lib.inverse_of("father", "male") == "son" and lib.inverse_of("father", "female") is None
    assert rep["rejected"]["uncle (male)"]["why"] == "conflict"


def test_an_inverse_is_never_borrowed_from_a_coarser_term():
    """The hierarchy says what does not contradict, not what is true: borrowing the inverse of a coarser
    term spoke 'ancestor' for a grand-nephew on Wikidata (exp_r35, first run)."""
    lib = Library(entails=TAX)
    lib.adopt_inverse("uncle", "male", "nephew", 3, [], "n1")
    assert lib.inverse_of("uncle", "male") == "nephew" and lib.inverse_of("father's brother", "male") is None


def test_relate_back_speaks_the_inverse_certified_with_the_sex_fact():
    """The store leads only from Dorothy to Donald ('donald is the nephew of dorothy'); asked how Dorothy is
    related to Donald, the path composes to nephew and the inverse for a woman turns it into aunt."""
    from cubbyllm.reasoning.planner import Triple
    from cubbyllm.reasoning.skills import relate_back
    lib = Library(entails=TAX)
    lib.adopt_inverse("nephew", "female", "aunt", 3, [], "n1")
    ix = TripleIndex(["donald is the nephew of dorothy", "female is the sex or gender of dorothy"])
    assert ix.paths("donald", "dorothy")[0] == []                  # nothing leads from Donald to Dorothy
    back, _ = ix.paths("dorothy", "donald")
    calls = []
    sex = ("female", Triple(obj="female", rel="sex or gender", subj="dorothy"))
    r = relate_back(back, sex, lib, echo_vm(calls=calls))
    assert r["verified"] and r["answer"] == "aunt" and r["direction"] == "inverse"
    assert calls == ["solve", "hop_2", "hop_3", "control"]         # the fact, the sex fact, the inverse step
    assert relate_back(back, None, lib, echo_vm())["reason"] == "no_rule"   # no sex, no inverse
    assert relate_back(back, ("male", sex[1]), lib, echo_vm())["reason"] == "no_rule"
