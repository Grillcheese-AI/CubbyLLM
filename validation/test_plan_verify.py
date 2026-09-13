"""Pins for cubbyllm.reasoning.plan_verify — the plan disposer. Fake vocab only:
no store, no encoder, no VM. Run: python -m pytest validation/test_plan_verify.py -q

What the pins protect (each one was chosen because a mutation that breaks it
changes a number in exp_r6_plan_verify):
  * covers() is exact on the chain body and order-blind on a 1-hop inversion
  * unknown_relation fires on a COMPOUND relation the store does not hold, and
    on the tail's relation prefix — longest-known-prefix, never shortest
  * a verdict is False iff it carries a reason, and the reason names the first
    failing check (coverage before vocabulary)
  * segment() is exhaustive, returns [] when no reading is answerable, and
    never proposes a plan verify_plan() would reject
"""
from __future__ import annotations

import pathlib
import sys

VAL = pathlib.Path(__file__).resolve().parent
ROOT = VAL.parent
for p in (str(ROOT), str(VAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.plan_verify import (  # noqa: E402
    PlanVerdict, StoreRelations, canonical_body, covers, segment, tail_relation, verify_plan)
from cubbyllm.reasoning.planner import QuestionPlan, parse_question  # noqa: E402

KNOWN = frozenset({
    "country", "capital", "country of citizenship", "director of photography",
    "award received", "genre", "instance", "administrative territorial entity",
    "office held by head of government",
})


class Vocab:
    """The minimum KnownRelations: normalized membership + the walk's paraphrase tier."""
    def __init__(self, rels):
        from cubbyllm.reasoning.planner import normalize, relation_matches
        self._r = sorted({normalize(r) for r in rels})
        self._n, self._rm = normalize, relation_matches
    def __contains__(self, rel):
        return self._n(rel) in self._r
    def match(self, rel):
        if rel in self:
            return self._n(rel)
        return next((r for r in self._r if self._rm(rel, r)), None)
    def words(self):
        from cubbyllm.reasoning.plan_verify import _relation_words
        return _relation_words(self._r)


V = Vocab(KNOWN)


def plan(q):
    p = parse_question(q)
    assert p is not None, q
    return p


# ---- covers ---------------------------------------------------------------

def test_grammar_plan_covers_its_own_question():
    q = "What is the capital of the country of citizenship of Cynthia Basinet?"
    assert covers(q, plan(q))
    assert canonical_body(plan(q)) == "capital of the country of citizenship of cynthia basinet"


def test_dropped_hop_does_not_cover():
    q = "What is the capital of the country of citizenship of Cynthia Basinet?"
    p = plan(q)
    shorter = QuestionPlan(relations=[None], tail=p.tail, n_hop=1)      # emitter dropped 'capital'
    # a 1-hop is checked by halves, and both halves ARE in the question -> covers must be judged
    # on the chain the question actually states; the disposer catches this via n_hop+vocab, not
    # coverage alone. Pin the honest behaviour: a 1-hop reading of a 2-hop question still
    # 'covers' (its parts are present) -- the vocabulary check is what has to bite.
    assert covers(q, shorter)
    invented = QuestionPlan(relations=[None, "capital", "population"], tail=p.tail, n_hop=3)
    assert not covers(q, invented)                                       # emitter invented a hop
    assert verify_plan(q, invented, V).reason == "plan_does_not_cover_question"


def test_one_hop_inversion_is_order_blind():
    q = "What genre is Tombo (album)?"
    p = plan(q)
    assert p.n_hop == 1 and p.tail == "genre of Tombo (album)"
    assert covers(q, p)
    # ... but never entity-blind: the same relation asked of another entity is a different question
    assert not covers(q, QuestionPlan(relations=[None], tail="genre of Kind of Blue", n_hop=1))
    assert not covers(q, QuestionPlan(relations=[None], tail="label of Tombo (album)", n_hop=1))


# ---- vocabulary -----------------------------------------------------------

def test_compound_relation_is_rejected():
    q = "What is the award received by the director of photography of Some Film?"
    p = plan(q)                       # grammar: 1 hop, tail = the whole compound
    v = verify_plan(q, p, V)
    assert not v and v.reason == "unknown_relation"
    assert v.unknown_relations == ["award received by the director of photography"]


def test_disposer_has_exactly_the_walks_tolerance():
    """hop>=1: the walk accepts 'office held by THE head of government' against the
    store's 'office held by head of government' via relation_matches -- so must we,
    and the verdict must SAY it was a paraphrase. hop 0 was exact in the walk until
    2026-09-11 (lever 1) and this pin refused a paraphrased TAIL; the walk's hop 0
    now has the same two tiers, so the same paraphrase at the tail is tolerated AND
    recorded -- the disposer follows the walk, never leads it."""
    q = "Which country is the country of the office held by the head of government of the country of citizenship of X?"
    v = verify_plan(q, plan(q), V)
    assert v.ok and v.reason is None
    assert v.paraphrased == [("office held by the head of government", "office held by head of government")]
    assert v.unknown_relations == []
    # the same paraphrase at hop 0 (the tail) is tolerated the same way, and recorded
    q0 = "What is the office held by the head of government of France?"
    v0 = verify_plan(q0, plan(q0), V)
    assert v0.ok and v0.reason is None and v0.tail_relation == "office held by the head of government"
    assert v0.paraphrased == [("office held by the head of government", "office held by head of government")]
    # and a compound relation is still not a paraphrase of its atomic part (Jaccard < 0.6)
    qc = "What is the instance of the award received by the creator of Some Show?"
    vc = verify_plan(qc, plan(qc), V)
    assert not vc.ok and vc.unknown_relations == ["award received by the creator"] and vc.paraphrased == []


def test_tail_relation_takes_the_longest_known_prefix():
    assert tail_relation("country of citizenship of henry allcock", V) == "country of citizenship"
    assert tail_relation("country of henry allcock", V) == "country"
    assert tail_relation("population of henry allcock", V) is None


def test_ok_verdict_carries_no_reason_and_is_truthy():
    q = "What is the capital of the country of citizenship of Cynthia Basinet?"
    v = verify_plan(q, plan(q), V)
    assert v and v.ok and v.reason is None and v.unknown_relations == []
    assert v.tail_relation == "country of citizenship"


def test_coverage_failure_wins_over_vocabulary():
    q = "What is the capital of the country of citizenship of Cynthia Basinet?"
    bad = QuestionPlan(relations=[None, "population"], tail="nope of Cynthia Basinet", n_hop=2)
    v = verify_plan(q, bad, V)
    assert v.reason == "plan_does_not_cover_question" and not v.covers
    assert v.unknown_relations           # still reported, never hidden by the earlier failure


def test_store_relations_is_built_from_facts_that_parse():
    sr = StoreRelations([
        "paris is the capital of france",
        "france is the country of citizenship of jean",
        "this line does not parse",
    ])
    assert sr.n_parsed == 2 and len(sr) == 2
    assert "Capital" in sr and "country of citizenship" in sr and "population" not in sr


# ---- segment --------------------------------------------------------------

def test_segment_is_exhaustive_and_only_proposes_verifiable_plans():
    body = "capital of the country of citizenship of henry allcock"
    # exhaustive: the 2-hop reading AND the 1-hop reading whose entity swallows the joint
    # ("capital" of "the country of citizenship of henry allcock") -- both verify, so both
    # are returned; the disposer must not pick.
    plans = segment(body, V)
    assert sorted(p.n_hop for p in plans) == [1, 2]
    assert all(verify_plan(f"What is the {body}?", p, V).ok for p in plans)
    # strict: the entity may not contain ' of the ' -> the split is decidable
    strict = segment(body, V, joint_in_entity=False)
    assert [p.n_hop for p in strict] == [2]
    assert strict[0].relations == [None, "capital"]
    assert strict[0].tail == "country of citizenship of henry allcock"
    # and the one reading strict throws away is exactly the joint-in-entity one
    assert {p.tail for p in plans} - {p.tail for p in strict} == {body}


def test_segment_returns_nothing_when_no_reading_is_answerable():
    assert segment("award received by the director of photography of some film", V) == []


def test_segment_respects_max_hop():
    body = "capital of the capital of the capital of the capital of the country of x"
    assert segment(body, V, max_hop=4, joint_in_entity=False) == []      # needs 5 hops
    assert [p.n_hop for p in segment(body, V, max_hop=5, joint_in_entity=False)] == [5]
    # exhaustive mode still honours the bound: nothing deeper than max_hop
    assert max(p.n_hop for p in segment(body, V, max_hop=3)) <= 3


def test_verdict_is_frozen():
    v = PlanVerdict(ok=True, covers=True)
    try:
        v.ok = False                                   # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("PlanVerdict must be immutable")


# ---- covers v3 (2026-09-11, lever 2): either reading order, either wording of a paraphrase ----

def test_possessive_inner_first_reading_covers():
    """exp_r9: 'E's P1 -- what is its P2?' states the entity first and the hops
    inner-first; v2 refused all 180. The same chain in either order is covered."""
    p = QuestionPlan(relations=[None, "capital"], tail="country of citizenship of jean", n_hop=2)
    assert covers("What is the capital of the country of citizenship of Jean?", p, V)
    assert covers("Jean's country of citizenship -- what is its capital?", p, V)
    # a SWAPPED chain is not the same chain, in either order
    swapped = QuestionPlan(relations=[None, "country of citizenship"], tail="capital of jean", n_hop=2)
    assert not covers("Jean's country of citizenship -- what is its capital?", swapped, V)
    assert not covers("What is the capital of the country of citizenship of Jean?", swapped, V)


def test_inner_first_reading_still_refuses_a_dropped_hop():
    p1 = QuestionPlan(relations=[None], tail="country of citizenship of jean", n_hop=1)
    assert not covers("Jean's country of citizenship -- what is its capital?", p1, V)     # 'capital' left over


def test_paraphrased_relation_may_appear_under_the_stores_wording():
    """gen 2, arm C: 'Which languages spoken, written or signed by X?' planned as
    `languages spoken written signed` -- accepted as a paraphrase by the vocabulary
    tier, then refused for coverage because the plan's wording is not a substring.
    v3 accepts either wording; a dropped hop beside it still fails."""
    class V2(Vocab):
        def match(self, rel, reused_only=False):
            return Vocab.match(self, rel)
    v = V2(KNOWN | {"languages spoken written or signed"})
    p = QuestionPlan(relations=[None], tail="languages spoken written signed of manfred rusing", n_hop=1)
    q = "Which languages spoken, written or signed by Manfred Rusing?"
    assert covers(q, p, v)
    verdict = verify_plan(q, p, v)
    assert verdict.ok and verdict.paraphrased == [("languages spoken written signed", "languages spoken written or signed")]
    q2 = "Which languages spoken, written or signed by the capital of Manfred Rusing?"
    assert not covers(q2, p, v)                                                   # 'capital' dropped


def test_a_number_left_over_is_an_unbound_constraint():
    """exp_r11 (2026-09-11): 'As of 2022, what is the population of Mersin Province?'
    planned as `population of mersin province` covered under v3 -- '2022' is not a
    relation word -- and the walk spoke one census for a year the store cannot
    check. A leftover number is a constraint the plan did not bind: not covered."""
    v = Vocab(KNOWN | {"population"})
    p = QuestionPlan(relations=[None], tail="population of mersin province", n_hop=1)
    assert covers("What is the population of Mersin Province?", p, v)
    assert not covers("As of 2022, what is the population of Mersin Province?", p, v)
    assert not covers("What was the population of Mersin Province in 1990?", p, v)
    # a number INSIDE the entity is the entity's, not a constraint
    p2 = QuestionPlan(relations=[None], tail="primary classification of 1881 in india", n_hop=1)
    assert covers("What is the primary classification of 1881 in India?", p2, Vocab(KNOWN | {"primary classification"}))


def test_covers_v4_a_parenthetical_is_an_aside_about_the_entity_not_a_dropped_hop():
    """exp_r14 (2026-09-12): 'Dina Nath Walli (an Indian watercolor artist and poet from
    Srinagar city)' -- the aside's words ('artist', 'city') are relation words in the wiki
    world and read as a dropped hop. Removed before the residual check; a real dropped hop
    outside a parenthesis still fails."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, covers
    store = ["1932 is the date of birth of dina nath walli", "srinagar is the city of dina nath walli",
             "painting is the artist of dina nath walli", "srinagar is the city of jean"]
    known = StoreRelations(store)
    plan = QuestionPlan(relations=[None], tail="date of birth of dina nath walli", n_hop=1)
    aliases = {"date of birth": ["born"]}
    assert covers("In which year was Dina Nath Walli (an Indian watercolor artist and poet from Srinagar city) born?", plan, known, aliases)
    assert not covers("In which year was the artist of Dina Nath Walli born?", plan, known, aliases)


def test_covers_v5_a_verb_form_outer_hop_follows_the_entity_the_inner_hop_precedes_it():
    """The gen-3 builder (2026-09-13): 'When was the father of X born?' states hop 1 as a
    noun phrase before the entity and hop 2 as a verb after it -- neither the answer-first
    nor the entity-first reading. The mixed reading accepts inner hops, the entity, then the
    rest in walk order; the residual rule is unchanged, so a dropped hop still fails."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, covers
    store = ["jean is the father of marie", "1950 is the born of jean", "lyon is the spouse of jean", "paris is the capital of france"]
    known = StoreRelations(store)
    two = QuestionPlan(relations=[None, "born"], tail="father of marie", n_hop=2)
    assert covers("When was the father of Marie born?", two, known)
    assert covers("Who is the father of Marie married to?", QuestionPlan(relations=[None, "married"], tail="father of marie", n_hop=2),
                  known, aliases={"spouse": ["married"]})
    one = QuestionPlan(relations=[None], tail="born of marie", n_hop=1)
    assert not covers("When was the father of Marie born?", one, known)          # 'father' is a dropped hop


def test_a_relation_that_ends_in_of_splits_at_the_overlapping_of():
    """hdc (2026-09-13): Wikidata's commonest labels end in 'of' -- 'instance of', 'member of',
    'part of'. In a tail or a fact, 'instance of of X' has two ' of 's that overlap at one
    space; `str.split(' of ')` sees only the first and reads 'instance' | 'of X'. Every
    ' of ' split -- the held tier, the paraphrase tier, the question-decided tier, the fact
    parse, the hop-0 readings -- now offers the overlapping prefix too."""
    from cubbyllm.reasoning.plan_verify import split_tail
    from cubbyllm.reasoning.planner import of_prefixes, parse_fact, tail_splits
    assert of_prefixes("instance of of Bari station") == ["instance of", "instance"]
    assert of_prefixes("country of citizenship of x of y") == ["country of citizenship of x", "country of citizenship", "country"]
    known = StoreRelations([])
    known.declare("instance of"); known.add("Train station is the instance of of Bari station")
    known.declare("member of"); known.add("imf is the member of of united stated")
    assert "instance of" in known
    assert tail_relation("instance of of Bari station", known) == "instance of"
    assert split_tail("instance of of Bari station", known) == ("instance of", "Bari station")
    assert split_tail("member of of united stated", known, "What is united stated a member of?") == ("member of", "united stated")
    # unheld, the question decides -- and never an entity that begins with 'of'
    assert split_tail("part of of the whole thing", None, "What is the whole thing part of?") == ("part of", "the whole thing")
    t = parse_fact("Train station is the instance of of Bari station", known={"instance of"})
    assert (t.rel, t.subj, t.obj) == ("instance of", "Bari station", "Train station")
    assert tail_splits("instance of of Bari station")[0] == ("instance of", "bari station")
    plan = QuestionPlan(relations=[None], tail="instance of of Bari station", n_hop=1)
    assert verify_plan("What is the instance of Bari station?", plan, known).ok


def test_covers_v6_a_split_wording_is_claimed_word_by_word_and_a_dropped_hop_still_fails():
    """hdc (2026-09-13): 3,463 of 8,840 rephrased questions worded a relation with all its content
    words, not contiguously -- 'Which country does X hold citizenship in?'. The split tier claims
    each word outside every other claim; the residual rule is unchanged, so a hop the plan dropped
    is still refused, and a wording with a word missing is not a split wording."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, covers
    store = ["france is the country of citizenship of dren hoxha", "albania is the country of dren hoxha",
             "pierre is the father of dren hoxha", "french is the languages spoken, written or signed of dren hoxha",
             "human is the instance of of dren hoxha"]
    known = StoreRelations(store)
    one = QuestionPlan(relations=[None], tail="country of citizenship of dren hoxha", n_hop=1)
    assert covers("Which country does Dren Hoxha hold citizenship in?", one, known)
    assert covers("What languages are spoken, written, or signed by Dren Hoxha?",
                  QuestionPlan(relations=[None], tail="languages spoken, written or signed of dren hoxha", n_hop=1), known)
    # the dropped hop: 'father' is a relation word left over
    assert not covers("Which country does the father of Dren Hoxha hold citizenship in?", one, known)
    two = QuestionPlan(relations=[None, "country of citizenship"], tail="father of dren hoxha", n_hop=2)
    assert covers("Which country does the father of Dren Hoxha hold citizenship in?", two, known)
    # a word missing is not a split wording: 'country' alone does not word `country of citizenship`
    assert not covers("Which country is Dren Hoxha from?", one, known)
    # a relation worded contiguously but out of order is a swapped chain, never a split wording
    swapped = QuestionPlan(relations=[None, "father"], tail="country of citizenship of dren hoxha", n_hop=2)
    assert not covers("What is the country of citizenship of the father of Dren Hoxha?", swapped, known)
    # the documented limit: a split wording carries no order, so the swapped chain of a split question
    # passes coverage and it is the walk that refuses it (no 'father' of a country in any store)
    assert covers("Which country does the father of Dren Hoxha hold citizenship in?", swapped, known)
    # a where-question's verb of location is its frame ('found' is a word of `found in taxon`)
    known2 = StoreRelations(store + ["aves is the found in taxon of feather"])
    assert covers("In which country can Dren Hoxha be found?", QuestionPlan(relations=[None], tail="country of dren hoxha", n_hop=1), known2)
    assert not covers("What is the country of the found in taxon of Dren Hoxha?",
                      QuestionPlan(relations=[None], tail="country of dren hoxha", n_hop=1), known2)


def test_a_place_class_noun_is_the_ask_only_where_it_is_asked_and_a_relation_everywhere_else():
    """exp_r9 after the place ask (2026-09-13): 'location' had left the vocabulary for every place
    question, so 'What location does the location of X have?' let a 1-hop plan cover a 2-hop question
    and the walk spoke the inner hop -- a wrong answer. Now: 'what is the location of X' is not a
    place ask at all (a class noun followed by 'of' is the relation); an asked span ('in which city',
    'what location') is stripped only when its noun occurs once; otherwise every occurrence must be
    claimed by the plan."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, ask_type, covers
    store = ["laika dog type is the instance of of primitive dogs", "northern russia is the location of laika dog type",
             "bergen mall is the location of playhouse on the mall", "paramus is the location of bergen mall",
             "1950 is the born of jean", "lyon is the birthplace of jean"]
    known = StoreRelations(store)
    assert ask_type("What is the location of the instance of Primitive Dogs?") != "place"
    one = QuestionPlan(relations=[None], tail="instance of of primitive dogs", n_hop=1)
    assert not covers("What is the location of the instance of Primitive Dogs?", one, known)          # 'location' dropped
    two = QuestionPlan(relations=[None, "location"], tail="instance of of primitive dogs", n_hop=2)
    assert covers("What is the location of the instance of Primitive Dogs?", two, known)
    assert ask_type("What location does the location of Playhouse On The Mall have?") == "place"
    one = QuestionPlan(relations=[None], tail="location of playhouse on the mall", n_hop=1)
    assert not covers("What location does the location of Playhouse On The Mall have?", one, known)   # the second 'location' is a hop
    two = QuestionPlan(relations=[None, "location"], tail="location of playhouse on the mall", n_hop=2)
    assert covers("What location does the location of Playhouse On The Mall have?", two, known)
    assert covers("What location does Playhouse On The Mall have?", one, known)                       # the ask, once: stripped
    # the asked class noun, once, is still the frame of a where-question
    assert covers("In which city was Jean born?", QuestionPlan(relations=[None], tail="birthplace of jean", n_hop=1), known,
                  aliases={"birthplace": ["born"]})


def test_a_dropped_hop_hiding_behind_an_inflection_is_still_a_dropped_hop():
    """hdc emitter arm (2026-09-13): 'How is the flag of the country of X depicted?' planned as [country,
    flag] spoke the flag -- 'depicted' is no relation word, but `depicts` is; 'the membership of the
    country' planned without `member of` spoke the country's history. The residual rule now reads a
    leftover word of five letters or more by its stem too; short words ('hold' beside `holder`) are not."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, covers, _stem
    store = ["royaume-uni is the country of robin hood gardens", "second union jack is the flag of royaume-uni",
             "x-shaped cross is the depicts of second union jack", "united nations is the member of of bulgaria",
             "bulgaria is the country of radomir", "the imf is the holder of bulgaria"]
    known = StoreRelations(store)
    assert _stem("depicted") == _stem("depiction") == _stem("depicts") == "depict" and _stem("membership") == _stem("member")
    two = QuestionPlan(relations=[None, "flag"], tail="country of robin hood gardens", n_hop=2)
    assert not covers("How is the flag of the country of Robin Hood Gardens depicted?", two, known)
    assert not covers("What is the depiction of the flag of the country of Robin Hood Gardens?", two, known)
    three = QuestionPlan(relations=[None, "flag", "depicts"], tail="country of robin hood gardens", n_hop=3)
    assert covers("What is the depicts of the flag of the country of Robin Hood Gardens?", three, known)
    one = QuestionPlan(relations=[None], tail="country of radomir", n_hop=1)
    assert not covers("What is the membership of the country of Radomir?", one, known)       # 'membership' ~ member of
    assert covers("Which country does Radomir hold?", one, known)                            # 'hold' is short: not `holder`
    # a whole relation, by stems -- not one word of a longer one: 'associated' is not `genetic association`
    known3 = StoreRelations(store + ["brca1 is the genetic association of cancer", "george is the given name of george branche"])
    assert covers("Which given name is associated with George Branche?", QuestionPlan(relations=[None], tail="given name of george branche", n_hop=1), known3)


def test_a_place_class_noun_that_is_a_held_relation_is_a_hop_the_plan_must_claim():
    """hdc emitter arm (2026-09-13): 'In which country is the location of formation of X?' asks for the
    COUNTRY of that location; with 'country' stripped as the ask, a one-hop plan spoke Minneapolis for the
    United States. The asked class noun is stripped only when the store holds no relation made of it; a
    class noun that is a relation ('country', 'location') must be claimed, and 'in which country was X
    born' is then refused for a plan that stops at the place of birth -- the answer would be a city."""
    from cubbyllm.reasoning.plan_verify import StoreRelations, covers
    store = ["minneapolis is the location of formation of mentor corporation", "united states is the country of minneapolis",
             "ulm is the place of birth of einstein", "germany is the country of ulm", "1879 is the born of einstein"]
    known = StoreRelations(store)
    one = QuestionPlan(relations=[None], tail="location of formation of mentor corporation", n_hop=1)
    two = QuestionPlan(relations=[None, "country"], tail="location of formation of mentor corporation", n_hop=2)
    assert not covers("In which country is the location of formation of Mentor Corporation?", one, known)
    assert covers("In which country is the location of formation of Mentor Corporation?", two, known)
    assert covers("In which country can Mentor Corporation be found?", QuestionPlan(relations=[None], tail="country of mentor corporation", n_hop=1), known)
    birth = QuestionPlan(relations=[None], tail="place of birth of einstein", n_hop=1)
    assert covers("In which city was Einstein born?", birth, known, aliases={"place of birth": ["born"]})     # 'city' is no relation: the ask
    assert not covers("In which country was Einstein born?", birth, known, aliases={"place of birth": ["born"]})  # 'country' is: a hop
    assert covers("In which country was Einstein born?", QuestionPlan(relations=[None, "country"], tail="place of birth of einstein", n_hop=2),
                  known, aliases={"place of birth": ["born"]})
    # a class noun that is only a WORD of a longer relation is the ask, not a hop: exp_r17 (2026-09-13) lost
    # 29 certified "In which city was X born?" plans on the wiki world, where `capital city` is a relation
    known2 = StoreRelations(store + ["berlin is the capital city of germany"])
    assert "city" not in known2 and "city" in known2.words()
    assert covers("In which city was Einstein born?", birth, known2, aliases={"place of birth": ["born"]})
    assert not covers("In which country was Einstein born?", birth, known2, aliases={"place of birth": ["born"]})
