"""Grammar of the .pq corpus: templated questions and (two) fact templates."""
import time

from cubbyllm.reasoning.planner import (
    MAX_QUESTION_LEN, QuestionPlan, Triple, normalize, parse_fact,
    parse_question, relation_matches)


def test_parse_three_hop_question():
    q = ("What is the continent of the country of the country of "
         "citizenship of cynthia basinet?")
    p = parse_question(q)
    assert p is not None and p.n_hop == 3
    assert p.tail == "country of citizenship of cynthia basinet"
    assert p.relations == [None, "country", "continent"]  # walk order


def test_parse_one_hop_question():
    p = parse_question("What is the capital of france?")
    assert p is not None and p.n_hop == 1
    assert p.relations == [None]
    assert p.tail == "capital of france"


def test_unparseable_question_returns_none():
    assert parse_question("Tell me about cheese.") is None


def test_parse_fact_standard_template():
    t = parse_fact("united stated is the country of citizenship of cynthia basinet")
    assert t == Triple(obj="united stated",
                       rel="country of citizenship", subj="cynthia basinet")


def test_parse_fact_is_in_template():
    # the corpus's second template: "X is the R Y is in"
    t = parse_fact("united stated is the country united stated is in")
    assert t == Triple(obj="united stated", rel="country", subj="united stated")


def test_parse_fact_rejects_freeform():
    assert parse_fact("cheese tastes great on toast") is None


def test_relation_matches_exact_and_fuzzy():
    assert relation_matches("country of citizenship", "country of citizenship")
    assert relation_matches("country", "country")
    assert not relation_matches("continent", "country")


def test_normalize_strips_articles_case_punct():
    assert normalize("  The Oceania Portal! ") == "oceania portal"


# --- grammar v2: WH-frame coverage (the 337/800 ceiling) ---------------


def test_parse_which_class_three_hop_chain():
    # "Which <class> is the R of the R of ... <tail>?" — leading class noun
    # duplicates the final relation's answer type; captured, not a hop.
    q = ("Which country is the country of the currency of the country of "
         "citizenship of Henry Allcock?")
    p = parse_question(q)
    assert p is not None and p.n_hop == 3
    assert p.answer_class == "country"
    assert p.tail == "country of citizenship of Henry Allcock"
    assert p.relations == [None, "currency", "country"]


def test_parse_which_class_direct_in_form():
    # "Which <class> is <entity> in?" — no "of the" chain at all.
    p = parse_question("Which country is arcadia, maryland in?")
    assert p is not None and p.n_hop == 1
    assert p.answer_class == "country"
    assert p.relations == [None]
    assert p.tail == "country of arcadia, maryland"


def test_parse_where_one_hop():
    p = parse_question("Where is the administrative territorial entity of inster?")
    assert p is not None and p.n_hop == 1
    assert p.relations == [None]
    assert p.tail == "administrative territorial entity of inster"


def test_parse_where_two_hop():
    q = ("Where is the legislative body of the administrative territorial "
         "entity of Grand Saconnex?")
    p = parse_question(q)
    assert p is not None and p.n_hop == 2
    assert p.relations == [None, "legislative body"]
    assert p.tail == "administrative territorial entity of Grand Saconnex"


def test_parse_who_one_hop():
    p = parse_question("Who is the parent taxon of Biwia tama?")
    assert p is not None and p.n_hop == 1
    assert p.relations == [None]
    assert p.tail == "parent taxon of Biwia tama"


def test_parse_does_have_inversion():
    # "What R does <entity> have?" — synthesized into canonical "R of E".
    p = parse_question("What given name does margit koloczy have?")
    assert p is not None
    assert p.relations == [None]
    assert p.tail == "given name of margit koloczy"


def test_parse_does_have_inversion_predecessor():
    p = parse_question("What Predecessor does attitude adjuster (album) have?")
    assert p is not None
    assert p.relations == [None]
    assert p.tail == "Predecessor of attitude adjuster (album)"


def test_parse_which_is_variant():
    # "Which is the R of the R of ... <tail>?" — no class noun this time.
    p = parse_question("Which is the flag of the country of xia county?")
    assert p is not None and p.n_hop == 2
    assert p.answer_class is None
    assert p.relations == [None, "flag"]
    assert p.tail == "country of xia county"


def test_parse_nested_relative_clause_stays_unparseable():
    # By-design exclusion: recursion (relative-clause nesting) stays out
    # of the grammar tier — future LM-tier job, not this parser's.
    q = ("Which list includes the list that includes the component of "
         "the instance of Speranza coortaria?")
    assert parse_question(q) is None


# --- backtracking guard (reviewer finding on grammar v2) ---------------


def test_parse_adversarial_long_input_returns_none_fast():
    # Many near-miss ' is ' occurrences and no closing '?' used to trigger
    # quadratic backtracking in the v2 frames (~10.7s measured at 80k
    # chars). MAX_QUESTION_LEN + the '?' pre-check reject this in O(len)
    # before any regex runs, so it must return instantly.
    q = "which " + ("is " * 33334)  # ~100k chars, no '?'
    assert len(q) > 90_000
    start = time.perf_counter()
    result = parse_question(q)
    elapsed = time.perf_counter() - start
    assert result is None
    assert elapsed < 1.0


def test_parse_boundary_length_question_still_parses():
    # A legitimate question at exactly MAX_QUESTION_LEN must still parse —
    # the length guard rejects strictly-longer input, not the boundary.
    prefix, suffix = "What is the capital of ", "?"
    tail_entity = "x" * (MAX_QUESTION_LEN - len(prefix) - len(suffix))
    q = f"{prefix}{tail_entity}{suffix}"
    assert len(q) == MAX_QUESTION_LEN
    p = parse_question(q)
    assert p is not None
    assert p.relations == [None]
    assert p.tail == f"capital of {tail_entity}"


def test_causal_phrasings_rewrite_onto_the_cause_and_impact_relations():
    """GoT challenge pre-check (k), 2026-09-03: the gap was lexical."""
    from cubbyllm.reasoning.planner import normalize_causal, parse_question
    assert normalize_causal("What led to the French Revolution?") == "what is the cause of French Revolution?", "leading article dropped: ' of the ' is a chain boundary"
    assert normalize_causal("What caused the fall of Rome?") == "what is the cause of fall of Rome?"
    assert normalize_causal("What happened because of the printing press?") == "what is the impact of printing press?"
    assert normalize_causal("What were the consequences of the Black Death?") == "what is the impact of Black Death?"
    assert normalize_causal("what is the capital of France?") == "what is the capital of France?", "non-causal text untouched"
    for q in ("What led to the French Revolution?", "What caused the fall of Rome?", "What happened because of the printing press?",
              "What is the cause of Bronze Age collapse?", "What is the impact of Black Death?"):
        plan = parse_question(q)
        assert plan is not None and plan.n_hop == 1 and plan.tail.split(" of ")[0] in ("cause", "impact"), q
    assert parse_question("Did the Great Depression cause the New Deal?") is None, "a yes/no question is not a chain"
