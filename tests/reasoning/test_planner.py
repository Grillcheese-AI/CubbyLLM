"""Grammar of the .pq corpus: templated questions and (two) fact templates."""
from cubbyllm.reasoning.planner import (
    QuestionPlan, Triple, normalize, parse_fact, parse_question,
    relation_matches)


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
