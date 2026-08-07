from cubbyllm.reasoning.planner import Triple
from cubbyllm.reasoning.programs import build_chain_program, sanitize_role

T = [Triple(obj="united stated", rel="country of citizenship", subj="cynthia basinet"),
     Triple(obj="united stated", rel="country", subj="united stated"),
     Triple(obj="oceania portal", rel="continent", subj="united stated")]
RELS = ["country of citizenship", "country", "continent"]


def test_sanitize_role():
    assert sanitize_role(1, "country of citizenship") == "H1_COUNTRY_OF_CITIZENSHIP"
    assert sanitize_role(2, "spouse (2nd)") == "H2_SPOUSE_2ND"


def test_program_shape_and_fn_order():
    src, fns = build_chain_program(T, RELS)
    assert fns == ["solve", "hop_2", "hop_3", "control"]
    assert src.count('bind frame, H1_COUNTRY_OF_CITIZENSHIP, "united stated";') == 4
    assert 'return recover(frame, H3_CONTINENT);' in src
    assert 'return recover(frame, ABSENT_CTRL);' in src
    assert 'bind frame, ABSENT_CTRL' not in src           # control never bound
    assert src.startswith("use vsa;")
    assert "program CotChain implements ISolve" in src
    assert "public function solve(mention: str): str" in src


def test_filler_escaping():
    src, _ = build_chain_program(
        [Triple(obj='he said "hi"', rel="quote", subj="x")], ["quote"])
    assert 'bind frame, H1_QUOTE, "he said \\"hi\\"";' in src


def test_duplicate_relations_get_distinct_roles():
    src, _ = build_chain_program(
        [Triple(obj="a", rel="country", subj="s"),
         Triple(obj="b", rel="country", subj="a")], ["country", "country"])
    assert "H1_COUNTRY" in src and "H2_COUNTRY" in src
