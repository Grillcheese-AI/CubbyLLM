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


# ── chunking: how many hops share a frame (WO-2.6 / exp_r28) ────────────────
LONG = [Triple(obj=chr(66 + i), rel=f"r{i + 1}", subj=chr(65 + i)) for i in range(5)]
LONG_RELS = [t.rel for t in LONG]


def test_chunk_zero_is_the_shipped_shape_byte_for_byte():
    """The serving path passes chunk=0. If this ever differs, every number the
    repo has measured against the old shape is off by an unknown amount."""
    assert build_chain_program(LONG, LONG_RELS) == \
        build_chain_program(LONG, LONG_RELS, chunk=0)
    src, _ = build_chain_program(LONG, LONG_RELS, chunk=0)
    assert src.count("bind frame") == 6 * 5      # 5 hops + control, each binding all 5


def test_chunk_groups_the_bindings_and_keeps_the_fn_contract():
    src, fns = build_chain_program(LONG, LONG_RELS, chunk=2)
    # the pipeline relies on this: hops in order, control LAST
    assert fns == ["solve", "hop_2", "hop_3", "hop_4", "hop_5", "control"]
    # groups [0,1] [2,3] [4]; control binds group 0 -> 2+2+2+2+1+2
    assert src.count("bind frame") == 11

    def binds(fn):
        return src.split(f"function {fn}(")[1].split("}")[0].count("bind frame")

    assert binds("solve") == 2 and binds("hop_2") == 2      # same group, same frame
    assert binds("hop_5") == 1                              # the short last group
    assert binds("control") == 2                            # the noise floor at group size
    assert "bind frame, ABSENT_CTRL" not in src             # still never bound


def test_chunk_one_is_one_binding_per_frame():
    src, _ = build_chain_program(LONG, LONG_RELS, chunk=1)
    assert src.count("bind frame") == 6                     # 5 hops + control, one each


def test_chunk_is_a_no_op_on_a_single_hop():
    one, rels = [Triple(obj="B", rel="r", subj="A")], ["r"]
    base = build_chain_program(one, rels)
    for c in (0, 1, 2, 9):
        assert build_chain_program(one, rels, chunk=c) == base
