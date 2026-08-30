"""Unit pins for the pure helpers in standin/data/build_emitter_sft.py — no
VM, no corpus. Run: python -m pytest standin/tests -q
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_emitter_sft as b  # noqa: E402

ISOLVER_NO_SHIM = """# q
program GSM0 implements ISolver {
    type Input = str;
    type Output = quantity;

    @external
    public function solve(input: Input): Output {
        create s0 : quantity;
        assign s0 = 1;
        return s0;
    }
}
"""
ISOLVE_CHAIN = """use vsa;

program CotChain implements ISolve {
    public function solve(mention: str): str {
        create frame: number;
        bind frame, H1_CAPITAL, "paris";
        return recover(frame, H1_CAPITAL);
    }
}
"""


def test_shim_inserts_parse_and_verify_before_solve_once():
    out = b.shim_isolver(ISOLVER_NO_SHIM)
    assert out.count("function parse(raw: str): Input") == 1
    assert out.count("pure function verify(input: Input, output: Output): bool") == 1
    assert out.index("function parse(") < out.index("function verify(") < out.index("function solve(")
    assert b.shim_isolver(out) == out                       # idempotent


def test_shim_leaves_isolve_chain_programs_and_kernels_alone():
    assert b.shim_isolver(ISOLVE_CHAIN) == ISOLVE_CHAIN
    kernel = ISOLVER_NO_SHIM.replace("    @external\n    public function solve(",
                                     "    public pure function verify(input: Input, output: Output): bool { return true; }\n\n"
                                     "    @external\n    public function solve(")
    assert b.shim_isolver(kernel) == kernel


def test_answer_fn_is_last_hop_for_chains_and_solve_otherwise():
    three_hop = ("program CotChain implements ISolve {\n    public function solve(mention: str): str { }\n"
                 "    public function hop_2(): str { }\n    public function hop_3(): str { }\n"
                 "    public function control(): str { }\n}\n")
    assert b.answer_fn(three_hop) == "hop_3"
    assert b.answer_fn(ISOLVE_CHAIN) == "solve"          # 1-hop chain answers from solve()
    assert b.answer_fn(ISOLVER_NO_SHIM) == "solve"       # arithmetic / role-binding / kernels


def test_gsm_question_strips_socratic_wrapper():
    assert b.gsm_question("Question: Janet has 3 ducks. How many?\nAnswer: 3 ducks.\n#### 3") == "Janet has 3 ducks. How many?"


def test_split_is_deterministic_and_roughly_val_frac():
    prompts = [f"prompt number {i}" for i in range(4000)]
    a = [b.split_of(p, 0.05) for p in prompts]
    assert a == [b.split_of(p, 0.05) for p in prompts]
    frac = a.count("val") / len(a)
    assert 0.03 < frac < 0.07


def test_parse_aug_txt_yields_instruction_program_pairs(tmp_path):
    txt = tmp_path / "aug.txt"
    txt.write_text("[INSTRUCTION]\nDo a thing.\n[/INSTRUCTION]\n# Do a thing.\nprogram Ev implements ISolver {\n}\n<|endofdoc|>\n"
                   "[INSTRUCTION]\nTwo\nlines.\n[/INSTRUCTION]\nprogram GSM3 implements ISolver {\n}\n<|endofdoc|>\n",
                   encoding="utf-8")
    pairs = list(b.parse_aug_txt(str(txt)))
    assert [p[0] for p in pairs] == ["Do a thing.", "Two\nlines."]
    assert pairs[0][1].startswith("# Do a thing.\nprogram Ev") and pairs[1][1].startswith("program GSM3")


def test_cubbyllm_never_imports_standin():
    """Guardrail 2 (standin/README.md): the stand-in sits BEHIND the trunk
    interface; the package must never depend on it."""
    import re
    offenders = []
    for py in (ROOT / "cubbyllm").rglob("*.py"):
        if re.search(r'^\s*(from|import)\s+standin\b', py.read_text(encoding="utf-8", errors="replace"), re.M):
            offenders.append(str(py.relative_to(ROOT)))
    assert offenders == [], offenders


def test_emitter_protocol_and_replay():
    sys.path.insert(0, str(ROOT))
    from standin.emitter import Emitter, LlamaCppEmitter, LlamaServerEmitter, ReplayEmitter
    rep = ReplayEmitter([{"prompt": "q1", "generated": "program X"}])
    assert isinstance(rep, Emitter) and isinstance(LlamaServerEmitter(), Emitter)
    lc = LlamaCppEmitter("C:/x/emitter-q4_k_m.gguf")          # lazy: no load until emit()
    assert isinstance(lc, Emitter) and lc.name == "llama-cpp:emitter-q4_k_m.gguf" and lc._llm is None
    assert rep.emit("q1") == "program X"
    try:
        rep.emit("unknown"); assert False, "must raise on an unrecorded prompt"
    except KeyError:
        pass


def test_strip_fences_and_think_blocks():
    sys.path.insert(0, str(ROOT / "standin"))
    from eval_emitter_vm import strip_fences, strip_think
    assert strip_fences("```cubelang\nprogram A {}\n```") == "program A {}\n"
    assert strip_fences("program A {}") == "program A {}\n"
    # LFM2.5 reasoning prefix: with or without the opening tag, multi-line, dropped once
    assert strip_think("<think>let me reason\nabout it</think>\nprogram A {}") == "program A {}"
    assert strip_think("The user wants X.\nLet me...\n</think># q\nprogram A {}") == "# q\nprogram A {}"
    assert strip_think("No — I'm Cubby.") == "No — I'm Cubby."                       # nothing to strip
    assert strip_fences("thinking...</think>```\nprogram A {}\n```") == "program A {}\n"


def test_role_binding_fillers_dropped_and_prompts_wrapped():
    assert b.is_chat_filler("Let me know what you'd like to focus on next!")
    assert b.is_chat_filler("Thanks!") and b.is_chat_filler("Sure, go ahead.") and b.is_chat_filler("Hi there")
    assert b.is_chat_filler("Do it.")                                             # < 4 words
    assert not b.is_chat_filler("Give three tips for staying healthy.")
    assert not b.is_chat_filler("Rearrange the words to create a complete sentence.")
    assert b.wrap_role_prompt("Give three tips for staying healthy.") == "Record this as an event: Give three tips for staying healthy."
    assert b.ROLE_SOURCES == ["svc"] and b.CAP_ROLE == 1500 and b.CHAIN_MULT == 3


def test_render_chatml_matches_the_lfm25_template():
    sys.path.insert(0, str(ROOT))
    from standin.emitter import NO_THINK_PREFILL, render_chatml
    t = render_chatml("SYS", "hello")
    assert t == "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    assert render_chatml("", "hello", "") == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    assert NO_THINK_PREFILL == "<think>\n</think>\n"


def test_gold_matches_numeric_and_string_and_missing():
    assert b.gold_matches("72", 72) is True
    assert b.gold_matches("72.0", 72) is True
    assert b.gold_matches("71", 72) is False
    assert b.gold_matches("Paris ", "paris") is False              # exact string compare after strip
    assert b.gold_matches("paris", "paris") is True
    assert b.gold_matches("anything", None) is None
