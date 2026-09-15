"""Unit pins for the pure helpers in standin/data/build_emitter_sft.py — no
VM, no corpus. Run: python -m pytest standin/tests -q
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_emitter_sft as b  # noqa: E402

_FAKE_GGUF_DIR = pathlib.Path(tempfile.mkdtemp(prefix="cb-fake-gguf-"))


def fake_gguf(name: str) -> str:
    """A path that EXISTS but holds no model. LlamaCppEmitter validates its path in
    __init__ so a launch mistake fails at LAUNCH, not minutes later inside _speak; the
    model LOAD is still lazy, which is what the assertions below pin."""
    f = _FAKE_GGUF_DIR / name
    f.touch()
    return str(f)

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
    lc = LlamaCppEmitter(fake_gguf("emitter-q4_k_m.gguf"))          # lazy: no load until emit()
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


def test_chat_family_is_sniffed_from_the_gguf_template_and_rendered_per_family():
    """A talk adapter on a non-LFM base (the bake-off) gets its own template: Gemma from <start_of_turn>,
    Qwen3/LFM stay ChatML; an unknown or missing template falls back to ChatML; prefill follows the family."""
    from standin.emitter import CHAT_FAMILIES, LlamaCppEmitter, chat_family, render_gemma
    gemma_tpl = "{% for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n{% endfor %}"
    chatml_tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    assert chat_family(gemma_tpl, "gemma4") == "gemma" and chat_family(gemma_tpl) == "gemma"
    assert chat_family(chatml_tpl, "lfm2") == "lfm" and chat_family(chatml_tpl, "lfm2moe") == "lfm"   # the 2.6B and the 8B-A1B
    assert chat_family(chatml_tpl, "qwen3") == "chatml"                                              # ChatML without a think block
    assert chat_family(None) == "lfm" and chat_family("") == "lfm"                                    # no metadata: every GGUF served so far
    assert render_gemma("SYS", "hello") == "<start_of_turn>user\nSYS\n\nhello<end_of_turn>\n<start_of_turn>model\n"
    assert render_gemma("", "hello", "Answer:") == "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\nAnswer:"
    assert CHAT_FAMILIES["lfm"][1] == CHAT_FAMILIES["chatml"][1] == ["<|im_end|>"] and CHAT_FAMILIES["gemma"][1] == ["<end_of_turn>"]
    g = LlamaCppEmitter(fake_gguf("talk_gemma.gguf"), family="gemma")       # explicit family: no load needed
    assert g.family == "gemma" and g.prefill == ""
    assert LlamaCppEmitter(fake_gguf("emitter_v7.Q4_K_M.gguf"), family="lfm").prefill == "<think>\n</think>\n"
    assert LlamaCppEmitter(fake_gguf("talk_qwen.gguf"), family="chatml").prefill == ""
    assert LlamaCppEmitter(fake_gguf("a.gguf"), family="gemma", prefill="<think>\n</think>\n").prefill == "<think>\n</think>\n"


def test_gold_matches_numeric_and_string_and_missing():
    assert b.gold_matches("72", 72) is True
    assert b.gold_matches("72.0", 72) is True
    assert b.gold_matches("71", 72) is False
    assert b.gold_matches("Paris ", "paris") is False              # exact string compare after strip
    assert b.gold_matches("paris", "paris") is True
    assert b.gold_matches("anything", None) is None



def test_llama_cpp_emitters_take_turns_on_the_gpu_across_threads():
    """serve_api is a ThreadingHTTPServer: a chat turn and a /pac step reach the emitters from two threads.
    Llama is not thread-safe, so every LlamaCppEmitter decodes under one process-wide lock — two adapters
    included (the two-model reset). Pinned with a fake Llama that fails on re-entry."""
    import threading, time
    from standin.emitter import LlamaCppEmitter

    class FakeLlama:
        busy = 0
        overlaps = 0
        metadata = {"tokenizer.chat_template": "<|im_start|>"}

        def tokenize(self, text, add_bos=True, special=False):
            return [9] if text == b"<|im_end|>" else [1, 2]

        def token_eos(self):
            return 9

        def detokenize(self, toks, prev_tokens=None, special=False):
            return b"ok" if toks else b""

        def generate(self, ids, **kw):              # the whole decode holds the lock: re-entry from another thread is the reset
            FakeLlama.busy += 1
            if FakeLlama.busy > 1:
                FakeLlama.overlaps += 1
            time.sleep(0.01)
            yield 3
            FakeLlama.busy -= 1
            yield 9

    a, b = LlamaCppEmitter(fake_gguf("a.gguf"), family="chatml"), LlamaCppEmitter(fake_gguf("b.gguf"), family="chatml")
    a._llm, b._llm = FakeLlama(), FakeLlama()
    outs = []
    ts = [threading.Thread(target=lambda e=e: outs.append(e.emit("hi", max_new_tokens=4))) for e in (a, b, a, b, a, b)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    assert outs == ["ok"] * 6 and FakeLlama.overlaps == 0



def test_clean_reply_strips_qwen_fragments_and_keeps_complete_tool_calls():
    """Live Qwen session (2026-09-04): '<tool_response>\n\n</tool_call>\n\nHi…' and '<think>\n\n<tool_call>\n\nHi!…'
    reached the user; a closed think block was the only thing stripped."""
    from standin.emitter import clean_reply
    assert clean_reply("<tool_response>\n\n</tool_call>\n\nHi, I'm Cubby. Ask me anything.") == "Hi, I'm Cubby. Ask me anything."
    assert clean_reply("<think>\n\n<tool_call>\n\nHi! I'm doing great.") == "Hi! I'm doing great."
    assert clean_reply("<think>\nreasoning\n</think>\n\nCanberra.") == "Canberra."
    call = '<tool_call>\n{"name": "news_search", "arguments": {"query": "quebec"}}\n</tool_call>'
    assert clean_reply("<think>\n\n</think>\n\n" + call) == call, "a complete tool call is kept for the host"
    assert clean_reply("Sure. " + call + "\n</tool_call>") == "Sure. " + call



class _FakeLlama:
    """n_vocab/tokenize/detokenize(special=)/generate/token_eos/set_seed/metadata — enough to load and decode
    through a LlamaCppEmitter without a model. Piece 5 is a CONTROL token: rendered only with special=True."""
    PIECES = [b"Hi", b" there", "建筑师".encode(), b" 8", "Ещё".encode(), b"<|tool_call_start|>", b"\xe5"]   # the last: a lone byte
    SPECIAL = {5, 7}
    EOS = 7
    SCRIPT = [0, 1, EOS]          # what generate() yields

    def __init__(self, **kw):
        self.kw = kw
        self.metadata = {"general.architecture": "qwen3", "tokenizer.chat_template": "{% for m in messages %}<|im_start|>..."}
        self.calls = []

    def n_vocab(self):
        return len(self.PIECES)

    def token_eos(self):
        return self.EOS

    def set_seed(self, seed):
        self.seed = seed

    def tokenize(self, text, add_bos=True, special=False):
        if text == b"<|im_end|>":
            return [self.EOS]
        return [0] * (len(text) // 4 + 1)

    def detokenize(self, toks, prev_tokens=None, special=False):
        return b"".join(self.PIECES[t] if (t not in self.SPECIAL or special) else b"" for t in toks if t < len(self.PIECES))

    def generate(self, ids, **kw):
        self.calls.append(kw)
        yield from self.SCRIPT


def test_script_ban_ids_come_from_the_vocab_and_zero_the_logits(tmp_path, monkeypatch):
    import numpy as np
    from standin import emitter as em
    ban = em.ScriptBan.from_vocab(_FakeLlama())
    assert list(ban.ids) == [2, 4], "the CJK and Cyrillic pieces, not the byte-fallback token"
    scores = np.zeros(7, dtype=np.float32)
    out = ban(np.array([0]), scores)
    assert np.isinf(out[2]) and np.isinf(out[4]) and out[0] == 0 and out[6] == 0
    # the identity-side guard sees the same script class
    from identity import _NON_LATIN
    assert em.NON_LATIN.pattern == _NON_LATIN.pattern


def test_llama_emitter_passes_the_script_ban_to_every_decode(tmp_path, monkeypatch):
    import llama_cpp
    import numpy as np
    from standin import emitter as em
    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)
    gguf = tmp_path / "talk.gguf"
    gguf.write_bytes(b"GGUF")
    e = em.LlamaCppEmitter(str(gguf))
    assert e.emit("hello") == "Hi there"
    lp = e._llm.calls[0]["logits_processor"]
    assert lp is not None and list(lp[0].ids) == [2, 4]
    assert e._llm.calls[0]["temp"] == 0.0, "temperature 0 = greedy, as create_completion did"
    assert (tmp_path / "talk.gguf.scriptban.npy").exists(), "cached beside the GGUF"
    assert list(np.load(tmp_path / "talk.gguf.scriptban.npy")) == [2, 4]
    off = em.LlamaCppEmitter(str(gguf), script_ban=False)
    off.emit("hello")
    assert off._llm.calls[0]["logits_processor"] is None



def test_llama_emitter_renders_control_tokens_and_stops_at_eos(tmp_path, monkeypatch):
    """The v10 tool probe (2026-09-04): create_completion dropped LFM's <|tool_call_start|> (a control token) and
    the host saw a bare call body. The token-level decode renders it; EOS and the family's stop token end it."""
    import llama_cpp
    from standin import emitter as em

    class _Tool(_FakeLlama):
        SCRIPT = [5, 0, 1, 5, _FakeLlama.EOS, 3, 3]      # the EOS ends the reply; the trailing pieces are never read

    monkeypatch.setattr(llama_cpp, "Llama", _Tool)
    gguf = tmp_path / "talk.gguf"
    gguf.write_bytes(b"GGUF")
    e = em.LlamaCppEmitter(str(gguf), script_ban=False)
    assert e.emit("news?") == "<|tool_call_start|>Hi there<|tool_call_start|>"
    assert e.emit("news?", max_new_tokens=2) == "<|tool_call_start|>Hi", "max_new_tokens bounds the decode"
    e2 = em.LlamaCppEmitter(str(gguf), script_ban=False)
    assert e2.emit("x", temperature=0.7, seed=3) and e2._llm.seed == 3 and e2._llm.calls[0]["temp"] == 0.7
