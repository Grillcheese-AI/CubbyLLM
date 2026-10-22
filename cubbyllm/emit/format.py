"""cubbyllm/emit/format.py -- the program format every seat and the brain share: the prompt frame of the 450M's
emitter (``PROMPT_HEAD`` / ``PROMPT_TAIL``) and the decode back to program text (``decode_program``); the ISolver
shim (``shim_isolver``), the function a program answers from (``answer_fn``) and the role-binding wrap
(``wrap_role_prompt``). Order 0031 moved them here, verbatim, from ``validation/emitter_data.py`` and
``standin/data/build_emitter_sft.py``; both re-export them. ``vm_run``: a program run on the VM the way the
harvest runs it (from ``validation/exp_he18_ab.py``, which re-exports it; the step loop's default VM).

``STANDIN_ROLE_WRAP`` (env, read once at import): the role-binding wrap's template; absent = "Record this as an
event: {text}", the one every emitter was trained on.

Wired: WIRED (the 450M seats; the brain's program turns).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

PROMPT_HEAD = "Question:\n"
PROMPT_TAIL = "\nProgram:\n"

CONTROL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>")


def decode_program(tk, ids, eos=None) -> str:
    """Generated ids back to program text: cut at </s>, drop the control tokens, keep every other special token.

    The role names (ACTION, AGENT, OBJECT) are special tokens in bbpe128k, and `Tokenizer.decode` skips
    special tokens by default -- `bind evt, ACTION, "x";` came back as `bind evt, , "x";`, which never runs."""
    ids = list(ids)
    if eos is not None and eos in ids:
        ids = ids[:ids.index(eos)]
    drop = {i for i in (tk.token_to_id(t) for t in CONTROL_TOKENS) if i is not None}
    return tk.decode([i for i in ids if i not in drop], skip_special_tokens=False)


ROLE_WRAP = os.environ.get("STANDIN_ROLE_WRAP", "Record this as an event: {text}")

SHIM = ("    @external\n    public function parse(raw: str): Input { return raw; }\n\n"
        "    public pure function verify(input: Input, output: Output): bool { return true; }\n\n")
_SOLVE_RE = re.compile(r'(\n\s*@external\s*\n\s*public function solve\()')


def shim_isolver(src: str) -> str:
    """Insert parse()+verify() before the @external solve() of a program that
    `implements ISolver` but lacks them. No-op when verify() exists or the
    program is not an ISolver (our ISolve chain programs are untouched)."""
    if "implements ISolver" not in src or re.search(r'function verify\(', src):
        return src
    return _SOLVE_RE.sub("\n" + SHIM + r"\1".lstrip("\n"), src, count=1)


_HOP_RE = re.compile(r'public function hop_(\d+)\(')


def answer_fn(src: str) -> str:
    """The VM function whose result IS the program's answer. Chain programs
    (ISolve dialect) recover hop 1 in solve() and the final hop in hop_N();
    every other family answers from solve(). Found 2026-08-30: scoring every
    program on solve() dropped all 180 multi-hop chains as 'gold mismatch'."""
    hops = [int(h) for h in _HOP_RE.findall(src)]
    return f"hop_{max(hops)}" if hops else "solve"


def wrap_role_prompt(text: str, template: str = ROLE_WRAP) -> str:
    return template.format(text=text.strip())


def vm_run(program: str) -> tuple[bool, object]:
    """(executes, result) from the real VM, with the harvest's ISolver shim and answer function."""
    from ..bridges import cubelang_client as cc
    src = shim_isolver(program)
    try:
        out = cc.run_program_proto(src, fn=answer_fn(src))
        return bool(out.get("ok")), out.get("result")
    except Exception:
        return False, None
