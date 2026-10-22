"""Pins for validation/emitter_data.py: decode_program. The role names ACTION / AGENT / OBJECT are special
tokens in bbpe128k, and `Tokenizer.decode` skips special tokens by default: the 450M emitter's role-binding
programs came back as `bind evt, , "x";` and none ran (H-E18 control, 2026-09-30). decode_program keeps every
special token but the control ones and cuts at </s>. A tiny in-memory tokenizer; no bbpe128k needed.
Run: python -m pytest standin/tests/test_emitter_decode.py -q -p no:hypothesispytest"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from emitter_data import decode_program  # noqa: E402


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    tk = Tokenizer(WordLevel({"<unk>": 0, "bind": 1, "evt,": 2, ",": 3, '"x";': 4}, unk_token="<unk>"))
    tk.pre_tokenizer = WhitespaceSplit()
    tk.add_special_tokens(["<pad>", "<s>", "</s>", "ACTION", "AGENT"])
    return tk


def test_decode_program_keeps_the_role_tokens_the_default_decode_drops():
    tk = _tokenizer()
    ids = tk.encode('bind evt, ACTION , "x";').ids
    assert "ACTION" not in tk.decode(ids)                       # the bug: the default skips special tokens
    assert decode_program(tk, ids) == 'bind evt, ACTION , "x";'


def test_decode_program_cuts_at_eos_and_drops_the_control_tokens():
    tk = _tokenizer()
    eos, pad, bos = tk.token_to_id("</s>"), tk.token_to_id("<pad>"), tk.token_to_id("<s>")
    ids = [bos] + tk.encode('bind evt, AGENT , "x";').ids + [pad, eos] + tk.encode("bind").ids
    assert decode_program(tk, ids, eos) == 'bind evt, AGENT , "x";'


def test_drop_step_values_keeps_the_plan_and_drops_the_mental_sum():
    from emitter_data import drop_step_values, slot_record
    prog = ("create s0 : quantity;   # step 0: 71 - 59 = 12\n        assign s0 = 71;\n        sub s0, 59;\n"
            "create s1 : quantity;   # step 1: 38 - 12 = 26\n")
    assert drop_step_values(prog).count("=") == 1 and "# step 0: 71 - 59\n" in drop_step_values(prog)
    rec = {"id": "a", "task": "arithmetic", "split": "train", "gold": 26, "program": "# q\n" + prog,
           "prompt": "Teresa is 59 and Morio is 71; the baby came when Morio was thirty-eight."}
    s = slot_record(rec, words=True, step_values=False)
    assert "# step 0: $N2 - $N1\n" in s["program"] and "# step 1: $N3 - 12\n" in s["program"]
    assert "assign s0 = $N2;" in s["program"] and "sub s0, $N1;" in s["program"]
    assert {sp["id"]: sp.get("value") for sp in s["spans"]}["$N3"] == "38"
    kept = slot_record(rec, words=False, step_values=True)
    assert "= 26" in kept["program"] and "$N3" not in kept["program"]


def test_slot_record_never_touches_a_register_digit():
    """`div s2, 2` slots the literal 2, never the 2 of `s2` (the 2026-09-30 bug: `div s$N5, 2` in ~9% of the
    arithmetic slot rows, which the fill turned back into `s2` so the VM never noticed)."""
    from emitter_data import slot_record
    rec = {"id": "x", "task": "arithmetic", "subtype": "steps=3", "split": "train", "gold": "5", "vm_ok": True,
           "gold_match": True, "prompt": "A jar has 10 coins split between 2 kids, each pays 2 more. Total?",
           "program": ("program X implements ISolver {\n    public function solve(input: Input): Output {\n"
                       "        create s0 : quantity;   # step 0: 10 / 2 = 5\n        assign s0 = 10;\n        div s0, 2;\n"
                       "        create s1 : quantity;   # step 1: 5 + 2 = 7\n        assign s1 = 0;\n        add s1, s0;\n        add s1, 2;\n"
                       "        create s2 : quantity;   # step 2: 7 - 2 = 5\n        assign s2 = 0;\n        add s2, s1;\n        sub s2, 2;\n"
                       "        sum s2;\n        query s2;\n        return s2;\n    }\n}\n")}
    out = slot_record(rec, words=True, step_values=False)["program"]
    assert "div s0, $N2;" in out and "sub s2, $N2;" in out and "add s1, $N2;" in out
    assert "s$" not in out
