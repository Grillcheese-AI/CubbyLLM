"""Order 0081: the programs seat names only relations the store holds (`cubbyllm/emit/plan_trie.py`).

Pinned without a card:
  * the trie on a stand-in tokenizer that merges the quote at each end: a held label written whole, a prefix two
    labels share, and text outside a plan's literals left free;
  * the same on bbpe128k (its own merges: ``Ġ"$`` for a slot, ``";`` at the close);
  * the seed literal admits only the table's slot ids;
  * ``CB_EMIT_TRIE`` off: no trie is built and the seat's decode call is the one before (no hook passed);
  * the whole seat (`SlotEmitter(Cubby450mEmitter)`) under a scripted decoder: a relation the store does not hold
    comes out as one it holds, and `emitted_plan` reads the program;
  * the ask loop's hand-off, and the ops processor (None leaves the scores untouched).
"""
from __future__ import annotations

import pathlib
import string
import threading

import numpy as np
import pytest
from tokenizers import Tokenizer

from cubbyllm.emit import plan_trie as pt
from cubbyllm.emit.seats_450m import Cubby450mEmitter
from cubbyllm.emit.talk import ContextualEmitter
from cubbyllm.reasoning.emitted import emitted_plan
from cubbyllm.reasoning.planner import normalize
from cubbyllm.reasoning.slots import NameIndex, SlotEmitter

ROOT = pathlib.Path(__file__).resolve().parents[2]
TK = Tokenizer.from_file(str(ROOT / "data" / "grillcheese_bbpe128k.json"))
EOS = TK.token_to_id("</s>")


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class PieceTokenizer:
    """Greedy longest match over a fixed list of pieces, one character otherwise: a stand-in whose pieces merge the
    opening quote with a label's first letters (``, "bi``) and its last letters with the closing quote (``th";``)."""

    def __init__(self, pieces):
        chars = sorted({c for p in pieces for c in p} | set(string.printable))
        self.vocab = list(dict.fromkeys(list(pieces) + chars))
        self.id = {p: i for i, p in enumerate(self.vocab)}
        self.order = sorted(self.vocab, key=len, reverse=True)

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            p = next(p for p in self.order if text.startswith(p, i))
            out.append(self.id[p]); i += len(p)
        return _Enc(out)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


def walk(trie, tk, text_before: str, choose):
    """Feed ``text_before`` (free text), then pick tokens with ``choose(allowed)`` until the trie lets go; the text."""
    ids = tk.encode(text_before).ids
    assert trie.allowed(ids) is not None                 # the role was just written: the literal is constrained
    while (nxt := trie.allowed(ids)) is not None:
        ids.append(choose(nxt))
    return tk.decode(ids, skip_special_tokens=False)


def test_options_are_written_in_context_with_the_quote_merged_at_each_end():
    tk = PieceTokenizer([', "bi', 'rth";', 'birth', 'place', ' date', 'th";', 'e";', 'bind', ' frame', ' HOP'])
    opts = pt.Options(tk, ["birthplace", "birth date", "birth"], "HOP")
    assert opts.n == 3 and not opts.skipped
    first = list(k for k in opts.root if k is not None)
    assert [tk.vocab[t] for t in first] == [', "bi']     # the quote merged with the label's first letters, in context
    node = opts.node([tk.id[', "bi'], tk.id['rth";']])
    assert node is not None and node[None] == "birth"  # the label's last letters merged with the closing quote


def test_the_trie_holds_the_seat_to_held_labels_and_leaves_other_text_free():
    tk = PieceTokenizer([', "bi', 'rth', 'place', ' date', '";', 'bind', ' frame', ' HOP', 'create', ' number'])
    trie = pt.PlanTrie(tk, rel=pt.Options(tk, ["birthplace", "birth date"], "HOP"))
    assert trie.allowed(tk.encode("    create frame: number;\n    bind frame,").ids) is None     # outside: free
    ids = tk.encode("    bind frame, HOP1").ids
    assert [tk.vocab[t] for t in trie.allowed(ids)] == [', "bi']
    ids += [tk.id[', "bi'], tk.id['rth']]
    assert sorted(tk.vocab[t] for t in trie.allowed(ids)) == [" date", "place"]   # a prefix two labels share
    ids += [tk.id["place"]]
    assert [tk.vocab[t] for t in trie.allowed(ids)] == ['";']
    ids += [tk.id['";']]
    assert trie.allowed(ids) is None                     # the label written whole: free again
    assert trie.written == [("HOP", "birthplace")] and trie.entered == {"HOP": 1, "SEED": 0}


def test_bbpe128k_relation_and_seed_literals():
    rel = pt.Options(TK, ["birthplace", "birth date", "country of citizenship"], "HOP")
    seed = pt.Options(TK, ["$E1", "$E12"], "SEED")
    assert rel.n == 3 and seed.n == 2
    trie = pt.PlanTrie(TK, rel, seed)
    first = lambda nxt: sorted(nxt)[0]                   # noqa: E731 -- any choice stays inside the options
    text = walk(trie, TK, "        create frame: number;\n        bind frame, SEED", first)
    assert text.endswith('bind frame, SEED, "$E1";') or text.endswith('bind frame, SEED, "$E12";')
    assert any(TK.id_to_token(t) == 'Ġ"$' for t in seed.root[TK.token_to_id(",")])    # bbpe128k merges `"$`
    trie.reset()
    text = walk(trie, TK, '        bind frame, SEED, "$E1";\n        bind frame, HOP1', first)
    lit = text.rsplit('HOP1, "', 1)[1]
    assert lit.endswith('";') and lit[:-2] in {"birthplace", "birth date", "country of citizenship"}


def test_the_seed_literal_admits_only_the_tables_ids(monkeypatch):
    monkeypatch.setenv("CB_EMIT_TRIE", "rel+slot")
    seat = Cubby450mEmitter(FakeTrunk(""))
    trie = pt.for_seat(seat, None, ["$E1", "$E2"])
    ids = TK.encode("    bind frame, SEED").ids
    got = set()

    def explore(ids):
        nxt = trie.allowed(ids)
        if nxt is None:
            got.add(TK.decode(ids).rsplit("SEED", 1)[1]); return
        for t in nxt:
            trie.reset(); explore(ids + [t])
    explore(ids)
    assert got == {', "$E1";', ', "$E2";'}
    monkeypatch.setenv("CB_EMIT_TRIE", "rel")
    assert pt.for_seat(seat, None, ["$E1"]) is None      # rel only: the seed is left free (and nothing else is asked)


def test_the_flag_reads_off_by_default_and_refuses_an_unknown_value(monkeypatch):
    monkeypatch.delenv("CB_EMIT_TRIE", raising=False)
    assert pt.mode() == "off"
    monkeypatch.setenv("CB_EMIT_TRIE", "on")
    with pytest.raises(ValueError):
        pt.mode()


class FakeTrunk:
    """`Cubby450mTrunk`'s surface, no card. Decodes ``script`` greedily: where ``allowed`` returns ids, the one that
    keeps most of what the script wanted to write, else the first."""

    def __init__(self, script: str):
        self.adapter_dirs = {"programs": "/adapters/emitter_x"}
        self.max_ctx, self.script, self.calls = 2048, script, []
        self._lock = threading.Lock()

    tokenizer, eos = TK, EOS

    def generate(self, role, rows, max_new_tokens, temperature=0.0, top_k=40, seed=None, **hook):
        self.calls.append(sorted(hook))
        want = TK.encode(self.script + "</s>", add_special_tokens=False).ids
        allowed = hook.get("allowed")
        if allowed is None:
            return [want]
        out, target = [], self.script
        while len(out) < max_new_tokens:
            done, nxt = TK.decode(out, skip_special_tokens=False), allowed(out)
            if nxt is not None:                          # inside a literal: the script's token if allowed, else the longest
                rest = target[len(done):] if target.startswith(done) else ""
                t = max(nxt, key=lambda t: (rest.startswith(TK.decode([t])), len(TK.decode([t]))))
            else:
                if not target.startswith(done):          # the trie changed a literal: the script's next lines after it
                    target = done + "\n" + "\n".join(target.split("\n")[done.count("\n") + 1:])
                    want = out + TK.encode(target[len(done):] + "</s>", add_special_tokens=False).ids
                t = want[len(out)]
            out.append(t)
            if t == EOS:
                break
        return [out]


PROGRAM = ('use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n'
           '        create frame: number;\n        bind frame, SEED, "$E1";\n        bind frame, HOP1, "birth";\n'
           '        return recover(frame, SEED);\n    }\n}\n')


def seat(script=PROGRAM):
    s = SlotEmitter(Cubby450mEmitter(FakeTrunk(script)), index=NameIndex({"Marie Curie"}))
    s.relations = ["birthplace", "birth date", "country of citizenship"]
    return s


def test_off_decodes_exactly_as_before(monkeypatch):
    monkeypatch.delenv("CB_EMIT_TRIE", raising=False)
    s = seat()
    out = s.emit("Where was Marie Curie born?")
    assert s.inner.trunk.calls == [[]] and "trie" not in s.last       # no hook handed to the decoder
    assert emitted_plan(out, normalize) == (["birth"], "marie curie")


def test_rel_turns_a_relation_the_store_does_not_hold_into_one_it_holds(monkeypatch):
    monkeypatch.setenv("CB_EMIT_TRIE", "rel+slot")
    s = seat()
    out = s.emit("Where was Marie Curie born?")
    assert s.inner.trunk.calls == [["allowed"]]
    rels, seed = emitted_plan(out, normalize)
    assert rels[0] in s.relations and seed == "marie curie"
    assert s.last["trie"]["entered"] == {"HOP": 1, "SEED": 1} and s.last["trie"]["left"] == 0


def test_the_ask_loop_hands_its_relations_to_the_slotted_seat():
    s = SlotEmitter(Cubby450mEmitter(FakeTrunk("")))
    known = ["birthplace"]
    assert pt.hand(ContextualEmitter({"programs": s, "talk": object.__new__(FakeTrunk)}), known) == 1
    assert s.relations is known


class _T(np.ndarray):
    def scatter_(self, dim, index, value):
        np.put_along_axis(self, index, value, axis=dim)


class _G:
    """The few grilly calls `allowed_processor` makes, on numpy."""
    int64 = np.int64

    @staticmethod
    def full(shape, v):
        return np.full(shape, v, dtype=np.float32).view(_T)

    @staticmethod
    def tensor(x, dtype=None):
        return np.asarray(x)


def test_the_processor_masks_only_inside_a_literal():
    from cubbyllm.ops.grilly_seats import allowed_processor
    scores = np.zeros((1, 6), dtype=np.float32)
    rows = np.array([[9, 9, 1, 2]])
    seen = []
    proc = allowed_processor(lambda ids: seen.append(ids) or None, 2, g=_G)
    assert proc(rows, scores) is scores and seen == [[1, 2]]
    proc = allowed_processor(lambda ids: [3, 4], 2, g=_G)
    got = proc(rows, scores)
    assert np.isfinite(got).tolist() == [[False, False, False, True, True, False]]
