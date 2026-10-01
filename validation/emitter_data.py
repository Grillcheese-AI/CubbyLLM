"""The 450M emitter adapter's data (H-E15, 2026-09-29): the emitter SFT set in SLOT form.

Shared by the torch trainer (Colab, `train_emitter_torch.py`) and the local grilly2 emitter
(`standin/emitter.py: Cubby450mEmitter`), so both see one prompt format and one slotting rule.

The emitter never copies a name or a number the host could have placed. Per family:

  chain         hop objects (`bind frame, H1_CAPITAL, "Berlin"`) -> `$E` slots; the index is what the
                host knows at serve time: every name in the fact lines (`X is the R of Y`) and the seed
  plan          the seed -> `$E1`; the HOP words stay words (a closed vocabulary the emitter learns)
  arithmetic /  every number the prompt states -> `$N` by position (`assign s0 = $N1;`), in the step
  kernel        comments too; a derived constant (`div s0, 2` for "half") stays a literal
  role_binding  NO slots: the event recorder's literals are free spans of the sentence, which no index
                holds at serve time. It is the copy control: the family a 450M with copy loss 1.2 is
                expected to lose against the 2.6B stand-in, and the measurement of how much.

The leading `# <question>` comment (a verbatim copy of the prompt) is dropped from every program: the
VM ignores it and a 450M would spend its whole budget copying it. A chain/plan record whose bound
literal is NOT in its prompt is dropped (the coverage gate, same rule as the SFT builder's).

Records: `{id, task, subtype, split, prompt (annotated), program (slotted), gold, spans}`; the loss mask
covers the program and its </s>. `PROMPT_HEAD` / `PROMPT_TAIL` frame the prompt for a base LM with no
chat template; the serve path uses the same frame.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cubbyllm.reasoning.slots import (  # noqa: E402
    BIND_LIT_RX, NUM_LIT_RX, SLOT_RX, SlotTable, Span, SpanIndex, _norm, canon, extract, is_entity_role, num_value)

PROMPT_HEAD = "Question:\n"
PROMPT_TAIL = "\nProgram:\n"
NUMBER_FAMILIES = ("arithmetic", "kernel")
ENTITY_FAMILIES = ("chain", "plan")
_FACT_RX = re.compile(r"^-\s*(?P<obj>.+?) is the (?P<rel>.+?) of (?P<subj>.+?)\s*$")
_LEAD_COMMENT_RX = re.compile(r"^\s*#[^\n]*\n")


def frame_prompt(annotated: str) -> str:
    return PROMPT_HEAD + annotated.strip() + PROMPT_TAIL


def strip_leading_comment(program: str) -> str:
    """The first line when it is a `# ...` comment (the echoed question), else nothing."""
    return _LEAD_COMMENT_RX.sub("", program, count=1)


def fact_names(prompt: str) -> list[str]:
    """The names the host knows for a chain prompt: subject and object of every fact line, in order."""
    names = []
    for line in prompt.split("\n"):
        m = _FACT_RX.match(line.strip())
        if m:
            names += [m.group("obj").strip(), m.group("subj").strip()]
    return names


def _entity_role(task: str, role: str) -> bool:
    if task == "chain":
        return bool(re.match(r"^H\d+_", role, re.I))      # the hop objects
    if task == "plan":
        return role.upper() == "SEED"                     # the HOP words stay words
    return False


_STEP_VALUE_RX = re.compile(r"(#\s*step\s+\d+:[^\n=]*?)\s*=\s*[^\n]*")


def drop_step_values(program: str) -> str:
    """`# step 0: $N1 - $N2 = 12` -> `# step 0: $N1 - $N2`. The comment's value is a sum the emitter must do in
    its head; the 450M got it wrong (59 - 71 = 48) and the next step built on the wrong number, while the VM
    computes every step exactly anyway (2026-09-30)."""
    return _STEP_VALUE_RX.sub(r"\1", program)


def slot_record(rec: dict, words: bool = True, step_values: bool = True, world=None) -> dict | None:
    """One SFT record -> its slot form, or None when a bound literal is not covered by the prompt.
    `words`: numbers written in words are slots too; `step_values`: keep the `= value` of step comments;
    `world`: the arithmetic world (`cubbyllm.reasoning.arith_world`) whose conversion facts become $K slots."""
    task, prompt = rec["task"], rec["prompt"]
    program = strip_leading_comment(rec["program"])
    if not step_values and task in NUMBER_FAMILIES:
        program = drop_step_values(program)
    stats = {"entities": 0, "numbers": 0, "comment_numbers": 0}
    if task not in ENTITY_FAMILIES and task not in NUMBER_FAMILIES:
        return {**_base(rec), "prompt": frame_prompt(prompt), "program": program, "spans": [], "stats": stats}

    names = []
    if task in ENTITY_FAMILIES:
        # the index matches on normalized words, so a literal the harvest normalized ("chung an chi")
        # still finds its span in the prompt's own spelling ("chung an-chi"), and the span's text is
        # what the host fills back in
        names = fact_names(prompt) + [m.group("lit") for m in BIND_LIT_RX.finditer(program)
                                      if _entity_role(task, m.group("role"))]
    index = SpanIndex(dict.fromkeys(names)) if names else None
    consts = world.constants(prompt) if world is not None and task == "arithmetic" else None
    table = extract(prompt, index, numbers=task in NUMBER_FAMILIES, words=words, constants=consts)

    out = program
    for m in list(BIND_LIT_RX.finditer(program)):
        if not _entity_role(task, m.group("role")):
            continue
        lit = m.group("lit")
        span = next((s for s in table.spans if s.kind == "E" and _norm(s.text) == _norm(lit)), None)
        if span is None:
            return None                                   # uncovered: the host could not have placed it
        out = out.replace(f'"{lit}"', f'"{span.id}"', 1)
        stats["entities"] += 1

    if task in NUMBER_FAMILIES:
        by_num = {}                                       # by value: "3.50" in the text is the program's 3.5
        for s in table.spans:
            if s.kind in "NK" and num_value(s.filled) is not None:
                by_num.setdefault(canon(num_value(s.filled)), s.id)

        def slot_of(text):
            v = num_value(text)
            return by_num.get(canon(v)) if v is not None else None

        def sub_lit(m):
            sid = slot_of(m.group("num"))
            if sid is None:
                return m.group(0)
            stats["numbers"] += 1
            a, b = m.start("num") - m.start(), m.end("num") - m.start()      # the literal's own span, never the
            return m.group(0)[:a] + sid + m.group(0)[b:]                     # register's digit (`div s2, 2` -> `div s2, $N5`)
        out = NUM_LIT_RX.sub(sub_lit, out)

        def sub_comment(m):
            head = re.match(r"#\s*step\s+\d+:", m.group(0))     # the step's index is not a number to slot
            lead = head.group(0) if head else ""
            text = m.group(0)[len(lead):]

            def one(n):
                sid = slot_of(n.group(0))
                if sid is None:
                    return n.group(0)
                stats["comment_numbers"] += 1
                return sid
            return lead + re.sub(r"(?<![\w.$-])-?\d+(?:\.\d+)?(?![\w.])", one, text)
        out = re.sub(r"#[^\n]*", sub_comment, out)
        # what the emitter still has to write itself: unit constants (60, 7), derived ones ("five days")
        stats["constants_left"] = sum(num_value(m.group("num")) not in (None, 0.0) for m in NUM_LIT_RX.finditer(out))

    spans = [{"id": s.id, "text": s.text, "start": s.start, "end": s.end, "kind": s.kind,
              **({"value": s.value} if s.value is not None else {})} for s in table.spans]
    return {**_base(rec), "prompt": frame_prompt(table.annotate()), "program": out, "spans": spans, "stats": stats}


def _base(rec: dict) -> dict:
    return {"id": rec["id"], "task": rec["task"], "subtype": rec.get("subtype", ""), "split": rec["split"],
            "gold": rec.get("gold"), "repeat": int(rec.get("repeat", 1) or 1), "reference": rec["program"],
            "question": rec["prompt"]}


def table_of(spans: list[dict], question: str = "") -> SlotTable:
    """The slot table back from a record's `spans`, to fill a generated program."""
    return SlotTable(question, [Span(s["id"], s["text"], s["start"], s["end"], s["kind"], s.get("value"))
                                for s in spans])


def fill(program: str, spans: list[dict]) -> str:
    return table_of(spans).fill(program)


def convert(src: str, dst: str, tasks: tuple[str, ...] | None = None, words: bool = True,
            step_values: bool = True, world=None) -> dict:
    """The SFT jsonl -> its slot form on disk; returns the manifest (kept/dropped per family, slot counts)."""
    kept, dropped = Counter(), Counter()
    slots = Counter()
    with open(dst, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if tasks and r["task"] not in tasks:
                continue
            if r.get("vm_ok") is False or r.get("gold_match") is False:
                dropped[f"{r['task']}:vm"] += 1
                continue
            s = slot_record(r, words=words, step_values=step_values, world=world)
            if s is None:
                dropped[f"{r['task']}:uncovered"] += 1
                continue
            kept[(r["task"], r["split"])] += 1
            for k, v in s["stats"].items():
                slots[f"{r['task']}:{k}"] += v
            slots[f"{r['task']}:records_with_constants"] += int(s["stats"].get("constants_left", 0) > 0)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return {"src": os.path.basename(src), "kept": {f"{t}/{sp}": n for (t, sp), n in sorted(kept.items())},
            "dropped": dict(dropped), "slots": dict(slots)}


def encode_records(path, tk, eos, split, max_len, tasks=None):
    """(prompt ids, program ids + </s>) rows of one split, with the training-time repeat applied."""
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] != split or (tasks and r["task"] not in tasks):
            continue
        p = tk.encode(r["prompt"]).ids
        t = tk.encode(r["program"].rstrip() + "\n").ids + [eos]
        if len(p) + len(t) > max_len:
            continue
        row = {"p": p, "t": t, "family": r["task"], "id": r["id"]}
        rows += [row] * (r.get("repeat", 1) if split == "train" else 1)
    return rows


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


def val_records(path, per_task=0, seed=1):
    """The val split's records for generation, stratified per family."""
    import random
    by = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "val":
            by.setdefault(r["task"], []).append(r)
    out = []
    for t in sorted(by):
        rows = by[t]
        random.Random(seed).shuffle(rows)
        out += rows[:per_task] if per_task else rows
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="convert an emitter SFT jsonl to its slot form")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tasks", default="", help="comma-separated families to keep (default: all)")
    ap.add_argument("--no-words", action="store_true", help="digits only (the 2026-09-29 slot files)")
    ap.add_argument("--drop-step-values", action="store_true", help="`# step 0: $N1 - $N2` without its `= value`")
    ap.add_argument("--world", action="store_true", help="the arithmetic world's conversion facts as $K slots")
    a = ap.parse_args()
    world = None
    if a.world:
        from cubbyllm.reasoning.arith_world import ArithmeticWorld
        world = ArithmeticWorld()
    m = convert(a.src, a.dst, tuple(t for t in a.tasks.split(",") if t) or None, words=not a.no_words,
                step_values=not a.drop_step_values, world=world)
    print(json.dumps(m, indent=1))
