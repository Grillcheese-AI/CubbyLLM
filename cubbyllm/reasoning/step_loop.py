"""step_loop -- arithmetic one step at a time, the state written back by the host (H-E19, 2026-09-30).

Wired: STANDALONE (the re-cut feeds `validation/train_emitter_torch.py`; the loop wraps any `Emitter` for the
VM read, `validation/exp_he19_step_loop.py`).

The teacher-forced read (`validation/exp_he19_next_step.py`, 2026-09-30) showed the 450M emitter picks a plan's
first step at 80% and every later one at ~22% with the gold program in front of it: 87% of later steps chain
from a register (`add s1, s0`) and the model starts from a fresh question number 65% of the time. A register
name is a pointer with no content the trunk can bind to; a number with its words in the question it binds at
91%. So the state leaves the program and comes back as slots the host places, in the trunk's own currency:

    Question:
    You split [N1: 308] dollars ... between [N2: 2] savings jars, then save [N3: 6] times ...
    So far:
    [S1: 154 = N1 / N2]
    Program:
            create s1 : quantity;   # step 1: $S1 * $N3
            assign s1 = $S1;
            mul s1, $N3;

One emission per step (~40 tokens, never a 640-token program), the host fills the slots, wraps the block into a
whole program (`wrap_step`), runs it in the VM, and writes the value back as the next $S; `return $S3;` is the
stop. Every intermediate value is the VM's, never the model's; a copied literal or an unknown slot refuses the
step (H-E15); the plan never lives in hidden state, so it can be inspected, stopped and blamed at its step.

`recut(record)` turns a slot-form arithmetic record (`emitter_data.slot_record`) into its step rows -- one per
block plus the stop -- with the gold state computed by simulating the gold program over the placed values (a
record whose simulation misses the gold is dropped: the 2% the replay read flagged). The chaining idiom
`assign s1 = 0; add s1, s0; mul s1, $N3` collapses to `assign s1 = $S1; mul s1, $N3`. `StepLoop` runs the loop
around an Emitter; `recut_file` is the converter (CLI below).

    python -m cubbyllm.reasoning.step_loop standin/data/out/emitter_sft_v12e_w_slots.jsonl \\
        standin/data/out/emitter_sft_v12e_w_step.jsonl [--only-arithmetic] [--limit N]
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from cubbyllm.reasoning.slots import SLOT_RX, SlotTable, Span, canon, extract, num_value

PROMPT_HEAD, PROMPT_TAIL = "Question:\n", "\nProgram:\n"
INDENT = "        "
MAX_STEPS = 12
_CREATE = re.compile(r"^\s*create\s+(?P<reg>s\d+)\s*:")
_OP = re.compile(r"^\s*(?:(?P<assign>assign)\s+(?P<r1>s\d+)\s*=\s*(?P<v1>[^;]+);|(?P<op>add|sub|mul|div)\s+(?P<r2>s\d+)\s*,\s*(?P<v2>[^;]+);)")
_CLOSE = re.compile(r"^\s*(sum|query|return)\b")
_RETURN_S = re.compile(r"^\s*return\s+(?P<sid>\$S\d+)\s*;")
_REG = re.compile(r"^s\d+$")
SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
PROGRAM_HEAD = """program {name} implements ISolver {{
    type Input = str;
    type Output = quantity;
    @external
    public function parse(raw: str): Input {{ return raw; }}

    public pure function verify(input: Input, output: Output): bool {{ return true; }}



    @external
    public function solve(input: Input): Output {{
"""


# ---- the program's blocks -----------------------------------------------------------------------------------
@dataclass
class Block:
    reg: str                              # the register the block creates (s0)
    ops: list = field(default_factory=list)   # [(op, operand)], operand as written: $N1, s0, 7, 0


def blocks_of(program: str) -> list[Block]:
    """The solve body as blocks: a `create` line and the ops that follow it, up to the closing sum/query/return."""
    out: list[Block] = []
    for line in (program or "").splitlines():
        if _CLOSE.match(line):
            break
        m = _CREATE.match(line)
        if m:
            out.append(Block(m.group("reg")))
            continue
        m = _OP.match(line)
        if m and out:
            if m.group("assign"):
                out[-1].ops.append(("assign", m.group("v1").strip()))
            else:
                out[-1].ops.append((m.group("op"), m.group("v2").strip()))
    return out


def simulate(ops: list, values: dict) -> Fraction | None:
    """A block's value over `values` (slot ids and registers -> Fraction); None on a bad reference or op."""
    acc = None
    for op, v in ops:
        try:
            x = values[v] if (v.startswith("$") or _REG.match(v)) else Fraction(v.replace(",", ""))
        except (KeyError, ValueError, ZeroDivisionError):
            return None
        if op == "assign":
            acc = x
        elif acc is None:
            return None
        elif op == "add":
            acc += x
        elif op == "sub":
            acc -= x
        elif op == "mul":
            acc *= x
        elif op == "div":
            if x == 0:
                return None
            acc /= x
        else:
            return None
    return acc


def derivation(ops: list) -> str:
    """`$N1 / $N2 + $S1` -- the ops as one expression (left to right, as the VM applies them)."""
    parts = []
    for op, v in ops:
        parts.append(v if op == "assign" else f"{SYMBOL[op]} {v}")
    return " ".join(parts)


def to_state_form(block: Block, reg_slot: dict) -> list:
    """The block's ops with registers as $S slots and the `assign 0; add s0` idiom collapsed."""
    ops = [(op, reg_slot.get(v, v)) for op, v in block.ops]
    if len(ops) >= 2 and ops[0] == ("assign", "0") and ops[1][0] == "add":
        ops = [("assign", ops[1][1])] + ops[2:]
    return ops


def render_step(k: int, ops: list) -> str:
    """One step as the emitter writes it: the create line with its plan, then the ops."""
    lines = [f"{INDENT}create s{k} : quantity;   # step {k}: {derivation(ops)}"]
    for op, v in ops:
        lines.append(f"{INDENT}assign s{k} = {v};" if op == "assign" else f"{INDENT}{op} s{k}, {v};")
    return "\n".join(lines) + "\n"


def render_stop(sid: str) -> str:
    return f"{INDENT}return {sid};\n"


def wrap_step(block_filled: str, name: str = "STEP") -> str:
    """A filled step block as a whole program the VM runs: the v12e header, the block, its register returned."""
    m = next((_CREATE.match(l) for l in block_filled.splitlines() if _CREATE.match(l)), None)
    if m is None:
        raise ValueError("no create line in the step")
    reg = m.group("reg")
    body = "\n".join(l for l in block_filled.splitlines() if l.strip()) + "\n"
    return PROGRAM_HEAD.format(name=name) + body + f"{INDENT}sum {reg};\n{INDENT}query {reg};\n{INDENT}return {reg};\n    }}\n}}\n"


def parse_emission(text: str):
    """What the emitter said: ('stop', '$S3') | ('step', Block) | ('other', None); the first block only."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return "other", None
    m = _RETURN_S.match(lines[0])
    if m:
        return "stop", m.group("sid")
    blocks = blocks_of("\n".join(lines))
    if blocks and _CREATE.match(lines[0]):
        return "step", blocks[0]
    return "other", None


# ---- the re-cut ---------------------------------------------------------------------------------------------
def state_spans(values: list) -> list:
    return [Span(f"$S{i + 1}", deriv, -1, -1, "S", canon(float(v))) for i, (v, deriv) in enumerate(values)]


def with_state(prompt: str, state: list) -> str:
    """The record's framed prompt with the `So far:` block before `Program:`."""
    from cubbyllm.reasoning.slots import state_block
    if not state:
        return prompt
    head, tail = prompt.rsplit(PROMPT_TAIL, 1)
    return head + "\n" + state_block(state) + PROMPT_TAIL + tail


def recut(rec: dict) -> list[dict]:
    """A slot-form arithmetic record -> its step rows (one per block, then the stop), [] if it cannot be re-cut."""
    blocks = blocks_of(rec["program"])
    if not blocks:
        return []
    values: dict = {}
    for s in rec.get("spans", []):
        v = num_value(s.get("value") if s.get("value") is not None else s.get("text"))
        if v is not None:
            values[s["id"]] = Fraction(str(v))
    reg_slot: dict = {}
    rows, state = [], []
    span_dicts = list(rec.get("spans", []))
    K = len(blocks)
    for k, b in enumerate(blocks):
        ops = to_state_form(b, reg_slot)
        v = simulate(ops, values)
        if v is None:
            return []
        sid = f"$S{k + 1}"
        rows.append({"id": f"{rec['id']}#{k}", "task": "arithmetic", "subtype": f"step {k}/{K}; {rec.get('subtype', '')}",
                     "split": rec["split"], "gold": canon(float(v)), "repeat": int(rec.get("repeat", 1) or 1),
                     "reference": render_step(k, ops), "question": rec.get("question", ""),
                     "prompt": with_state(rec["prompt"], state), "program": render_step(k, ops),
                     "spans": span_dicts + [{"id": s.id, "text": s.text, "start": -1, "end": -1, "kind": "S", "value": s.value}
                                            for s in state], "stats": {"k": k, "K": K}})
        values[sid] = v
        values[b.reg] = v
        reg_slot[b.reg] = sid
        state = state + [Span(sid, derivation(ops).replace("$", ""), -1, -1, "S", canon(float(v)))]
    gold = num_value(rec.get("gold"))
    if gold is None or abs(float(values[f"$S{K}"]) - gold) > 1e-6:
        return []                                     # the gold program does not reproduce its gold here: not a lesson
    rows.append({"id": f"{rec['id']}#{K}", "task": "arithmetic", "subtype": f"stop {K}/{K}; {rec.get('subtype', '')}",
                 "split": rec["split"], "gold": rec.get("gold"), "repeat": int(rec.get("repeat", 1) or 1),
                 "reference": render_stop(f"$S{K}"), "question": rec.get("question", ""),
                 "prompt": with_state(rec["prompt"], state), "program": render_stop(f"$S{K}"),
                 "spans": span_dicts + [{"id": s.id, "text": s.text, "start": -1, "end": -1, "kind": "S", "value": s.value}
                                        for s in state], "stats": {"k": K, "K": K}})
    return rows


def recut_file(src: str, dst: str, only_arithmetic: bool = False, limit: int = 0) -> dict:
    """Every arithmetic record of `src` re-cut into step rows; the other families copied as they are (unless
    `only_arithmetic`). Returns the manifest."""
    kept, dropped = Counter(), Counter()
    n_arith = 0
    with open(dst, "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") != "arithmetic":
                if not only_arithmetic:
                    out.write(line + "\n")
                    kept[f"{r['task']}/{r['split']}"] += 1
                continue
            if limit and n_arith >= limit:
                continue
            n_arith += 1
            rows = recut(r)
            if not rows:
                dropped[f"arithmetic/{r['split']}"] += 1
                continue
            kept[f"arithmetic_records/{r['split']}"] += 1
            kept[f"arithmetic_step_rows/{r['split']}"] += len(rows)
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"src": os.path.basename(src), "dst": os.path.basename(dst), "kept": dict(sorted(kept.items())),
            "dropped": dict(dropped)}


# ---- the loop -----------------------------------------------------------------------------------------------
class StepLoop:
    """Solve a question one VM-verified step at a time around an `Emitter` (annotated prompts, slotted blocks).

    `vm(program) -> (ok, result)`; default: the real VM through the harvest's shim (validation/exp_he18_ab.vm_run).
    `solve` returns {"answer", "steps": [...], "refused": reason | None, "emissions"}; the answer is None when a
    step was refused (unknown slot, copied literal, malformed, the VM), the loop ran past `max_steps`, or the
    stop named a slot the host does not have."""

    def __init__(self, emitter, vm=None, world=None, max_steps: int = MAX_STEPS, max_new_tokens: int = 64):
        self.emitter, self.world, self.max_steps, self.max_new_tokens = emitter, world, max_steps, max_new_tokens
        self.vm = vm or default_vm()

    def solve(self, question: str) -> dict:
        table = extract(question, constants=self.world.constants(question) if self.world else None)
        steps, emissions = [], []
        for k in range(self.max_steps + 1):
            prompt = PROMPT_HEAD + table.annotate().strip() + PROMPT_TAIL
            raw = self.emitter.emit(prompt, max_new_tokens=self.max_new_tokens)
            emissions.append(raw)
            kind, what = parse_emission(raw)
            if kind == "stop":
                s = table.get(what)
                if s is None:
                    return {"answer": None, "steps": steps, "refused": f"stop on unknown slot {what}", "emissions": emissions}
                return {"answer": s.filled, "steps": steps, "refused": None, "emissions": emissions}
            if kind != "step":
                return {"answer": None, "steps": steps, "refused": "malformed step", "emissions": emissions}
            block_text = render_step(k, what.ops)
            verdict = table.check(block_text)
            if not verdict.ok:
                return {"answer": None, "steps": steps, "refused": verdict.reason, "emissions": emissions}
            ok, result = self.vm(wrap_step(table.fill(block_text)))
            v = num_value(result) if ok else None
            if v is None:
                return {"answer": None, "steps": steps, "refused": "vm: the step did not run", "emissions": emissions}
            sid = f"$S{k + 1}"
            table.spans.append(Span(sid, derivation(what.ops).replace("$", ""), -1, -1, "S", canon(v)))
            steps.append({"k": k, "ops": what.ops, "value": canon(v)})
        return {"answer": None, "steps": steps, "refused": f"no stop within {self.max_steps} steps", "emissions": emissions}


    # ---- gate D: a vote per step, the VM as the judge of every candidate ------------------------------------
    def candidates(self, prompt: str, n: int, temperature: float, seed: int) -> list[str]:
        """The greedy emission plus n-1 samples (one batched decode when the emitter has `emit_samples`)."""
        out = [self.emitter.emit(prompt, max_new_tokens=self.max_new_tokens)]
        if n <= 1:
            return out
        if hasattr(self.emitter, "emit_samples"):
            return out + list(self.emitter.emit_samples(prompt, n - 1, max_new_tokens=self.max_new_tokens,
                                                        temperature=temperature, seed=seed))
        return out + [self.emitter.emit(prompt, max_new_tokens=self.max_new_tokens, temperature=temperature,
                                        seed=seed + i) for i in range(n - 1)]

    def solve_vote(self, question: str, n: int = 5, temperature: float = 0.7, seed: int = 0) -> dict:
        """Each step: n candidates, every one slot-checked, filled and run in the VM; the options are the VALUES
        the runnable steps computed and the stop; the most-voted option wins (ties go to the greedy candidate's)
        and its value is written back. The chain's confidence is its weakest step's agreement (votes for the
        winner / n): the host speaks only above a threshold, and the read sweeps it (risk against coverage)."""
        table = extract(question, constants=self.world.constants(question) if self.world else None)
        steps, agreements = [], []
        for k in range(self.max_steps + 1):
            prompt = PROMPT_HEAD + table.annotate().strip() + PROMPT_TAIL
            votes: dict = {}
            order: list = []
            for j, raw in enumerate(self.candidates(prompt, n, temperature, seed * 1000 + k)):
                kind, what = parse_emission(raw)
                if kind == "stop":
                    s = table.get(what)
                    if s is None or s.kind != "S":
                        continue
                    key = ("stop", s.filled)
                    rep = what
                elif kind == "step":
                    block_text = render_step(k, what.ops)
                    if not table.check(block_text).ok:
                        continue
                    ok, result = self.vm(wrap_step(table.fill(block_text)))
                    v = num_value(result) if ok else None
                    if v is None:
                        continue
                    key = ("step", canon(v))
                    rep = what.ops
                else:
                    continue
                if key not in votes:
                    votes[key] = [0, rep, j]
                    order.append(key)
                votes[key][0] += 1
            if not votes:
                return {"answer": None, "steps": steps, "refused": "no candidate ran", "agreement": 0.0,
                        "agreements": agreements + [0.0]}
            win = max(order, key=lambda key: (votes[key][0], -votes[key][2]))
            agreements.append(votes[win][0] / n)
            if win[0] == "stop":
                return {"answer": win[1], "steps": steps, "refused": None, "agreement": min(agreements),
                        "agreements": agreements}
            sid = f"$S{k + 1}"
            table.spans.append(Span(sid, derivation(votes[win][1]).replace("$", ""), -1, -1, "S", win[1]))
            steps.append({"k": k, "ops": votes[win][1], "value": win[1], "votes": votes[win][0],
                          "options": len(votes)})
        return {"answer": None, "steps": steps, "refused": f"no stop within {self.max_steps} steps",
                "agreement": min(agreements) if agreements else 0.0, "agreements": agreements}


def default_vm():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in (os.path.join(root, "validation"), os.path.join(root, "standin"), os.path.join(root, "standin", "data")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from exp_he18_ab import vm_run
    return vm_run


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="re-cut a slot-form jsonl's arithmetic into step rows")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--only-arithmetic", action="store_true", help="drop the other families instead of copying them")
    ap.add_argument("--limit", type=int, default=0, help="at most N arithmetic records (0 = all)")
    a = ap.parse_args(argv)
    print(json.dumps(recut_file(a.src, a.dst, a.only_arithmetic, a.limit), indent=1))


if __name__ == "__main__":
    main()
