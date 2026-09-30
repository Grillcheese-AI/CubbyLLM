"""build_program_first -- program-first emitter data, Caco-style (H-E18, 2026-09-29).

Wired: STANDALONE (stand-in data tooling; nothing in cubbyllm/ imports this).

Caco (arXiv 2510.04081, NeurIPS 2025) builds reasoning data the other way round: sample executable
programs first, run them, THEN have an LLM write the question, and keep a question only when a blind
re-solve agrees with the program. Our emitter set is built question -> program from fixed sources
(v12e: 11,845 records -- GSM8K train, the regen kernels, the harvest); this builder adds the reverse
direction for two families:

  arith   a grammar sampler of GSM-shaped step DAGs (+ - * /, integers, every step used), written in
          v12e's exact arithmetic dialect and run on the VM (its value must equal the sampler's). A
          question is then written for it -- several per program, each in a different situation
          (Caco's problem-level augmentation); the DAG shapes v12e never had are its pattern-level
          augmentation, counted per run (`novel_shape`).
  chain   paths walked through a fact store (subject -> object, the store's reused relations, ASCII
          names -- the VM's lexer mangles the rest), never a path whose question the store's own
          dataset asks (every record is a new composition, and no eval question can leak). The chain
          program is the serving path's own `build_chain_program`, run on the VM. The canonical
          question is then reworded K ways; each wording yields a `plan` record (question -> CotPlan
          with the STORE's relation labels: paraphrase -> label is what the planner must learn) and a
          `chain` record (question + facts -> CotChain).

Every question passes two host checks first:
  coverage   every number the program uses is in the question (digits, or its word: 'three', 'half',
             'dozen'), and the seed entity is in it (normalized words): the host can place every span
             (H-E15's slot discipline);
  no leak    no computed value, intermediate entity or answer is stated in the question;
then the ROUND TRIP (`--check answer`, `emitter`, or both):
  answer     Caco's answer consistency: a blind solver (the dataset LLM, shown only the question)
             answers; kept when it equals the VM's value (arith) or names the same plan from the
             store's relation list (chain);
  emitter    ours: an Emitter writes a program from the question alone and the VM runs it; kept when
             VM(P') == VM(P) (arith) or its plan is the path's under the walk's own relation tolerance
             (chain). Execution checks the question; there is no LLM judge anywhere (Caco's
             CoT-consistency judge has no counterpart here).
With both, a question is kept when either agrees, and `pf.passed` records which.

The LLM only BUILDS DATA (the standing rule: nothing external at serve time). The kept program is the
SAMPLED one -- verified by construction -- never the solver's. Records use v12e's schema (source
'program_first/<family>', subtype 'pf:...', `pf` = provenance), so validation/emitter_data.py converts
them to slot form unchanged. `--heldout` builds the gate's eval set from bucket 0 of the shape/seed hash
(train uses 1-9), with whatever writer you name -- use a different one than train's.

  python standin/data/build_program_first.py selfcheck --n 300
  python standin/data/build_program_first.py arith --n 2000 --variants 2 --writer openrouter:<model> --check answer --out <jsonl>
  python standin/data/build_program_first.py chain --n 1000 --variants 3 --writer local:<gguf> --check emitter --solver-gguf <gguf> --out <jsonl>
  python standin/data/build_program_first.py mix --base <emitter_sft_v12e.jsonl> --pf <a.jsonl> <b.jsonl> --out <jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.append(_p)                     # appended: standin/hypothesis.py must not shadow the package
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cubbyllm.reasoning.planner import normalize, parse_fact, relation_matches  # noqa: E402
from cubbyllm.reasoning.programs import build_chain_program  # noqa: E402
from cubbyllm.reasoning.slots import NUM_IN_TEXT_RX  # noqa: E402

__wiring__ = "STANDALONE"

EMITTER_SYSTEM = ("You are the CubeLang emitter. Given a question or instruction, output ONLY a complete "
                  "CubeLang program that solves it. No prose, no explanation.")   # = identity.EMITTER_SYSTEM (v12e's)
OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
MAX_VALUE = 100_000


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def bucket(key: str) -> int:
    """0-9 from the key's hash: 0 is the held-out eval bucket, 1-9 are train."""
    return int(_sha(key)[:8], 16) % 10


# ── arithmetic: the sampler ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Operand:
    kind: str                                   # "lit" | "reg"
    v: int                                      # the literal, or the register (step) index


@dataclass(frozen=True)
class Step:
    a: Operand
    op: str
    b: Operand


def _val(o: Operand, vals: list[int]) -> int:
    return o.v if o.kind == "lit" else vals[o.v]


def values(steps: list[Step]) -> list[int]:
    """Each step's value, in order; ValueError on a non-integer division."""
    vals: list[int] = []
    for st in steps:
        a, b = _val(st.a, vals), _val(st.b, vals)
        if st.op == "+":
            v = a + b
        elif st.op == "-":
            v = a - b
        elif st.op == "*":
            v = a * b
        else:
            if b == 0 or a % b:
                raise ValueError("inexact division")
            v = a // b
        vals.append(v)
    return vals


def inputs_of(steps: list[Step]) -> list[int]:
    """The literals the program takes from the world, in order of first use."""
    out: list[int] = []
    for st in steps:
        for o in (st.a, st.b):
            if o.kind == "lit" and o.v not in out:
                out.append(o.v)
    return out


class ArithSampler:
    """GSM-shaped step DAGs: 2-10 steps, every step but the last consumed later, integer values in
    (0, MAX_VALUE], no trivial step (x*1, x/1, x+0, x-x, a repeated step)."""

    OP_WEIGHTS = (("+", 0.38), ("*", 0.27), ("-", 0.20), ("/", 0.15))
    STEP_WEIGHTS = ((2, 0.20), (3, 0.24), (4, 0.20), (5, 0.14), (6, 0.10), (7, 0.06), (8, 0.04), (9, 0.01), (10, 0.01))

    def __init__(self, seed: int = 0, max_steps: int = 10) -> None:
        self.rng = random.Random(seed)
        keep = [(n, w) for n, w in self.STEP_WEIGHTS if n <= max_steps]
        total = sum(w for _, w in keep)
        self.step_weights = tuple((n, w / total) for n, w in keep)   # a weak writer gets the short programs

    def _pick(self, weights):
        r, acc = self.rng.random(), 0.0
        for x, w in weights:
            acc += w
            if r <= acc:
                return x
        return weights[-1][0]

    def _amount(self) -> int:
        r = self.rng.random()
        if r < 0.35:
            return self.rng.randint(2, 30)
        if r < 0.75:
            return 5 * self.rng.randint(2, 40)
        return self.rng.randint(31, 400)

    def _small(self) -> int:
        return self.rng.randint(2, 12)

    def _step(self, k: int, n: int, vals: list[int], unused: list[int]) -> Step | None:
        op = self._pick(self.OP_WEIGHTS)
        must = len(unused) >= 2 or (k == n - 1 and unused)
        take = [u for u in unused]
        if take and (must or self.rng.random() < 0.6):     # else a new literal branch (GSM's "her parents gave 15")
            a = Operand("reg", take.pop(-1 if self.rng.random() < 0.6 else self.rng.randrange(len(take))))
        elif k > 0 and self.rng.random() < 0.1:
            a = Operand("reg", self.rng.randrange(k))
        else:
            a = Operand("lit", self._amount())
        av = _val(a, vals)
        if take and (len(take) >= 1 and (k == n - 1 or self.rng.random() < 0.3)) and op in "+-*":
            b = Operand("reg", take.pop())
        elif op == "*":
            b = Operand("lit", self._small() if self.rng.random() < 0.75 else self._amount())
        elif op == "/":
            divs = [d for d in range(2, 13) if av % d == 0 and av // d >= 1]
            if not divs:
                if a.kind != "lit":
                    return None
                d = self._small()
                a = Operand("lit", d * self.rng.randint(2, 40))
                av = a.v
                divs = [d]
            b = Operand("lit", self.rng.choice(divs))
        elif op == "-":
            if av <= 2:
                return None
            b = Operand("lit", self.rng.randint(1, av - 1)) if self.rng.random() < 0.5 else \
                Operand("lit", max(1, min(av - 1, 5 * self.rng.randint(1, max(1, av // 5)))))
        else:
            b = Operand("lit", self._amount())
        return Step(a, op, b)

    def sample(self, n_steps: int | None = None, tries: int = 200) -> list[Step]:
        for _ in range(tries):
            n = n_steps or self._pick(self.step_weights)
            steps: list[Step] = []
            vals: list[int] = []
            unused: list[int] = []
            ok = True
            for k in range(n):
                st = self._step(k, n, vals, unused)
                if st is None:
                    ok = False
                    break
                steps.append(st)
                try:
                    v = values(steps)[-1]
                except ValueError:
                    ok = False
                    break
                if not (0 < v <= MAX_VALUE) or trivial(st, vals):
                    ok = False
                    break
                vals.append(v)
                for o in (st.a, st.b):
                    if o.kind == "reg" and o.v in unused:
                        unused.remove(o.v)
                unused.append(k)
            if ok and unused == [n - 1] and len(set(steps)) == len(steps):
                return steps
        raise RuntimeError("the sampler found no valid program in its tries")


def trivial(st: Step, vals: list[int]) -> bool:
    a, b = _val(st.a, vals), _val(st.b, vals)
    if st.op in "*/" and (b == 1 or a == 1):
        return True
    if st.op in "+-" and (a == 0 or b == 0):
        return True
    if st.op == "-" and st.a == st.b:
        return True
    return any(o.kind == "lit" and o.v == 0 for o in (st.a, st.b))


# ── arithmetic: v12e's dialect, and its shapes ──────────────────────────────
_HEAD = ("program {name} implements ISolver {{\n    type Input = str;\n    type Output = quantity;\n    @external\n"
         "    public function parse(raw: str): Input {{ return raw; }}\n\n"
         "    public pure function verify(input: Input, output: Output): bool {{ return true; }}\n\n\n\n"
         "    @external\n    public function solve(input: Input): Output {{\n")


def arith_program(steps: list[Step], question: str = "", name: str | None = None) -> str:
    """The program exactly as v12e writes one (GSM-derived: `create`, `assign`, one op per step, a copy
    `assign 0; add sK, sJ` when the left side is an earlier step, `sum; query; return` on the last),
    with the question echoed as the first-line comment."""
    vals = values(steps)
    name = name or f"GSM{int(_sha(repr(steps))[:6], 16) % 10000}"
    lines = []
    for k, st in enumerate(steps):
        a, b = _val(st.a, vals), _val(st.b, vals)
        lines.append(f"        create s{k} : quantity;   # step {k}: {a} {st.op} {b} = {vals[k]}")
        if st.a.kind == "lit":
            lines.append(f"        assign s{k} = {st.a.v};")
        else:
            lines += [f"        assign s{k} = 0;", f"        add s{k}, s{st.a.v};"]
        rhs = f"s{st.b.v}" if st.b.kind == "reg" else str(st.b.v)
        lines.append(f"        {OPS[st.op]} s{k}, {rhs};")
    last = len(steps) - 1
    lines += [f"        sum s{last};", f"        query s{last};", f"        return s{last};"]
    head = f"# {' '.join(question.split())}\n" if question else ""
    return head + _HEAD.format(name=name) + "\n".join(lines) + "\n    }\n}\n"


_STEP_LINE = re.compile(r"^\s*(create|assign|add|sub|mul|div|sum|query|return)\b\s*(.*?);", re.M)


def arith_shape(program: str) -> str | None:
    """The program's structure with its numbers erased: per step, where its left side comes from (a
    literal L or an earlier step rK) and each op with its right side. Computed from the TEXT, so a
    sampled program and a v12e program are compared by one rule. None when it is not the dialect."""
    if "public function solve(input: Input): Output" not in program:
        return None
    body = program.split("public function solve(input: Input): Output", 1)[1]
    steps: list[list[str]] = []
    for m in _STEP_LINE.finditer(body):
        op, rest = m.group(1), m.group(2).strip()
        if op == "create":
            steps.append([])
        elif op == "assign" and steps:
            val = rest.split("=", 1)[1].strip() if "=" in rest else rest
            steps[-1].append("0" if val == "0" else "L")
        elif op in ("add", "sub", "mul", "div") and steps:
            _reg, rhs = [x.strip() for x in rest.split(",", 1)]
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
            src = f"r{rhs[1:]}" if re.fullmatch(r"s\d+", rhs) else "L"
            if steps[-1] == ["0"] and sym == "+" and src != "L":      # the copy: the left side is an earlier step
                steps[-1] = [src]
            else:
                steps[-1].append(sym + src)
    return "|".join("".join(s) for s in steps) if steps else None


# ── the host checks on a written question ──────────────────────────────────
_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen twenty").split()
NUMBER_WORDS = {i: {w} for i, w in enumerate(_WORDS)}
NUMBER_WORDS.update({30: {"thirty"}, 40: {"forty"}, 50: {"fifty"}, 60: {"sixty"}, 70: {"seventy"}, 80: {"eighty"},
                     90: {"ninety"}, 100: {"hundred"}})
# a constant GSM text says without its digits; v12e's programs keep these literal (the slot converter leaves
# a number the question does not write as digits), so the emitter already knows them
IMPLICIT = {2: {"half", "halves", "twice", "double", "doubled", "doubles", "pair", "both"},
            3: {"triple", "tripled", "thrice", "third", "thirds"}, 4: {"quarter", "quarters", "fourth"},
            12: {"dozen", "dozens"}}
_TOKEN = re.compile(r"[a-z]+")


def numbers_in(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUM_IN_TEXT_RX.finditer(text)}


def covered(v: int, text: str, nums: set[str] | None = None) -> bool:
    nums = numbers_in(text) if nums is None else nums
    if str(v) in nums:
        return True
    words = set(_TOKEN.findall(text.lower()))
    return bool(words & (NUMBER_WORDS.get(v, set()) | IMPLICIT.get(v, set())))


def check_arith_question(q: str, steps: list[Step]) -> str | None:
    """None when the question is usable, else the reason it is not."""
    if not q or len(q) < 20 or len(q) > 700:
        return "length"
    if "?" not in q:
        return "no_question"
    nums = numbers_in(q)
    ins = inputs_of(steps)
    for v in ins:
        if not covered(v, q, nums):
            return "uncovered_number"
    vals = values(steps)
    for v in vals:                                      # a computed value the question gives away
        if v not in ins and str(v) in nums:
            return "states_value"
    return None


def check_plan_question(q: str, seed: str, hidden: list[str]) -> str | None:
    """The seed must be named (normalized words); no intermediate entity or the answer may be."""
    if not q or len(q) < 10 or len(q) > 400:
        return "length"
    nq = f" {normalize(q)} "
    ns = normalize(seed)
    if f" {ns} " not in nq:
        return "no_seed"
    nq = nq.replace(f" {ns} ", " ")                     # an entity inside the seed's own name is not stated
    for h in hidden:
        nh = normalize(h)
        if nh and nh != ns and f" {nh} " in nq:
            return "states_entity"
    return None


# ── the VM ──────────────────────────────────────────────────────────────────
def vm_run(program: str, fn: str = "solve") -> tuple[bool, object, str | None]:
    """(ok, result, error) from the real VM (`cubelang run-proto`)."""
    from cubbyllm.bridges import cubelang_client as cc
    try:
        out = cc.run_program_proto(program, fn=fn)
        return bool(out.get("ok")), out.get("result"), None
    except Exception as e:                              # compile or run error: the program does not pass
        return False, None, f"{type(e).__name__}: {e}"[:300]


def same_value(result, value) -> bool:
    try:
        return abs(float(result) - float(value)) < 1e-6
    except (TypeError, ValueError):
        return False


# ── the dataset LLM (writes questions and solves them blind; never judges) ──
class OpenRouterWriter:
    """`chat(system, user)` through standin/openrouter.py: cached on disk (a rerun is offline and
    byte-identical), usage tallied, the key never printed."""

    def __init__(self, model: str, temperature: float = 0.8, effort: str = "") -> None:
        from openrouter import OpenRouterProposer
        self.name = f"openrouter:{model}" + (f"@{effort}" if effort else "")
        # a thinking model's reasoning tokens count against max_tokens: "off" disables it, low/medium/high set it,
        # "" leaves the provider's default (and the cap gets headroom either way unless it is off)
        reasoning = {"enabled": False} if effort == "off" else ({"effort": effort} if effort else None)
        self.headroom = 0 if effort == "off" else 2000
        self.p = OpenRouterProposer(model, temperature=temperature, reasoning=reasoning)

    def chat(self, system: str, user: str, max_tokens: int = 400, temperature: float | None = None,
             retries: int = 8) -> str:
        import http.client
        import urllib.error
        if temperature is not None:
            self.p.temperature = temperature
        for attempt in range(retries + 1):
            try:
                return self.p.chat(user, system=system, max_tokens=max_tokens + self.headroom)
            except urllib.error.HTTPError as e:
                if e.code == 402:                       # no credit: every further call fails the same way
                    raise RuntimeError("OpenRouter 402: the account has no credit left (top up, or use a :free model)") from e
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                if attempt == retries:                  # a cut stream (IncompleteRead) is a blip, not an answer
                    raise
            time.sleep(min(60, 5 * 2 ** attempt))       # free models rate-limit; providers blip
        return ""

    @property
    def usage(self) -> dict:
        return dict(self.p.usage, calls=self.p.calls)


class LocalWriter:
    """A local instruct GGUF through llama-cpp's chat template (no cost, no network)."""

    def __init__(self, gguf: str, n_ctx: int = 4096, n_gpu_layers: int = -1, temperature: float = 0.8) -> None:
        self.name, self.gguf, self.n_ctx, self.ngl, self.temperature = f"local:{os.path.basename(gguf)}", gguf, n_ctx, n_gpu_layers, temperature
        self._llm = None
        self.calls = 0

    def chat(self, system: str, user: str, max_tokens: int = 400, temperature: float | None = None) -> str:
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.gguf, n_ctx=self.n_ctx, n_gpu_layers=self.ngl, verbose=False, seed=0)
        self.calls += 1
        t = self.temperature if temperature is None else temperature
        out = self._llm.create_chat_completion(messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": user}],
                                               temperature=t, max_tokens=max_tokens, seed=self.calls)
        return (out["choices"][0]["message"].get("content") or "").strip()

    @property
    def usage(self) -> dict:
        return {"calls": self.calls, "cost": 0.0}


def make_writer(spec: str, temperature: float = 0.8):
    kind, _, arg = spec.partition(":")
    if kind == "openrouter":                            # openrouter:<model>[@off|low|medium|high]
        model, _, effort = arg.partition("@")
        return OpenRouterWriter(model, temperature=temperature, effort=effort)
    if kind == "local":                                 # PF_WRITER_NGL=0 keeps it on the CPU beside a GPU emitter
        return LocalWriter(arg, temperature=temperature, n_gpu_layers=int(os.environ.get("PF_WRITER_NGL", "-1")))
    raise ValueError(f"writer must be openrouter:<model> or local:<gguf>, got {spec!r}")


# ── prompts ─────────────────────────────────────────────────────────────────
ARITH_WRITE_SYSTEM = """You write short math word problems for a data set. You are given a computation as steps over some input numbers. Write ONE word problem, set in the situation you are given, whose solution is exactly these steps, in this order.
Rules:
- Use every input number exactly as given, written as digits.
- Each step is something that happens in the story (bought, gave away, split equally, packed into boxes of ...).
- Never state the result of any step, and never state the answer.
- End with one question that asks for the result of the last step.
- One to four sentences. Write only the problem: no title, no list, no steps, no solution.

Example.
Situation: a bakery
Computation:
start with 12, multiply by 3 -> A
A minus 5 -> B
the question asks for B
Problem: A baker fills 3 trays with 12 muffins each. She gives 5 muffins to her neighbor. How many muffins does she have left?"""

ARITH_SOLVE_SYSTEM = """Solve the math word problem. Work it out briefly, then give the final answer on the last line as: Answer: <number>"""

PLAN_REWORD_SYSTEM = """You rewrite a question in different words, for a data set.
Rules:
- Keep the entity name exactly as it is written in the question.
- Keep the meaning: the same chain of relations, in the same order, asking for the same thing.
- Change the wording: say the relations the way a person would, and vary the sentence frame (who, which, where, name the, tell me).
- Never add a name, a date or any fact that is not in the question.
Write the rewrites one per line. No numbering, no quotes, nothing else."""

PLAN_SOLVE_SYSTEM = """You turn a question into a relation-chain plan over a knowledge graph.
Answer with ONE JSON object and nothing else: {"seed": <the entity the question starts from, exactly as named>, "hops": [<relation>, ...]}.
hops[0] applies to the seed, hops[1] to its result, and so on; the last hop's value is the answer.
Choose every relation from this list, spelled exactly as listed: """

SITUATIONS = ("a bakery", "a school trip", "a farm", "a bookstore", "a garden", "a road trip", "a toy shop",
              "a soccer season", "saving pocket money", "a birthday party", "a library", "a factory shift",
              "a camping weekend", "a bake sale", "a science fair", "a zoo", "a music practice schedule",
              "a fundraiser", "a construction site", "a fishing trip", "a coffee shop", "a train journey",
              "a knitting project", "an apple orchard", "a video game tournament", "a charity run",
              "a hardware store", "a summer job", "a recycling drive", "a flower shop")


_OP_WORDS = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def steps_text(steps: list[Step]) -> str:
    """The computation as the writer sees it: one step per line in words, results named A, B, ... and never a
    computed value (so the writer cannot state one). Symbol tables made the 2.6B writer echo them back."""
    names = [chr(ord("A") + i) for i in range(len(steps))]

    def side(o: Operand) -> str:
        return str(o.v) if o.kind == "lit" else names[o.v]
    lines = []
    for k, st in enumerate(steps):
        if k == 0 and st.a.kind == "lit" and st.b.kind == "lit":
            verb = {"+": "add", "-": "subtract", "*": "multiply by", "/": "divide by"}[st.op]
            lines.append(f"start with {st.a.v}, {verb} {st.b.v} -> {names[k]}")
        else:
            lines.append(f"{side(st.a)} {_OP_WORDS[st.op]} {side(st.b)} -> {names[k]}")
    lines.append(f"the question asks for {names[-1]}")
    return "\n".join(lines)


def arith_user(situation: str, steps: list[Step]) -> str:
    return f"Situation: {situation}\nComputation:\n{steps_text(steps)}\nProblem:"


def last_number(text: str) -> str | None:
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text
    nums = [m.group(0).replace(",", "") for m in NUM_IN_TEXT_RX.finditer(tail)]
    return nums[-1] if nums else None


_ECHO = re.compile(r"^\s*(#|inputs?\s*:|situation\s*:|computation\s*:|the question asks|start with \d|"
                   r"[A-Z]\s*=|\w+\s+(plus|minus|times|divided by)\s+\w+\s*->|answer\s*:|solution\s*:)", re.I)


def _clean_question(text: str) -> str:
    """The problem alone: think blocks and stray tags out, echoed computation / solution lines out, and
    everything after the first question mark (the answer chatter a small writer appends)."""
    t = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    t = re.sub(r"</?(think|s|answer|final|response|analysis)>", " ", t)
    lines = [l for l in t.splitlines() if l.strip() and not _ECHO.match(l)]
    t = " ".join(" ".join(lines).split()).strip().strip('"').strip()
    t = re.sub(r"^(problem|question|word problem)\s*:\s*", "", t, flags=re.I)
    if "?" in t:
        t = t[: t.index("?") + 1]
    return t


def _clean_program(text: str) -> str:
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts[1::2], key=len) if len(parts) > 2 else t.replace("```", "")
        t = re.sub(r"^\w*\n", "", t)
    from build_emitter_sft import shim_isolver
    return shim_isolver(t.strip() + "\n")


# ── the round trip ──────────────────────────────────────────────────────────
@dataclass
class Checks:
    """Which round trips to run. `solver` is the dataset LLM (answer mode), `emitter` any object with
    `.emit(prompt, system=, max_new_tokens=, temperature=, seed=)` (emitter mode); `vm` is (program, fn) ->
    (ok, result, error), injectable for tests."""
    modes: tuple = ("answer",)
    solver: object = None
    emitter: object = None
    samples: int = 3                                    # emitter tries: greedy, then sampled at 0.7
    vm: object = vm_run
    counts: Counter = field(default_factory=Counter)

    def arith(self, q: str, value: int) -> list[str]:
        passed = []
        if "answer" in self.modes and self.solver is not None:
            got = last_number(self.solver.chat(ARITH_SOLVE_SYSTEM, q, max_tokens=500, temperature=0.0))
            if got is not None and same_value(got, value):
                passed.append("answer")
        if "emitter" in self.modes and self.emitter is not None:
            for i in range(max(1, self.samples)):
                prog = _clean_program(self.emitter.emit(q, system=EMITTER_SYSTEM, max_new_tokens=700,
                                                        temperature=0.0 if i == 0 else 0.7, seed=i))
                ok, res, _err = self.vm(prog, "solve")
                if ok and same_value(res, value):
                    passed.append("emitter" if i == 0 else "emitter_sampled")
                    break
        for p in passed:
            self.counts[p] += 1
        return passed

    def plan(self, q: str, seed: str, rels: list[str], vocab: list[str]) -> list[str]:
        passed = []
        if "answer" in self.modes and self.solver is not None:
            text = self.solver.chat(PLAN_SOLVE_SYSTEM + json.dumps(vocab), q, max_tokens=300, temperature=0.0)
            s, hops = parse_plan_json(text)
            if s is not None and normalize(s) == normalize(seed) and [normalize(h) for h in hops] == [normalize(r) for r in rels]:
                passed.append("answer")
        if "emitter" in self.modes and self.emitter is not None:
            for i in range(max(1, self.samples)):
                s, hops = parse_cot_plan(self.emitter.emit(q, system=EMITTER_SYSTEM, max_new_tokens=300,
                                                           temperature=0.0 if i == 0 else 0.7, seed=i))
                if (s is not None and normalize(s) == normalize(seed) and len(hops) == len(rels)
                        and all(relation_matches(r, h) for r, h in zip(rels, hops))):
                    passed.append("emitter" if i == 0 else "emitter_sampled")
                    break
        for p in passed:
            self.counts[p] += 1
        return passed


_BIND = re.compile(r'bind\s+\w+\s*,\s*(SEED|HOP(\d+))\s*,\s*"((?:[^"\\]|\\.)*)"\s*;')


def parse_cot_plan(text: str) -> tuple[str | None, list[str]]:
    seed, hops = None, {}
    for m in _BIND.finditer(text or ""):
        if m.group(1) == "SEED":
            seed = m.group(3)
        else:
            hops[int(m.group(2))] = m.group(3)
    return seed, [hops[k] for k in sorted(hops)]


def parse_plan_json(text: str) -> tuple[str | None, list[str]]:
    t = (text or "").strip()
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except (ValueError, TypeError):
        return None, []
    seed, hops = obj.get("seed"), obj.get("hops") or []
    if not isinstance(seed, str) or not isinstance(hops, list) or not all(isinstance(h, str) for h in hops):
        return None, []
    return seed.strip(), [h.strip() for h in hops]


def cot_plan(seed: str, rels: list[str]) -> str:
    """v12e's plan program: SEED and HOP1..k, normalized (the planner's own form)."""
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    binds = "\n".join([f'        bind frame, SEED, "{esc(normalize(seed))}";'] +
                      [f'        bind frame, HOP{i + 1}, "{esc(normalize(r))}";' for i, r in enumerate(rels)])
    return ("use vsa;\n\nprogram CotPlan implements ISolve {\n    public function solve(mention: str): str {\n"
            "        create frame: number;\n" + binds + "\n        return recover(frame, SEED);\n    }\n}\n")


# ── records ─────────────────────────────────────────────────────────────────
def record(task: str, subtype: str, family: str, prompt: str, program: str, gold, vm_result, split: str,
           pf: dict) -> dict:
    return {"id": f"{task}-pf-{_sha(prompt + program)[:12]}", "task": task, "subtype": subtype,
            "source": f"program_first/{family}", "prompt": prompt, "program": program,
            "gold": None if gold is None else str(gold), "vm_ok": True,
            "vm_result": None if vm_result is None else str(vm_result), "vm_error": None, "gold_match": True,
            "split": split, "system": EMITTER_SYSTEM, "repeat": 1, "state": None, "pf": pf}


def _example(examples: dict[str, list], why: str, item: dict, keep: int = 3) -> None:
    """A few rejects per reason, into the manifest: the first thing to read when acceptance is low."""
    bucket_ = examples.setdefault(why, [])
    if len(bucket_) < keep:
        bucket_.append(item)


def v12e_shapes(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == "arithmetic":
                s = arith_shape(r["program"])
                if s:
                    out.add(s)
    return out


def build_arith(n: int, writer, checks: Checks, variants: int = 2, seed: int = 0, heldout: bool = False,
                known_shapes: set[str] | None = None, vm=vm_run, log=print, max_steps: int = 10) -> tuple[list[dict], dict]:
    """`n` programs sampled and VM-verified, `variants` questions written for each, kept when the host
    checks and a round trip pass. Held-out mode keeps bucket-0 shapes, train mode buckets 1-9."""
    sampler = ArithSampler(seed, max_steps=max_steps)
    rng = random.Random(seed + 1)
    known = known_shapes or set()
    stats = Counter()
    examples: dict[str, list] = {}
    out: list[dict] = []
    t0 = time.time()
    while stats["programs"] < n and stats["sampled"] < 50 * n:
        steps = sampler.sample()
        stats["sampled"] += 1
        skeleton = arith_program(steps)
        shape = arith_shape(skeleton)
        if (bucket(shape) == 0) != heldout:
            stats["other_bucket"] += 1
            continue
        vals = values(steps)
        ok, res, err = vm(skeleton, "solve")
        if not ok or not same_value(res, vals[-1]):
            stats["vm_disagrees"] += 1
            continue
        stats["programs"] += 1
        novel = shape not in known
        stats["novel_shape"] += novel
        spec = steps_text(steps)
        for vi in range(variants):
            situation = rng.choice(SITUATIONS)
            raw = writer.chat(ARITH_WRITE_SYSTEM, arith_user(situation, steps), max_tokens=400)
            q = _clean_question(raw)
            stats["written"] += 1
            stats[f"written:steps={len(steps)}"] += 1
            why = check_arith_question(q, steps)
            if why:
                stats[f"rejected:{why}"] += 1
                _example(examples, why, {"spec": spec, "raw": raw[:600]})
                continue
            passed = checks.arith(q, vals[-1])
            if not passed:
                stats["rejected:round_trip"] += 1
                _example(examples, "round_trip", {"spec": spec, "q": q, "value": vals[-1]})
                continue
            stats["kept"] += 1
            stats[f"kept:steps={len(steps)}"] += 1
            prog = arith_program(steps, q)
            out.append(record("arithmetic", f"pf:steps={len(steps)}", "arith", q, prog, vals[-1], res,
                              "val" if heldout else "train",
                              {"shape": shape, "novel_shape": novel, "situation": situation, "variant": vi,
                               "writer": getattr(writer, "name", "?"), "passed": passed}))
        if stats["programs"] % 50 == 0:
            log(f"  arith {stats['programs']}/{n} programs, kept {stats['kept']}/{stats['written']} questions "
                f"({time.time() - t0:.0f}s)")
    return out, dict(stats, examples=examples)


# ── chains: paths through a fact store ──────────────────────────────────────
def question_body(q: str) -> str:
    """'Where is the X of the Y of Z?' -> 'x of the y of z' (normalized): the part a path decides."""
    t = normalize(q.strip().rstrip("?"))
    t = re.sub(r"^(what|who|where|which|when|whom|whose|how many|how much)\s+(is|was|are|were)\s+", "", t)
    return normalize(t)


def canonical_question(seed: str, rels: list[str]) -> str:
    return "What is the " + " of the ".join(reversed(rels)) + f" of {seed}?"


def load_pq(path: str) -> tuple[list[str], set[str]]:
    """The chain-QA pool: every fact, and every question's body (the exclusion set)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["question_prompt", "facts"])
    facts, bodies = [], set()
    for q, fs in zip(t.column("question_prompt").to_pylist(), t.column("facts").to_pylist()):
        if q:
            bodies.add(question_body(q))
        for f in fs or []:
            facts.append(" ".join(str(f).split()))
    return list(dict.fromkeys(facts)), bodies


class PathSampler:
    """Paths subject -> object through a TripleIndex: every relation one the store REUSES (>= 2 facts;
    a one-off 'relation' is usually a mis-split entity), every name ASCII (the VM lexer mangles the
    rest), no entity twice, and never a path whose question body is in `exclude`."""

    HOP_WEIGHTS = ((1, 0.25), (2, 0.45), (3, 0.30))

    def __init__(self, facts: list[str], exclude: set[str] | None = None, seed: int = 0) -> None:
        from cubbyllm.reasoning import TripleIndex
        self.index = TripleIndex(facts)
        self.rel_n = self.index.relations()
        self.facts = [f for f in facts if f in self.index]
        self.exclude = exclude or set()
        self.rng = random.Random(seed)

    def reused(self) -> list[str]:
        return sorted(r for r, n in self.rel_n.items() if n >= 2)

    def _ok(self, t) -> bool:
        return (self.rel_n.get(normalize(t.rel), 0) >= 2 and all(x.isascii() and 1 <= len(x) <= 80 for x in (t.obj, t.rel, t.subj)))

    def sample(self, tries: int = 400):
        """-> (seed, rels, triples, facts) or None."""
        for _ in range(tries):
            r, acc, want = self.rng.random(), 0.0, 1
            for h, w in self.HOP_WEIGHTS:
                acc += w
                if r <= acc:
                    want = h
                    break
            f0 = self.rng.choice(self.facts)
            t0 = parse_fact(f0, known={x for x, n in self.rel_n.items() if n >= 2})
            if t0 is None or not self._ok(t0):
                continue
            triples, facts, seen = [t0], [f0], {normalize(t0.subj), normalize(t0.obj)}
            while len(triples) < want:
                nxt = [(f, t) for f, t in self.index.about(triples[-1].obj) if self._ok(t) and normalize(t.obj) not in seen]
                if not nxt:
                    break
                f, t = self.rng.choice(nxt)
                triples.append(t)
                facts.append(f)
                seen.add(normalize(t.obj))
            if len(triples) < want:
                continue
            seed, rels = t0.subj, [t.rel for t in triples]
            if question_body(canonical_question(seed, rels)) in self.exclude:
                continue
            return seed, rels, triples, facts
        return None


def build_chain(n: int, sampler: PathSampler, writer, checks: Checks, variants: int = 3, heldout: bool = False,
                vm=vm_run, distractors: int = 15, log=print) -> tuple[list[dict], dict]:
    """`n` new paths, each VM-verified, reworded `variants` ways; every kept wording is a plan record and a
    chain record. Held-out mode keeps bucket-0 seeds, train mode buckets 1-9."""
    stats = Counter()
    examples: dict[str, list] = {}
    out: list[dict] = []
    seen_paths: set = set()
    vocab_all = sampler.reused()
    rng = random.Random(7)
    t0 = time.time()
    while stats["paths"] < n and stats["sampled"] < 40 * n:
        got = sampler.sample()
        stats["sampled"] += 1
        if got is None:
            stats["no_path"] += 1
            continue
        seed, rels, triples, facts = got
        key = (normalize(seed), tuple(normalize(r) for r in rels))
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if (bucket(normalize(seed)) == 0) != heldout:
            stats["other_bucket"] += 1
            continue
        chain, _fns = build_chain_program(triples, rels)
        fn = f"hop_{len(rels)}" if len(rels) > 1 else "solve"
        ok, res, err = vm(chain, fn)
        answer = triples[-1].obj
        if not ok or res is None or normalize(str(res)) != normalize(answer):
            stats["vm_disagrees"] += 1
            continue
        stats["paths"] += 1
        canon = canonical_question(seed, rels)
        text = writer.chat(PLAN_REWORD_SYSTEM, f"Write {variants} rewrites of: {canon}", max_tokens=120 * variants)
        wordings = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l) for l in text.splitlines()]
        wordings = [w for w in (_clean_question(l) for l in wordings) if w][:variants]
        hidden = [t.obj for t in triples]
        vocab = sorted(set([normalize(r) for r in rels] + rng.sample(vocab_all, min(distractors, len(vocab_all)))))
        for vi, q in enumerate(wordings):
            stats["written"] += 1
            why = "verbatim" if normalize(q) == normalize(canon) else check_plan_question(q, seed, hidden)
            if why:
                stats[f"rejected:{why}"] += 1
                _example(examples, why, {"canonical": canon, "q": q})
                continue
            passed = checks.plan(q, seed, rels, vocab)
            if not passed:
                stats["rejected:round_trip"] += 1
                _example(examples, "round_trip", {"canonical": canon, "q": q})
                continue
            stats["kept"] += 1
            pf = {"seed": seed, "relations": rels, "canonical": canon, "variant": vi,
                  "writer": getattr(writer, "name", "?"), "passed": passed}
            split = "val" if heldout else "train"
            sub = f"pf:n_hop={len(rels)}"
            out.append(record("plan", sub, "chain", q, cot_plan(seed, rels), None, None, split, pf))
            out.append(record("chain", sub, "chain", q + "\nFacts:\n" + "\n".join(f"- {f}" for f in facts),
                              chain, answer, res, split, pf))
        if stats["paths"] % 50 == 0:
            log(f"  chain {stats['paths']}/{n} paths, kept {stats['kept']}/{stats['written']} wordings ({time.time() - t0:.0f}s)")
    return out, dict(stats, examples=examples)


# ── CLI ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(ROOT, "standin", "data", "out")
V12E = os.path.join(OUT_DIR, "emitter_sft_v12e.jsonl")
PQ = os.environ.get("STANDIN_CHAIN_PQ", r"E:\valid_scaling_law_with_facts.pq")


def _write(records: list[dict], path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = dict(manifest, output=os.path.basename(path), n_records=len(records),
                    by_task=dict(Counter(r["task"] for r in records)),
                    built=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.splitext(path)[0] + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f"wrote {path}: {len(records)} records {manifest['by_task']}")


def _checks(args, writer=None) -> Checks:
    modes = tuple(m.strip() for m in args.check.split(",") if m.strip())
    solver = None
    if "answer" in modes:                               # the writer's own instance when it is the same model (one load)
        solver = writer if (writer is not None and not args.solver) else make_writer(args.solver or args.writer, temperature=0.0)
    emitter = None
    if "emitter" in modes:
        from standin.emitter import LlamaCppEmitter
        emitter = LlamaCppEmitter(args.solver_gguf, n_ctx=4096)
    return Checks(modes=modes, solver=solver, emitter=emitter, samples=args.samples)


def cmd_selfcheck(args) -> None:
    """Sampled programs through the real VM: the sampler's value must be the VM's, and the shapes are
    counted against v12e's (no LLM)."""
    s = ArithSampler(args.seed)
    known = v12e_shapes(V12E)
    agree = novel = 0
    shapes = set()
    t0 = time.time()
    for i in range(args.n):
        steps = s.sample()
        prog = arith_program(steps, "self-check")
        ok, res, err = vm_run(prog, "solve")
        v = values(steps)[-1]
        if ok and same_value(res, v):
            agree += 1
        elif i < 50 or not ok:
            print(f"  DISAGREE: {v} vs {res} ({err})\n{prog}")
        sh = arith_shape(prog)
        shapes.add(sh)
        novel += sh not in known
    print(f"selfcheck: {agree}/{args.n} programs: VM value == sampler value; {len(shapes)} distinct shapes, "
          f"{novel}/{args.n} programs with a shape v12e has none of ({len(known)} v12e shapes); "
          f"{(time.time() - t0) / max(1, args.n) * 1000:.0f} ms/program")


def cmd_arith(args) -> None:
    writer = make_writer(args.writer)
    checks = _checks(args, writer)
    recs, stats = build_arith(args.n, writer, checks, variants=args.variants, seed=args.seed, heldout=args.heldout,
                              known_shapes=v12e_shapes(V12E), max_steps=args.max_steps)
    stats["round_trip"] = dict(checks.counts)
    stats["writer_usage"] = getattr(writer, "usage", {})
    stats["solver_usage"] = "the writer" if checks.solver is writer else getattr(checks.solver, "usage", {})
    print(json.dumps(stats, indent=1, default=str))
    _write(recs, args.out, {"family": "arith", "args": vars(args), "stats": stats})


def cmd_chain(args) -> None:
    facts, bodies = load_pq(args.pq)
    print(f"store: {len(facts):,} facts, {len(bodies):,} dataset questions excluded")
    sampler = PathSampler(facts, exclude=bodies, seed=args.seed)
    writer = make_writer(args.writer)
    checks = _checks(args, writer)
    recs, stats = build_chain(args.n, sampler, writer, checks, variants=args.variants, heldout=args.heldout)
    stats["round_trip"] = dict(checks.counts)
    stats["writer_usage"] = getattr(writer, "usage", {})
    stats["solver_usage"] = "the writer" if checks.solver is writer else getattr(checks.solver, "usage", {})
    print(json.dumps(stats, indent=1, default=str))
    _write(recs, args.out, {"family": "chain", "args": vars(args), "stats": stats})


def cmd_mix(args) -> None:
    """The base set plus program-first records (train only; `repeat` 1), for the H-E18 arm. The control
    is the base set alone, trained for the SAME number of steps (train_emitter_torch --steps)."""
    rows = [json.loads(l) for l in open(args.base, encoding="utf-8")]
    n_base = len(rows)
    ids = {r["id"] for r in rows}
    added = Counter()
    for p in args.pf:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["split"] != "train" or r["id"] in ids:
                continue
            ids.add(r["id"])
            rows.append(r)
            added[r["task"]] += 1
    _write(rows, args.out, {"family": "mix", "base": os.path.basename(args.base), "n_base": n_base,
                            "added": dict(added), "pf": [os.path.basename(p) for p in args.pf]})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfcheck")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    for name in ("arith", "chain"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=200, help="programs (arith) or paths (chain)")
        p.add_argument("--variants", type=int, default=2 if name == "arith" else 3)
        p.add_argument("--writer", required=True, help="openrouter:<model> or local:<gguf>")
        p.add_argument("--solver", default="", help="the blind solver for --check answer (default: the writer)")
        p.add_argument("--check", default="answer", help="answer, emitter, or answer,emitter (either keeps)")
        p.add_argument("--solver-gguf", default=os.path.join(ROOT, "standin", "models", "emitter_v12e.Q4_K_M.gguf"))
        p.add_argument("--samples", type=int, default=3)
        p.add_argument("--heldout", action="store_true", help="bucket 0 only, split=val: the gate's eval set")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)
        if name == "chain":
            p.add_argument("--pq", default=PQ)
        else:
            p.add_argument("--max-steps", type=int, default=10, help="longest program (2-10); a weak writer keeps few long ones")
    m = sub.add_parser("mix")
    m.add_argument("--base", default=V12E)
    m.add_argument("--pf", nargs="+", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"selfcheck": cmd_selfcheck, "arith": cmd_arith, "chain": cmd_chain, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
