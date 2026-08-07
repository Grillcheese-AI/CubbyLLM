# CoT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cubbyllm → cubelang → bridge → output chain-of-thought pipeline: planner walks a question's relation chain, retrieval finds the facts, a CubeLang program holds the chain in a VSA frame and reads the answer back with per-hop cosine confidence, and the output is a verified answer + auditable trace or an honest "cannot verify".

**Architecture:** Three new torch-free modules in `cubbyllm/reasoning/` (planner, program builder, pipeline) with the retriever and VM runner injected as callables; the M1 retriever (v4 FastWordEncoder) and the eval live in `validation/exp_m3_cot_pipeline.py`. Spec: `docs/superpowers/specs/2026-08-07-cot-pipeline-design.md`.

**Tech Stack:** pure Python + regex in the package (numpy only in the eval script); `cubbyllm.bridges.cubelang_client.run_program_proto` for VM execution; pytest.

## Global Constraints

- **Symbolic boundary:** roles/fillers/program source cross to the VM; raw hypervectors never do.
- **No mowm import anywhere in `cubbyllm/`** (port-don't-link): the retriever is an injected callable; only the validation script touches mowm's `semantic_words` (via the existing `exp_m3_domain_routing._load_semantic_words` file-path loader).
- Every new module declares `__wiring__` (`from ..core.protocols import Wiring`); no file named `model.py`; `import cubbyllm` stays torch-free.
- Repair budget: **3 per question total**; `verified=true` is impossible with any hop below `tau_vm` (asserted in code).
- Both taus calibrated in the eval on a 200-question slice disjoint from the 800 eval questions; never hardcoded as universal.
- Test command: `python -m pytest tests -q` from the repo root; full suite must stay green.
- Commit trailer (every commit):
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01NMtc4rVecHjhJDrnpLhaZg`

---

### Task 1: Planner (question + fact grammar)

**Files:**
- Create: `cubbyllm/reasoning/__init__.py`
- Create: `cubbyllm/reasoning/planner.py`
- Test: `tests/reasoning/__init__.py` (empty), `tests/reasoning/test_planner.py`

**Interfaces:**
- Produces: `QuestionPlan(relations: list[str | None], tail: str, n_hop: int)` — `relations` in walk order, `relations[0] is None` (the tail hop resolves via the hop-1 fact), `relations[i]` for `i>=1` are relation segments.
- Produces: `Triple(obj: str, rel: str, subj: str)`.
- Produces: `parse_question(q: str) -> QuestionPlan | None`, `parse_fact(f: str) -> Triple | None`, `relation_matches(expected: str, got: str) -> bool`, `normalize(s: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/reasoning/test_planner.py
"""Grammar of the .pq corpus: templated questions and (two) fact templates."""
from cubbyllm.reasoning.planner import (
    QuestionPlan, Triple, normalize, parse_fact, parse_question,
    relation_matches)


def test_parse_three_hop_question():
    q = ("What is the continent of the country of the country of "
         "citizenship of cynthia basinet?")
    p = parse_question(q)
    assert p is not None and p.n_hop == 3
    assert p.tail == "country of citizenship of cynthia basinet"
    assert p.relations == [None, "country", "continent"]  # walk order


def test_parse_one_hop_question():
    p = parse_question("What is the capital of france?")
    assert p is not None and p.n_hop == 1
    assert p.relations == [None]
    assert p.tail == "capital of france"


def test_unparseable_question_returns_none():
    assert parse_question("Tell me about cheese.") is None


def test_parse_fact_standard_template():
    t = parse_fact("united stated is the country of citizenship of cynthia basinet")
    assert t == Triple(obj="united stated",
                       rel="country of citizenship", subj="cynthia basinet")


def test_parse_fact_is_in_template():
    # the corpus's second template: "X is the R Y is in"
    t = parse_fact("united stated is the country united stated is in")
    assert t == Triple(obj="united stated", rel="country", subj="united stated")


def test_parse_fact_rejects_freeform():
    assert parse_fact("cheese tastes great on toast") is None


def test_relation_matches_exact_and_fuzzy():
    assert relation_matches("country of citizenship", "country of citizenship")
    assert relation_matches("country", "country")
    assert not relation_matches("continent", "country")


def test_normalize_strips_articles_case_punct():
    assert normalize("  The Oceania Portal! ") == "oceania portal"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/reasoning/test_planner.py -q`
Expected: FAIL (ModuleNotFoundError: cubbyllm.reasoning)

- [ ] **Step 3: Implement**

```python
# cubbyllm/reasoning/__init__.py
"""Chain-of-thought reasoning pipeline (spec 2026-08-07-cot-pipeline).

Planner walks a question's relation chain; retrieval (injected) finds the
facts; a CubeLang program holds the chain in a VSA frame and reads the
answer out with per-hop cosine confidence. Symbols only ever cross the VM
boundary.
"""
from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED
```

```python
# cubbyllm/reasoning/planner.py
"""Question/fact grammar for the multi-hop corpus (spec section 4.1).

The .pq questions follow `What is the R_n of the R_{n-1} of ... of <tail>?`
where the tail mixes the hop-1 relation and the seed entity (relations
contain "of" themselves, e.g. "country of citizenship"), so the tail is
resolved by the hop-1 FACT parse, never guessed here. Facts follow two
observed templates; anything else parses to None — parse failures are
counted by callers, not raised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED

_Q = re.compile(r"^\s*what\s+is\s+the\s+(?P<body>.+?)\s*\?\s*$", re.I)
# "O is the R of S"  (non-greedy obj/rel, greedy subj)
_F_OF = re.compile(r"^(?P<obj>.+?) is the (?P<rel>.+?) of (?P<subj>.+)$")
# "O is the R S is in"  (the corpus's second template)
_F_IN = re.compile(r"^(?P<obj>.+?) is the (?P<rel>\S+) (?P<subj>.+) is in$")
_ARTICLES = ("the ", "a ", "an ")


@dataclass(frozen=True)
class QuestionPlan:
    relations: list[str | None]     # walk order; [0] is None (tail hop)
    tail: str
    n_hop: int


@dataclass(frozen=True)
class Triple:
    obj: str
    rel: str
    subj: str


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = " ".join(s.split())
    for a in _ARTICLES:
        if s.startswith(a):
            s = s[len(a):]
    return s


def parse_question(q: str) -> QuestionPlan | None:
    m = _Q.match(q)
    if not m:
        return None
    segments = m.group("body").split(" of the ")
    # segments run answer-side first: [R_n, R_{n-1}, ..., tail]
    tail = segments[-1].strip()
    rels = [s.strip() for s in segments[:-1]]
    relations: list[str | None] = [None] + list(reversed(rels))
    return QuestionPlan(relations=relations, tail=tail, n_hop=len(segments))


def parse_fact(f: str) -> Triple | None:
    f = " ".join(f.split())
    m = _F_IN.match(f) or _F_OF.match(f)
    if not m:
        return None
    return Triple(obj=m.group("obj").strip(), rel=m.group("rel").strip(),
                  subj=m.group("subj").strip())


def relation_matches(expected: str, got: str) -> bool:
    a = set(normalize(expected).split())
    b = set(normalize(got).split())
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.6
```

Note `_F_IN` is tried FIRST: "united stated is the country united stated is in"
also matches `_F_OF` (wrongly), so the more specific template wins.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/reasoning/test_planner.py -q`
Expected: 8 passed. Also run `python -m pytest tests -q` — guard tests must
accept the new modules (they declare `__wiring__`).

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/reasoning/__init__.py cubbyllm/reasoning/planner.py tests/reasoning/__init__.py tests/reasoning/test_planner.py
git commit -m "feat(reasoning): planner — question/fact grammar for the multi-hop corpus"
```
(with the Global Constraints trailer block)

---

### Task 2: Program builder

**Files:**
- Create: `cubbyllm/reasoning/programs.py`
- Test: `tests/reasoning/test_programs.py`

**Interfaces:**
- Consumes: `Triple` from Task 1.
- Produces: `build_chain_program(triples: list[Triple], relations: list[str]) -> tuple[str, list[str]]` — returns (CubeLang source, function names in hop order + `"control"` last). `relations[i]` is the display relation for hop i (for role naming); function names are `["solve", "hop_2", ..., "control"]`; roles are `H{i}_<SANITIZED>`; the control role `ABSENT_CTRL` is never bound.

- [ ] **Step 1: Write the failing tests**

```python
# tests/reasoning/test_programs.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/reasoning/test_programs.py -q`
Expected: FAIL (no module `programs`)

- [ ] **Step 3: Implement**

```python
# cubbyllm/reasoning/programs.py
"""Emit the per-question CubeLang chain program (spec section 4.3).

One frame holds every hop's (role -> object) binding in superposition; each
public function re-binds the frame (the VM is stateless per call) and
recovers ONE role, because `run-proto` returns one (symbol, similarity) per
call — the client batches functions. `solve(mention)` is hop 1 (ISolve's
required entry; the arg is ignored, matching the shipped
reasoning_bridge.cube), `hop_i` are the rest, `control` recovers the
never-bound ABSENT_CTRL role and must come back empty/low.
"""
from __future__ import annotations

import re

from ..core.protocols import Wiring
from .planner import Triple

__wiring__ = Wiring.WIRED

CONTROL_ROLE = "ABSENT_CTRL"


def sanitize_role(hop: int, rel: str) -> str:
    core = re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_").upper()
    return f"H{hop}_{core}"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_chain_program(triples: list[Triple],
                        relations: list[str]) -> tuple[str, list[str]]:
    roles = [sanitize_role(i + 1, r) for i, r in enumerate(relations)]
    binds = "\n".join(
        f'        bind frame, {role}, "{_escape(t.obj)}";'
        for role, t in zip(roles, triples))

    def fn(name: str, sig: str, role: str) -> str:
        return (f"    public function {name}({sig}): str {{\n"
                f"        create frame: number;\n{binds}\n"
                f"        return recover(frame, {role});\n    }}")

    parts = [fn("solve", "mention: str", roles[0])]
    fns = ["solve"]
    for i in range(1, len(roles)):
        parts.append(fn(f"hop_{i + 1}", "", roles[i]))
        fns.append(f"hop_{i + 1}")
    parts.append(fn("control", "", CONTROL_ROLE))
    fns.append("control")
    body = "\n\n".join(parts)
    return (f"use vsa;\n\nprogram CotChain implements ISolve {{\n{body}\n}}\n",
            fns)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/reasoning/test_programs.py -q` → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/reasoning/programs.py tests/reasoning/test_programs.py
git commit -m "feat(reasoning): CubeLang chain-program builder"
```

---

### Task 3: Pipeline (walk, repair, verify, honest fail)

**Files:**
- Create: `cubbyllm/reasoning/pipeline.py`
- Test: `tests/reasoning/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1's planner API; Task 2's `build_chain_program`, `CONTROL_ROLE`.
- Produces:
  - `HopTrace(query: str, fact: str, triple, ret_score: float, symbol: str | None, similarity: float | None)` (`triple: Triple | None`)
  - `CoTResult(answer: str | None, verified: bool, trace: list[HopTrace], repairs_used: int, reason: str | None)`
  - `answer(question: str, retrieve, run_fn, tau_vm: float, tau_ret: float, top_k: int = 3, max_repairs: int = 3) -> CoTResult` where
    `retrieve(query: str, k: int) -> list[tuple[float, str]]` (score, fact text) and
    `run_fn(source: str, fn: str) -> dict` (the `run_program_proto` shape: keys `result`, `similarity`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/reasoning/test_pipeline.py
"""Pipeline paths with a fake retriever and fake VM (no subprocess)."""
import pytest

from cubbyllm.reasoning.pipeline import answer

Q3 = ("What is the continent of the country of the country of "
      "citizenship of cynthia basinet?")
F1 = "united stated is the country of citizenship of cynthia basinet"
F2 = "united stated is the country united stated is in"
F3 = "oceania portal is the continent of united stated"


def good_retriever(query, k):
    ql = query.lower()
    if "cynthia" in ql:
        return [(0.9, F1)]
    if "continent" in ql:
        return [(0.9, F3)]
    return [(0.9, F2)]


def good_vm(source, fn):
    # echoes the bound object for each hop role; empty for the control
    want = {"solve": "united stated", "hop_2": "united stated",
            "hop_3": "oceania portal"}
    if fn == "control":
        return {"ok": True, "result": None, "similarity": None}
    return {"ok": True, "result": want[fn], "similarity": 0.93}


def test_happy_three_hop_walk_is_verified():
    r = answer(Q3, good_retriever, good_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is True
    assert r.answer == "oceania portal"
    assert len(r.trace) == 3
    assert all(h.similarity >= 0.5 for h in r.trace)


def test_unparseable_question_fails_honestly():
    r = answer("Tell me about cheese.", good_retriever, good_vm,
               tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False and r.answer is None
    assert r.reason == "unparseable"


def test_missing_hop_repairs_then_fails_honestly():
    def no_hop2(query, k):
        ql = query.lower()
        if "cynthia" in ql:
            return [(0.9, F1)]
        return [(0.9, "totally unrelated text with no template")]
    r = answer(Q3, no_hop2, good_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False and r.answer is None
    assert r.repairs_used == 3
    assert len(r.trace) >= 1                       # partial trace preserved


def test_low_similarity_recover_fails_honestly():
    def weak_vm(source, fn):
        out = good_vm(source, fn)
        if fn == "hop_3":
            out = {"ok": True, "result": "oceania portal", "similarity": 0.1}
        return out
    r = answer(Q3, good_retriever, weak_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False


def test_control_violation_fails():
    def leaky_vm(source, fn):
        if fn == "control":
            return {"ok": True, "result": "ghost", "similarity": 0.95}
        return good_vm(source, fn)
    r = answer(Q3, good_retriever, leaky_vm, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is False


def test_claimed_answer_invariant():
    r = answer(Q3, good_retriever, good_vm, tau_vm=0.5, tau_ret=0.2)
    if r.verified:
        assert all(h.similarity is not None and h.similarity >= 0.5
                   for h in r.trace)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/reasoning/test_pipeline.py -q`
Expected: FAIL (no module `pipeline`)

- [ ] **Step 3: Implement**

```python
# cubbyllm/reasoning/pipeline.py
"""The chain-of-thought walk: retrieve, verify in the VM, answer or refuse.

Retrieval walks the chain; the VM holds it, verifies it, and reads the
answer out (spec section 3). `verified=True` is impossible with any hop
below tau_vm — asserted at the single return site that sets it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .planner import (QuestionPlan, Triple, normalize, parse_fact,
                      parse_question, relation_matches)
from .programs import build_chain_program

__wiring__ = Wiring.WIRED


@dataclass
class HopTrace:
    query: str
    fact: str
    triple: Triple | None
    ret_score: float
    symbol: str | None = None
    similarity: float | None = None


@dataclass
class CoTResult:
    answer: str | None
    verified: bool
    trace: list[HopTrace] = field(default_factory=list)
    repairs_used: int = 0
    reason: str | None = None


def _accept(plan: QuestionPlan, hop: int, entity: str | None,
            t: Triple) -> bool:
    """Does this parsed fact serve hop `hop` of the plan?"""
    if hop == 0:
        # tail hop: the fact's "rel of subj" must reproduce the question tail
        return normalize(f"{t.rel} of {t.subj}") == normalize(plan.tail)
    expected = plan.relations[hop]
    assert expected is not None
    return (relation_matches(expected, t.rel)
            and normalize(t.subj) == normalize(entity or ""))


def _walk(plan: QuestionPlan, retrieve, tau_ret: float, top_k: int,
          budget: list[int], trace: list[HopTrace],
          banned: set[str]) -> list[Triple] | None:
    """Pick one accepted triple per hop; None when the budget dies.
    Bounded by `budget` alone (the spec's 3-per-question repair budget);
    `banned` holds facts a failed VM verify blacklisted."""
    triples: list[Triple] = []
    entity: str | None = None
    hop = 0
    question_tail = plan.tail
    while hop < plan.n_hop:
        if hop == 0:
            query = f"what is the {question_tail}"
        else:
            query = f"{entity} {plan.relations[hop]}"
        found = None
        while found is None:
            for score, fact in retrieve(query, top_k):
                if score < tau_ret or fact in banned:
                    continue
                t = parse_fact(fact)
                if t is not None and _accept(plan, hop, entity, t):
                    found = (score, fact, t)
                    break
            if found is None:
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                seen = " ".join(h.fact for h in trace)
                query = f"{query} {seen}" if seen else f"{query} {question_tail}"
        score, fact, t = found
        trace.append(HopTrace(query=query, fact=fact, triple=t,
                              ret_score=score))
        triples.append(t)
        entity = t.obj
        hop += 1
    return triples


def answer(question: str, retrieve, run_fn, tau_vm: float, tau_ret: float,
           top_k: int = 3, max_repairs: int = 3) -> CoTResult:
    plan = parse_question(question)
    if plan is None:
        return CoTResult(answer=None, verified=False, reason="unparseable")

    budget = [max_repairs]
    banned: set[str] = set()
    last_trace: list[HopTrace] = []
    # up to two walk+verify rounds (spec 4.4: one verify-stage repair pass)
    for _attempt in range(2):
        trace: list[HopTrace] = []
        triples = _walk(plan, retrieve, tau_ret, top_k, budget, trace, banned)
        used = max_repairs - budget[0]
        if triples is None:
            return CoTResult(answer=None, verified=False,
                             trace=trace or last_trace, repairs_used=used,
                             reason="retrieval_exhausted")

        display_rels = [(t.rel if i == 0 else plan.relations[i]) or t.rel
                        for i, t in enumerate(triples)]
        source, fns = build_chain_program(triples, display_rels)
        ok = True
        for i, fn in enumerate(fns[:-1]):                # hop functions
            out = run_fn(source, fn)
            trace[i].symbol = out.get("result")
            trace[i].similarity = out.get("similarity")
            if (trace[i].similarity is None or trace[i].similarity < tau_vm
                    or normalize(trace[i].symbol or "")
                    != normalize(triples[i].obj)):
                ok = False
        ctrl = run_fn(source, fns[-1])                   # control
        ctrl_sim = ctrl.get("similarity")
        if ctrl.get("result") is not None and ctrl_sim is not None \
                and ctrl_sim >= tau_vm:
            ok = False

        if ok:
            # the claimed-answer invariant, at the only verified=True site
            assert all(h.similarity is not None and h.similarity >= tau_vm
                       for h in trace)
            return CoTResult(answer=triples[-1].obj, verified=True,
                             trace=trace, repairs_used=used)

        # verify failed: blacklist the weakest hop's fact and retry once
        weakest = min(range(len(trace)),
                      key=lambda i: trace[i].similarity or -1.0)
        banned.add(trace[weakest].fact)
        last_trace = trace
        if budget[0] <= 0:
            break
        budget[0] -= 1

    return CoTResult(answer=None, verified=False, trace=last_trace,
                     repairs_used=max_repairs - budget[0],
                     reason="vm_verify_failed")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/reasoning/test_pipeline.py -q` → 6 passed.
Then `python -m pytest tests -q` → full suite green.

- [ ] **Step 5: Commit**

```bash
git add cubbyllm/reasoning/pipeline.py tests/reasoning/test_pipeline.py
git commit -m "feat(reasoning): CoT pipeline — walk, repair, VM verify, honest fail"
```

---

### Task 4: Package exports + live-VM smoke test

**Files:**
- Modify: `cubbyllm/reasoning/__init__.py`
- Test: `tests/reasoning/test_live_vm.py`

**Interfaces:**
- Produces: `from cubbyllm.reasoning import answer, CoTResult, HopTrace, parse_question, parse_fact, build_chain_program`.

- [ ] **Step 1: Write the (skippable) live test**

```python
# tests/reasoning/test_live_vm.py
"""End-to-end smoke through the real cubelang VM. Skipped when the binary
is absent (CI without the Rust toolchain)."""
import pytest

from cubbyllm.bridges import cubelang_client as cc
from cubbyllm.reasoning import answer

try:
    cc.find_cubelang_exe()
    HAVE_VM = True
except cc.CubelangNotFound:
    HAVE_VM = False

pytestmark = pytest.mark.skipif(not HAVE_VM, reason="cubelang exe not found")

Q2 = "What is the continent of the country of citizenship of ada lovelace?"
F1 = "england is the country of citizenship of ada lovelace"
F2 = "europe is the continent of england"


def retriever(query, k):
    return [(0.9, F1)] if "ada" in query.lower() else [(0.9, F2)]


def run_fn(source, fn):
    return cc.run_program_proto(source, fn=fn)


def test_two_hop_chain_through_real_vm():
    r = answer(Q2, retriever, run_fn, tau_vm=0.5, tau_ret=0.2)
    assert r.verified is True
    assert r.answer == "europe"
    assert all(h.similarity is not None and h.similarity >= 0.5 for h in r.trace)
```

- [ ] **Step 2: Update the package exports**

Append to `cubbyllm/reasoning/__init__.py` (below the `__wiring__` line):

```python
from .pipeline import CoTResult, HopTrace, answer  # noqa: E402
from .planner import parse_fact, parse_question    # noqa: E402
from .programs import build_chain_program          # noqa: E402

__all__ = ["CoTResult", "HopTrace", "answer", "parse_fact",
           "parse_question", "build_chain_program"]
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/reasoning/ -q` — the live test passes if the
cubelang exe is on this machine (it is: the reasoning-bridge tests use it),
otherwise skips. Then the full suite: `python -m pytest tests -q`.
Expected: green; live test PASSED (not skipped) on this machine.

- [ ] **Step 4: Commit**

```bash
git add cubbyllm/reasoning/__init__.py tests/reasoning/test_live_vm.py
git commit -m "feat(reasoning): exports + live-VM smoke (real recover roundtrip)"
```

---

### Task 5: Eval — `validation/exp_m3_cot_pipeline.py`

**Files:**
- Create: `validation/exp_m3_cot_pipeline.py`
- Output: `validation/logs/exp_m3_cot_pipeline.{log,json}`

**Interfaces:**
- Consumes: `cubbyllm.reasoning.answer`; `cubbyllm.bridges.cubelang_client.run_program_proto`; `exp_m3_domain_routing._load_semantic_words` + `youden_tau`; the v4 table at `D:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz`; `E:\valid_scaling_law_with_facts.pq`.

Structure (write exactly this shape; the numeric outputs go in the json):

```python
"""exp_m3_cot_pipeline — the CoT kill-criterion eval (spec section 7).

800 eval questions (seed 0, the exp_m3_injection sample), a DISJOINT
200-question calibration slice (seed 1) for tau_vm/tau_ret. Store = the
eval sample's unique facts. Arms: retrieval-only (top-1 fact's parsed
object), chase-only (n_hop-step chase, final fact's object), CoT pipeline
(walk + VM verify + readout). Normalized exact match vs the dataset answer,
by hop; claimed-answer precision; control pass rate.

Kill criterion (all four): CoT beats retrieval-only on 2-hop AND 3-hop;
claimed precision >= 0.90; zero verified results with a sub-tau hop
(asserted in the pipeline itself); control role below tau_vm in >= 95% of
programs. Standalone; never imported by cubbyllm/.
"""
```

Required elements (each is a concrete function in the script):

1. `load_sample(n, seed)` — reads `question_prompt, facts, answer, n_hop`
   via pyarrow; filters non-empty; dedups facts into a store exactly as
   `exp_m3_injection.load_rows` does; returns questions/answers/hops +
   store texts.
2. Retriever: encode the store once with the v4 table
   (`_load_semantic_words()`, `FastWordEncoder.from_npz(V4)`); `retrieve(q,
   k)` = unit-cosine top-k `(score, fact_text)`. `--table` flag defaults to
   the v4 path.
3. Calibration: run the pipeline on the 200-question calibration slice with
   permissive taus (`tau_vm=0.0, tau_ret=-1.0`), collect per-hop
   `(similarity, hop_correct)` where `hop_correct` = the recovered symbol
   equals the dataset chain's fact object for that hop; `tau_vm` = Youden
   on that separation via `youden_tau`; `tau_ret` = the 5th percentile of
   accepted-hop retrieval scores. Print both; store in the json; report the
   headline's sensitivity to ±20% tau_vm (re-verify the eval decisions at
   0.8·tau and 1.2·tau — cheap, no re-retrieval).
4. Baselines: `retrieval_only(q)` = top-1 fact, `parse_fact(...).obj` (None
   → wrong); `chase_only(q, n_hop)` = the exp_m3_injection chase protocol
   (reuse its logic inline with the same expansion), final fact's parsed
   object.
5. Main loop: per eval question, run all three arms; normalized exact match
   (`cubbyllm.reasoning.planner.normalize`) against the dataset answer.
   Track: per-hop-count accuracy per arm, CoT verified-coverage,
   claimed-answer precision, control pass rate, unparseable counts
   (questions and facts, separately), repairs histogram, mean wall ms per
   question per arm.
6. Verdict block: print PASS/FAIL for each of the four kill-criterion
   clauses, then the overall verdict.
7. Tee-able: plain prints, json to `validation/logs/exp_m3_cot_pipeline.json`.

Steps:

- [ ] **Step 1: Write the script** per the structure above.
- [ ] **Step 2: Smoke run**: `python validation/exp_m3_cot_pipeline.py --questions 40 --calibration 30 2>&1 | tee /tmp/cot_smoke.log` — verify it completes, the trace prints for one example question, and no arm errors.
- [ ] **Step 3: Full run**: `python validation/exp_m3_cot_pipeline.py 2>&1 | tee validation/logs/exp_m3_cot_pipeline.log` (defaults: 800 eval / 200 calibration).
- [ ] **Step 4: Record.** Read the verdict. Whatever it says — pass or informative falsification (chase-only ties CoT) — the numbers are the deliverable.
- [ ] **Step 5: Commit**

```bash
git add validation/exp_m3_cot_pipeline.py validation/logs/exp_m3_cot_pipeline.json validation/logs/exp_m3_cot_pipeline.log
git commit -m "feat(validation): CoT pipeline kill-criterion eval on the multi-hop corpus"
```

---

## Post-plan notes for the controller

- Task order is strict 1→5 (each consumes the previous task's interfaces).
- Tasks 1–3 are pure-Python TDD with complete code in this plan —
  cheapest-tier implementers. Task 4 needs the machine's cubelang exe
  (present; the bridges tests already use it). Task 5 is integration
  judgment — mid-tier, and its full run takes minutes (subprocess per
  hop-function: ~2,600 calls at tens of ms).
- After Task 5, update `CUBBYLLM_HYPOTHESES.md` (H-F2/H-B3 lineage) with the
  verdict and log links, per house rule — controller-level, not a subagent
  task.
- The mowm-worlds retriever, trunk-SFT program generation, and general-text
  triples are explicitly deferred (spec section 8) — do not scope-creep them in.
```
