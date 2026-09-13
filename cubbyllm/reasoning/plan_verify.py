"""plan_verify — the plan is a PROPOSAL; this disposes of it.

Wired: WIRED — `pipeline.answer(..., known=)` disposes of the plan here before any walk
(2026-09-11); `known=None` keeps the pre-disposer path.

WHY THIS EXISTS (2026-09-11)
----------------------------
The CubeLang program `programs.build_chain_program` emits is bind-only: it
carries the walked objects, and the VM verifies each recovery against them.
The PLAN — which relations, in which order, ending at which tail — never
crosses into the VM. `CubeLang.ISolver` has `parse / solve / verify`; the
emitted program has `solve` and the hop functions and nothing that checks the
chain derivation. So the plan is the one input in the whole pipeline that no
gate ever sees, and every misparse in the harvest (92 `misparsed_chain`, the
16 compound-relation residue of exp_r5) walked through that hole.

The fix is NOT to parse better. The grammar in `planner.py` is scaffolding —
it generates the training data for the emitter that replaces it — and the
emitter will propose plans that no regex ever produced. What is needed is a
DISPOSER: a pure, deterministic check that a proposed plan (a) is actually
the question, and (b) asks only for relations the store can answer. Model
proposes, host disposes.

WHAT IT CHECKS
--------------
  covers(question, plan)      the plan reconstructs the question: for a chain
                              (n_hop >= 2) the canonical body
                              "R_n of the R_{n-1} ... of the tail" is a
                              substring of the normalized question; for a
                              1-hop the tail's relation and entity both are.
                              Pure string. Catches an emitter that drops a
                              hop, invents one, or answers a different
                              question. Cannot catch a mis-SPLIT of the same
                              text (joining inverts splitting) — that is (b).
  unknown_relations(plan)     every relation the plan asks for — plan.relations
                              [1:] and the relation prefix of the tail — must
                              be a relation the store holds. This is what
                              catches the compound-relation misparse:
                                  want 'award received by the director of photography'
                                  have ['director of photography']
                              The store has no such edge, so the plan is
                              rejected BEFORE a walk is spent on it.
  segment(body, known)        the repair the disposer can propose back: every
                              segmentation of the body into known relations
                              and a tail whose relation prefix is known. When
                              exactly one exists, the split was decidable and
                              the grammar's " of the " guess was unnecessary.

`known` is INJECTED (`KnownRelations`: anything with `__contains__` over
normalized relation strings). Validation passes `StoreRelations(store)`;
tests pass a frozenset; MoWM's world can pass itself when a "possible edge"
oracle replaces "edge in store" — the spawn-a-latent-world path is the
unknown_relations != [] branch of this same verdict.

WHERE IT SITS
-------------
    question -> (grammar | emitter) -> plan -> **verify_plan** -> Retriever.seedable
                                                    |
                                                    +-- ok=False -> honest fail, reason carried
                                                    +-- repair    -> segment() -> unique? re-verify

WHICH HALF RUNS IN THE VM, AND WHY (read from cubelang/src, 2026-09-11)
-----------------------------------------------------------------------
(b) unknown_relations runs IN THE VM. `QUERY` (src/vm/engine.rs `op::QUERY`,
    src/vm/knowledge.rs) is an executing opcode: an EXACT, normalized hash
    lookup over an in-VM knowledge store loaded with `cubelang run --knowledge
    facts.jsonl`, pushing the chunk array -- ZERO chunks on a miss, never a
    nearest neighbour ("abstention is a result, not an error"). That is
    precisely "is this relation one the store holds", with the don't-know
    contract's own semantics. `VMRelations` below is the KnownRelations that
    does it: the store's relation vocabulary becomes the VM's knowledge, and
    membership is one QUERY per distinct relation through
    bridges/programs/plan_verify.cube. `StoreRelations` is the host-side
    reference it is checked against (exp_r6 --vm asserts they agree).

(a) covers() stays HOST-SIDE, and not by choice: the VM has no string
    semantics yet. `COMPARE` resolves both operands with `resolve_i64`, and
    `Value::Str.as_i64()` is 0 (src/vm/engine.rs), so `input == "abc"` is
    `0 == 0` -- true for any strings; `s.contains(x)` lowers to `CALL contains`
    (src/compiler.rs, MethodCall arm), an unresolved function. Both are
    pinned as `#[ignore]`d contracts in cubelang/tests/str_semantics.rs
    (2026-08-30). Until a string-equality/containment opcode executes, a
    substring check inside the VM would be a silent stub that passes
    verify-before-execute -- the exact failure class strict mode exists to
    catch. (CUBELANG_OPCODES_PLAN.md is stale: RETURN/CALL/LOOP/arrays are
    in the compiler. The gap is strings, not control flow.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from ..core.protocols import Wiring
from .planner import QuestionPlan, normalize, parse_fact, relation_matches, reused_relations

__wiring__ = Wiring.WIRED

JOINT = " of the "


@runtime_checkable
class KnownRelations(Protocol):
    """`rel in known` is EXACT membership (normalized). `match(rel)` is the
    walk's own hop>=1 tolerance -- `planner.relation_matches` (Jaccard >= 0.6
    over words) against the vocabulary -- and returns the relation it matched,
    or None. Two tiers on purpose: an approximate match never wears an exact
    hit's confidence (the same rule knowledge.rs states for QUERY), so the
    verdict records which relations were only paraphrase-known."""
    def __contains__(self, rel: str) -> bool: ...
    def match(self, rel: str) -> str | None: ...


# Words a question spends on its FRAME and its JOINTS, never on a hop. Anything
# else left over after the plan's relations and the seed entity are removed is
# a word the plan did not account for -- and if it is a word some relation in the
# store is made of, the plan very likely DROPPED A HOP.
_FRAME_AND_JOINT = frozenset("""
what which where who whom whose is was are were the a an of to in on at by for
that does do did have has had belong belongs contained within described describes
held located near from with as and or includes include contains contain
its it s thing
quel quelle quels quelles qui est sont le la les l de du des d un une à au aux en dans par pour
""".split())      # 'its it s thing': the possessive / relative frames (exp_r9, lever 2); the last line: French frames (lever 5)


def _relation_words(vocabulary) -> frozenset[str]:
    out = set()
    for r in vocabulary:
        out.update(w for w in normalize(r).split() if w not in _FRAME_AND_JOINT)
    return frozenset(out)


def _fuzzy_match(rel: str, vocabulary) -> str | None:
    """The walk's hop>=1 acceptance, applied to a vocabulary: first known relation
    `relation_matches` accepts, in sorted order (deterministic)."""
    for r in vocabulary:
        if relation_matches(rel, r):
            return r
    return None


class StoreRelations:
    """The relations the store can answer — a set over normalized `Triple.rel`."""

    def __init__(self, facts: Iterable[str]) -> None:
        self._rels: set[str] = set()
        self._n: dict[str, int] = {}          # facts per relation (the reuse guard)
        self.n_parsed = 0
        facts = list(facts)
        known = reused_relations(facts)       # the same two-pass split TripleIndex uses (lever 1b)
        for f in facts:
            t = parse_fact(f, known=known)
            if t is None:
                continue
            self.n_parsed += 1
            r = normalize(t.rel)
            self._rels.add(r)
            self._n[r] = self._n.get(r, 0) + 1

    @classmethod
    def from_keys(cls, keys: Iterable[str], counts: dict[str, int] | None = None) -> "StoreRelations":
        """A vocabulary from relation strings directly (e.g. the `key` column of a
        `write_vocab_jsonl` file) -- for a probe that has the vocabulary but not
        the corpus (exp_r3: no corpus, no encoder, just the GGUF). Without
        `counts` every key counts as reused (the paraphrase tier is open)."""
        sr = cls([])
        sr._rels = {normalize(k) for k in keys}
        sr._n = {r: 2 for r in sr._rels}
        if counts:
            sr._n.update({normalize(k): int(v) for k, v in counts.items()})
        sr.n_parsed = len(sr._rels)
        return sr

    @classmethod
    def from_vocab_jsonl(cls, path) -> "StoreRelations":
        import json, pathlib
        keys, counts = [], {}
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "key" in rec:
                    keys.append(rec["key"])
                    if "n" in rec:
                        counts[rec["key"]] = rec["n"]
        return cls.from_keys(keys, counts)

    def __contains__(self, rel: str) -> bool:
        return normalize(rel) in self._rels

    def declare(self, rel: str) -> None:
        """A relation a source states structurally: `add` splits with it from the
        first fact on (see TripleIndex.declare_relation). Not membership -- a
        declared relation is known only once a fact states it."""
        if not hasattr(self, "_declared"):
            self._declared: set[str] = set()
        self._declared.add(normalize(rel))

    def add(self, fact: str) -> str | None:
        """A fact learned after construction (search-and-learn, lever 3): its
        relation joins the vocabulary, split the way `TripleIndex.add` splits it
        (longest relation the store already reuses or a source declared, else
        greedy). Returns the normalized relation, or None for a non-template fact."""
        t = parse_fact(fact, known={r for r, k in self._n.items() if k >= 2} | getattr(self, "_declared", set()))
        if t is None:
            return None
        r = normalize(t.rel)
        self._rels.add(r)
        self._n[r] = self._n.get(r, 0) + 1
        return r

    def reused(self) -> list[str]:
        """The relations the store states in >= 2 facts -- the only ones hop 0's
        paraphrase tier may match. exp_m3 hop0 (2026-09-11): the greedy fact
        parse splits an entity with ' of ' in it into a one-off 'relation'
        ('genre of Joan Rivers: A Piece' | 'Work'); a plan's tail mis-split at
        the same point then paraphrase-matched it at Jaccard 0.625 and the VM
        verified a wrong answer. A relation is reused by nature; a fragment is
        not. Exact membership is unaffected."""
        return sorted(r for r in self._rels if self._n.get(r, 0) >= 2)

    def match(self, rel: str, *, reused_only: bool = False) -> str | None:
        """Exact tier, then the paraphrase tier; `reused_only` restricts the
        paraphrase tier to `reused()` -- what hop 0 asks for (see there)."""
        if rel in self:
            return normalize(rel)
        return _fuzzy_match(rel, self.reused() if reused_only else self)

    def words(self) -> frozenset[str]:
        return _relation_words(self)

    def __len__(self) -> int:
        return len(self._rels)

    def __iter__(self):
        return iter(sorted(self._rels))


def write_vocab_jsonl(facts: Iterable[str], path) -> int:
    """The store's relation vocabulary as `cubelang run --knowledge` input:
    one `{"key": rel, "text": rel, "source": "store", "n": facts}` per distinct
    relation (`n` feeds the reuse guard of hop 0's paraphrase tier). Returns
    the number of relations written."""
    import json
    rels = StoreRelations(facts)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rels:
            fh.write(json.dumps({"key": r, "text": r, "source": "store", "n": rels._n.get(r, 0)}) + "\n")
    return len(rels)


class VMRelations:
    """KnownRelations answered BY THE VM: one `QUERY` per distinct relation.

    `knowledge` is the jsonl `write_vocab_jsonl` produced; `program` is
    bridges/programs/plan_verify.cube, whose `solve(mention)` is
    examples/ground_min.cube's three-valued branch over `query mention` --
    it RETURNS THE VERDICT (0 unknown / 1 known / 2 ambiguous) as the VM's own
    integer. A bare chunk array (an older program) is also accepted:
    membership is then `len > 0`. Memoized per normalized relation (exp_r6
    --real --vm: 487 distinct QUERY calls over 763 questions, 0 disagreements
    with the host reference)."""

    def __init__(self, knowledge: str, program: str | None = None, exe: str | None = None,
                 run=None, session=None) -> None:
        """`session`: a `cubelang_client.CubelangSession` -- the RESIDENT
        transport (one process, `knowledge_path` on the request, store cached
        by mtime in the process). Without it, `run` (default `run_program`,
        one `cubelang run --knowledge` subprocess per distinct relation) is
        used; that is the transport whose 181 spawns cost more than they saved
        on the vp harvest, kept for the equivalence check and for a caller
        without protobuf."""
        import pathlib
        self.knowledge = str(knowledge)
        self.program = str(program or pathlib.Path(__file__).resolve().parents[1]
                           / "bridges" / "programs" / "plan_verify.cube")
        self.exe = exe
        self.session = session
        self._source: str | None = None
        if run is None and session is None:
            from ..bridges.cubelang_client import run_program
            run = run_program
        self._run = run
        self._memo: dict[str, bool] = {}
        self.n_calls = 0
        # the fuzzy tier reads the SAME vocabulary the VM was given, host-side
        self._vocab: list[str] | None = None

    def _vocabulary(self, reused_only: bool = False) -> list[str]:
        if self._vocab is None:
            import json, pathlib
            keys, n = set(), {}
            for line in pathlib.Path(self.knowledge).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if "key" in rec:
                        k = normalize(rec["key"]); keys.add(k)
                        n[k] = int(rec.get("n", 2))      # no count on file: counts as reused
            self._vocab = sorted(keys)
            self._reused = sorted(k for k in keys if n[k] >= 2)
        return self._reused if reused_only else self._vocab

    def match(self, rel: str, *, reused_only: bool = False) -> str | None:
        """Exact tier is the VM's QUERY; the paraphrase tier is the walk's
        relation_matches over the same vocabulary file, host-side, and the
        verdict labels it as such. `reused_only`: hop 0's reuse guard (see
        StoreRelations.reused)."""
        if rel in self:
            return normalize(rel)
        return _fuzzy_match(rel, self._vocabulary(reused_only))

    def words(self) -> frozenset[str]:
        return _relation_words(self._vocabulary())

    def __contains__(self, rel: str) -> bool:
        key = normalize(rel)
        if key not in self._memo:
            self.n_calls += 1
            if self.session is not None:
                if self._source is None:
                    import pathlib
                    self._source = pathlib.Path(self.program).read_text(encoding="utf-8")
                out = self.session.run(self._source, fn="solve", args=[key],
                                       knowledge_path=self.knowledge)
            else:
                out = self._run(self.program, fn="solve", args=[key], exe=self.exe,
                                knowledge=self.knowledge)
            res = out.get("result")
            if isinstance(res, list):                 # chunk array: count it here
                self._memo[key] = len(res) > 0
            else:                                     # the VM's verdict: 0 / 1 / 2
                self._memo[key] = int(res or 0) > 0
        return self._memo[key]


@dataclass(frozen=True)
class PlanVerdict:
    ok: bool
    covers: bool                       # the plan reconstructs the question
    unknown_relations: list[str] = field(default_factory=list)
    tail_relation: str | None = None   # the known relation prefix of the tail, if any
    reason: str | None = None          # why not ok
    # hop>=1 relations accepted only by the walk's paraphrase rule (relation_matches),
    # as (asked, matched) pairs. ok may be True with these present; they are never
    # hidden inside "known". Empty when every relation was an exact hit.
    paraphrased: list[tuple[str, str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def canonical_body(plan: QuestionPlan) -> str:
    """'R_n of the R_{n-1} ... of the tail' — the chain frame's body, normalized."""
    rels = [r for r in plan.relations[1:] if r]
    return normalize(JOINT.join(list(reversed(rels)) + [plan.tail]))


def tail_relation(tail: str, known: KnownRelations) -> str | None:
    """The LONGEST prefix of the tail, cut at an ' of ', that is a known relation.
    Longest first: 'country of citizenship of X' must yield 'country of
    citizenship', not 'country'."""
    parts = tail.split(" of ")
    for i in range(len(parts) - 1, 0, -1):
        cand = " of ".join(parts[:i])
        if cand in known:
            return cand
    return None


def tail_match(tail: str, known: KnownRelations) -> tuple[str, str] | None:
    """The paraphrase tier of `tail_relation` (2026-09-11, lever 1): the longest
    ' of '-prefix of the tail that `known.match` accepts -- (the plan's wording,
    the store's relation). Mirrors the walk's hop 0 since `planner.accepts` grew
    its paraphrase tier: subject exact, relation by `relation_matches`. Exact
    hits are `tail_relation`'s; call this only when that returned None."""
    if not hasattr(known, "match"):
        return None
    parts = tail.split(" of ")
    for i in range(len(parts) - 1, 0, -1):
        cand = " of ".join(parts[:i])
        try:
            m = known.match(cand, reused_only=True)       # the reuse guard (StoreRelations.reused)
        except TypeError:                                  # a KnownRelations without the guard
            m = known.match(cand)
        if m is not None:
            return cand, m
    return None


# ---- the typed answer class (2026-09-12, after exp_r11 lev6/lev6b) --------------------------------
# A question says what KIND of thing it wants: "in what year", "on what day, month, and
# year", "when" want a date; "how many", "how much" want a number; "who" wants a name.
# Two things follow. (1) The words that state the kind ('year', 'day', 'month', 'date';
# 'many', 'much', 'number') are the ask's frame, not a dropped hop: the residual rule had
# read a leftover 'year' as a hop whenever the store's vocabulary held a 'year'-bearing
# relation -- which changed with what the loop had learned (two birth dates gained in
# lev6b only because a refused fetch no longer brought that relation in). (2) A verified
# chain whose answer is the wrong kind -- a spouse's name for "in what year did X marry Y",
# a date for "who" -- answers a different question than asked, and is refused at the one
# verified=True site with the kinds named. Deterministic shapes only: a date is ISO or a
# year, a number is digits; a name is what is neither. The VM stays the truth gate; this
# is the disposer reading the question's ask, as it reads its relations.
_ASK_DATE = __import__("re").compile(
    r"\b(when|(what|which)\s+(year|date|day|month|decade|century)|(in|on|during)\s+(what|which)\s+(year|date|day|month)|"
    r"day,?\s+month,?\s+(and\s+)?year|month\s+(and|,)\s+year|(what|which)\s+day\s+of)\b")
_ASK_NUMBER = __import__("re").compile(r"\b(how\s+many|how\s+much|what\s+(number|percentage|amount|population|count))\b")
_ASK_NAME = __import__("re").compile(r"^\s*(who|whom|whose)\b")
_ASK_WORDS = {"date": frozenset("year date day month decade century when".split()),
              "number": frozenset("many much number percentage amount count".split()),
              "name": frozenset()}
_ISO_DATE = __import__("re").compile(r"^[+-]?\d{4}-\d{2}(-\d{2})?$")
_YEAR_OR_COUNT = __import__("re").compile(r"^[+-]?\d{3,4}$")
_NUMBER_VALUE = __import__("re").compile(r"^[+-]?\d+([.,]\d+)?$")


def ask_type(question: str) -> str | None:
    """'date' | 'number' | 'name' | None -- the kind of answer the question asks for."""
    q = question.lower()
    if _ASK_DATE.search(q):
        return "date"
    if _ASK_NUMBER.search(q):
        return "number"
    if _ASK_NAME.search(q):
        return "name"
    return None


def value_kinds(value: str) -> frozenset[str]:
    """The kinds a stored value can be, by its shape: an ISO date is a date; a bare 3-4
    digit integer is a year OR a count (both); other digits are a number; the rest a name."""
    v = (value or "").strip()
    if _ISO_DATE.match(v):
        return frozenset({"date"})
    if _YEAR_OR_COUNT.match(v):
        return frozenset({"date", "number"})
    if _NUMBER_VALUE.match(v):
        return frozenset({"number"})
    return frozenset({"name"})


def answer_type_mismatch(question: str, value: str) -> tuple[str, str] | None:
    """(asked, got) when the question's ask is a kind the value cannot be; None otherwise."""
    asked = ask_type(question)
    if asked is None:
        return None
    kinds = value_kinds(value)
    return None if asked in kinds else (asked, "/".join(sorted(kinds)))


def covers(question: str, plan: QuestionPlan, known: KnownRelations | None = None,
           aliases: dict[str, list[str]] | None = None) -> bool:
    """Does the plan account for the WHOLE question?

    v2 (2026-09-11, after exp_r7). v1 required the grammar's canonical body
    "R_n of the ... of the tail" to be a SUBSTRING of the question -- which
    means it could only accept plans inside the grammar's " of the " basin.
    On the 92 misparsed questions it refused 13 emitter plans that had the
    gold hop count purely because the question joins its hops with
    "contained within the" / "received by the"; and its 1-hop rule ("relation
    and entity both present") let 4 plans through that had DROPPED the outer
    hop ("source that describes the country of X" planned as 1-hop "country
    of X"), each of which the VM then verified into a wrong answer.

    v2 is joint-agnostic and hop-complete:
      1. the relations occur in the question IN ORDER (answer side first),
         then the seed entity, with anything at all between them;
      2. what is left of the question after removing them and the frame/joint
         words must contain NO word that any relation in the store is made of.
         "contained within" is a joint; "source describes" is a dropped hop.
    (2) needs the vocabulary (`known.words()`); without it, only (1) runs, and
    the 1-hop dropped-hop hole stays open -- pass `known`.

    v3 (2026-09-11, lever 2, after exp_r9 and exp_r7 gen 2):
      * ORDER. v2 read the question answer-side first only. exp_r9's possessive
        form ("E's P1 -- what is its P2?") states the entity first and the hops
        inner-first, and all 180 plans of the question were refused. v3 accepts
        either reading: answer-side first then the entity, or the entity first
        then the hops in walk order. Both are the same chain; the residual rule
        is what keeps a dropped hop out, and it runs on both.
      * WORDING. v2 looked for the plan's relation string verbatim. When the
        disposer accepted a relation as a PARAPHRASE of a store relation, the
        question may carry the store's wording ("languages spoken, written or
        signed by X" planned as `languages spoken written signed`): 6 of gen 2's
        21 arm-C coverage refusals. v3 looks for either wording of a paraphrased
        relation; the residual rule is unchanged, so a dropped hop still fails."""
    q = normalize(question)
    rels = [r for r in plan.relations[1:] if r]                      # walk order, inner-most first
    # split the tail into its relation and the seed entity: the LONGEST known
    # relation prefix when a vocabulary is given ('country of citizenship' before
    # 'country'), then the paraphrase tier's split, else the last ' of ' (a
    # relation with ' of ' in it is commoner than an entity with one)
    tr = tail_relation(plan.tail, known) if known is not None else None
    tail_alt = None
    if tr is None and known is not None:
        tm = tail_match(plan.tail, known)
        if tm:
            tr, tail_alt = tm
    if tr is not None:
        rel1, ent = tr, plan.tail[len(tr):].strip()
        ent = ent[3:] if ent.startswith("of ") else ent
    elif " of " in plan.tail:
        rel1, ent = plan.tail.rsplit(" of ", 1)
    else:
        rel1, ent = plan.tail, ""
    ent = normalize(ent)

    def wordings(r: str, alt: str | None = None) -> list[str]:
        out = [normalize(r)]
        if alt is None and known is not None and hasattr(known, "match") and r not in known:
            alt = known.match(r)
        if alt and normalize(alt) not in out:
            out.append(normalize(alt))
        # a relation the HOST rewrote into the store's wording (learn.py's alias step,
        # lever 4) is asked about in the question under the plan's original words
        for w in (aliases or {}).get(normalize(r), ()):
            if normalize(w) not in out:
                out.append(normalize(w))
        return out
    inner_first = [wordings(rel1, tail_alt)] + [wordings(r) for r in rels]      # walk order
    answer_first = list(reversed(inner_first))
    rw = known.words() if (known is not None and hasattr(known, "words")) else None
    kind = ask_type(question)
    if rw is not None and kind is not None:
        rw = rw - _ASK_WORDS[kind]                     # 'year' in "in what year" is the ask, not a hop
    # "Which <class> is the R of ...?" -- the class noun duplicates the answer type
    # and would otherwise be consumed as the first relation. Try with the frame
    # stripped first; "Which country is X in?" (where the class IS the relation)
    # then succeeds on the unstripped retry.
    stripped = _WHICH_CLASS.sub("", q, count=1)
    for text in ((stripped, q) if stripped != q else (q,)):
        if _covers_text(text, answer_first, ent, rw) or _covers_text(text, inner_first, ent, rw, entity_first=True):
            return True
    return False


# "Which <class> is/was/includes/contains the ..." -- the class noun duplicates the answer type. exp_r8
# (2026-09-11): "Which LIST includes the list that includes E" left the class noun 'list' in the residual,
# where it read as a dropped 'list' hop, and the disposer refused the one correct plan.
_WHICH_CLASS = __import__("re").compile(r"^which\s+[a-z0-9]+(?:\s+[a-z0-9]+){0,2}\s+(?:is|was|are|were|includes|contains|has)\s+")


def _find_any(q: str, alts: list[str], pos: int) -> tuple[int, int] | None:
    """The earliest occurrence at/after `pos` of any wording; the longer wording
    wins a tie. None when no wording occurs."""
    best = None
    for w in alts:
        i = q.find(w, pos)
        if i >= 0 and (best is None or i < best[0] or (i == best[0] and len(w) > best[1] - best[0])):
            best = (i, i + len(w))
    return best


def _covers_text(q: str, order: list[list[str]], ent: str, rw, entity_first: bool = False) -> bool:
    """`order`: one list of accepted wordings per relation, in the reading's order.
    Answer-side first (default): relations in order, then the entity after them
    (a 1-hop inversion frame may state the entity first: "What is erik bergvall a
    participant of?"). Entity first: the entity, then the relations in order."""
    pos, spans = 0, []
    if entity_first:
        if not ent:
            return False
        j = q.find(ent)
        if j < 0:
            return False
        spans.append((j, j + len(ent))); pos = j + len(ent)
    for alts in order:
        hit = _find_any(q, alts, pos)
        if hit is None:
            return False
        spans.append(hit); pos = hit[1]
    if ent and not entity_first:
        j = q.rfind(ent)
        if j >= pos:
            spans.append((j, j + len(ent)))
        elif len(order) == 1:
            j = q.find(ent)
            a, b = spans[0]
            if j < 0 or (j < b and j + len(ent) > a):
                return False
            spans.append((j, j + len(ent))); spans.sort()
        else:
            return False
    if rw is None:
        return True
    residual, last = [], 0
    for a, b in spans:
        residual.append(q[last:a]); last = b
    residual.append(q[last:])
    leftover = [w for w in " ".join(residual).split() if w not in _FRAME_AND_JOINT]
    # a number left over is a CONSTRAINT the plan did not bind ('as of 2022', 'in 1977'):
    # the store cannot check it, so a plan that ignores it would answer a different
    # question (exp_r11, 2026-09-11: 'population of Mersin Province' spoke one census
    # for 'as of 2022')
    if any(w.isdigit() for w in leftover):
        return False
    return not any(w in rw for w in leftover)


def verify_plan(question: str, plan: QuestionPlan, known: KnownRelations,
                aliases: dict[str, list[str]] | None = None) -> PlanVerdict:
    """Disposes of a plan with EXACTLY the walk's tolerance, no more, no less:
    every hop has an exact tier and a `relation_matches` tier, the walk's own
    acceptance (hop 0 grew its paraphrase tier 2026-09-11, lever 1; before
    that it was string equality on the whole tail and the disposer was exact
    there too). Paraphrase hits are recorded in `paraphrased`. So a
    refusal is a proof that no fact in the store could have been accepted at
    that hop -- the walk would have failed -- and a verifying chain is never
    refused (exp_m3 lookup_vp, 2026-09-11: the exact-only draft refused 2 of
    568 verified chains on 'office held by THE head of government' vs the
    store's 'office held by head of government'; this rule refuses 0)."""
    cov = covers(question, plan, known, aliases)
    unknown: list[str] = []
    paraphrased: list[tuple[str, str]] = []
    for r in plan.relations[1:]:
        if not r:
            continue
        if r in known:
            continue
        m = known.match(r) if hasattr(known, "match") else None
        if m is None:
            unknown.append(r)
        else:
            paraphrased.append((r, m))
    tr = tail_relation(plan.tail, known)
    if tr is None:
        tm = tail_match(plan.tail, known)          # hop 0's paraphrase tier (lever 1)
        if tm is not None:
            tr = tm[0]; paraphrased.append(tm)
        else:
            unknown.append(plan.tail.rsplit(" of ", 1)[0] if " of " in plan.tail else plan.tail)
    if not cov:
        reason = "plan_does_not_cover_question"
    elif unknown:
        reason = "unknown_relation"
    else:
        reason = None
    return PlanVerdict(ok=cov and not unknown, covers=cov, unknown_relations=unknown,
                       tail_relation=tr, reason=reason, paraphrased=paraphrased)


def segment(body: str, known: KnownRelations, max_hop: int = 4,
            joint_in_entity: bool = True) -> list[QuestionPlan]:
    """Every way to read `body` as R_n of the ... of the (R_1 of E) with every R known.

    Deterministic, exhaustive over the ' of the ' joints, bounded by max_hop.
    Returns plans outer-to-inner like `parse_question` (relations[0] is None).
    An empty list means no reading is answerable from the store; more than one
    means the split is ambiguous and the disposer must not guess.

    `joint_in_entity=True` (default, exhaustive) also admits readings whose
    ENTITY swallows a ' of the ' ("A Friend of the Family"). Those readings
    are real — it is the one misparse exp_r6 could repair — but they make
    nearly every multi-hop body ambiguous, because "capital of the country of
    X" can always also be read as 1-hop 'capital' of the entity "the country
    of X". `joint_in_entity=False` drops them: an honest, narrower reading
    set, not a smarter one."""
    segs = body.split(JOINT)
    n = len(segs)
    out: list[QuestionPlan] = []
    # choose which joints are chain joints: a subset of the n-1 joint positions
    for mask in range(1 << (n - 1)):
        rels: list[str] = []
        cur = [segs[0]]
        ok = True
        for j in range(n - 1):
            if mask >> j & 1:
                r = JOINT.join(cur).strip()
                if r not in known:
                    ok = False
                    break
                rels.append(r)
                cur = [segs[j + 1]]
            else:
                cur.append(segs[j + 1])
        if not ok or len(rels) + 1 > max_hop:
            continue
        tail = JOINT.join(cur).strip()
        if tail_relation(tail, known) is None:
            continue
        if not joint_in_entity:
            tr = tail_relation(tail, known)
            if JOINT in tail[len(tr):]:
                continue
        out.append(QuestionPlan(relations=[None] + list(reversed(rels)), tail=tail, n_hop=len(rels) + 1))
    return out
