"""skills -- the skill library: composition rules, mined from episodes, gated, and applied through the VM.

Wired: WIRED -- the ask loop answers "How is B related to A?" with `relate` (standin/ask.py), and the
sleep cycle mines and gates the library every night (`sleep.skills_night`).

Level 3 of self-evolution (docs/research/2026-09-24-sleep-cycle.md). A SKILL is something the host learned
to DO, kept outside the weights, on a ledger. The first kind is a composition rule:

    the SISTER of the FATHER of x  is the AUNT of x          Rule("father", "sister", "aunt")

(walk order: the first hop's relation first). A fact store cannot hold that. CLUTRR's own question --
"how is B related to A?" -- asks for the relation a chain of facts COMPOSES to, and until this module the
engine could only walk the chain (WO-2.6).

Where a rule comes from. Episodes: a chain of relations and the relation it composes to, as stated by
something other than the loop -- a bench's gold, or an asker. Never the loop's own answer: a rule confirmed
by the answers it produced would be the loop judging itself (invariant 6).

How a rule gets in (the gate; Nick, 2026-09-24: zero counterexamples). A candidate premise (X, Y) is read
off every episode that splits into two spans the library already composes -- a single relation composes to
itself -- so a rule seen only INSIDE longer chains is learned once its neighbours are (closure). It is
adopted only if
  * every instance of the premise states the SAME conclusion,
  * it has at least `min_support` instances from distinct sources (an asker counts once), and
  * with it added, EVERY past episode still derives its own stated relation or nothing -- none derives a
    different relation, none derives two.
An episode the library derives wrongly (a new one, at night) retires rules its wrong derivation used,
weakest first, until no episode on record is derived wrongly. Retired, never deleted: the ledger keeps
the line and the episode that retired it, and a retired rule is not adopted again.

How a rule is used. `derive` composes a chain over EVERY bracketing (CYK); the chain composes only if the
bracketings that complete all agree. Two relations for one chain is a split: refused, and evidence against
a rule. `relate` does this for every path between two people and speaks only if the paths agree. A path
the library cannot compose does not block the others -- composition is associative, so a path with no rule
is missing knowledge, not contrary evidence. A path that composes but fails the VM does block.

Each spoken derivation goes through the VM with its facts: the facts and the rule steps are bound in one
chain program, two bindings to a frame, and each is recovered and checked like a hop, by
`pipeline._check_chain` -- the one place a chain is certified. What the VM certifies here is what it
certifies for facts, binding fidelity. A rule's TRUTH comes from the gate (every past episode re-derived),
not from the VM.

Terms at different grain (2026-09-24, real families). A world's relation words can nest: Wikidata's
"paternal grandfather" is a "grandfather", "father's brother" an "uncle". With a term hierarchy
(`Library(entails=)`: term -> the terms it entails, itself included) two terms CONTRADICT only when neither
entails the other. A rule's conclusion is one of the terms its instances actually stated -- the finest that
enough of them confirm (their term entails it) and none contradicts -- so an episode saying only "uncle"
never blocks (father, brother) -> uncle, and a rule never concludes a vague word nobody stated. A chain or
a set of paths whose conclusions are all comparable speaks the finest; incomparable ones are a split.
Without a hierarchy every term entails only itself, which is the CLUTRR case.

Kinds. `rule` (this module) and `helper` (CubeLang sub-programs mined from the emitter's verified programs,
`helpers.py`) share the ledger; each line carries its `kind`, and the Library keeps helpers in `helpers`.

Advantage over a model that has the kinship algebra in its weights: every rule is explicit, carries the
episodes that support it, can be retired by one counterexample, and composes to any depth -- a chain of
ten relations needs no ten-relation training example, only rules that each passed the gate.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

from ..core.protocols import Wiring
from .planner import Triple, normalize

__wiring__ = Wiring.WIRED

KINDS = ("rule",)                  # "helper" (a CubeLang sub-program) is the next kind; not built
#: tau by the number of bindings a frame holds (exp_r11's thresholds; standin/ask.TAU_VM). `relate`
#: certifies two bindings to a frame, so a derivation's threshold does not fall with its length (exp_r28).
TAU_BUNDLE = {1: 1.0, 2: 0.4736328125}
MAX_HOPS = 10                      # the longest chain `relate` follows between two people
#: Wikidata's family properties (P22 P25 P40 P3373 P26), by the labels its facts carry
FAMILY = ("father", "mother", "child", "sibling", "spouse")
#: the word for a family relation held by a person of known sex: the host's LEXICON (English: a son is a male
#: child), not a rule the library learned. A person of unknown sex keeps the property's own word.
GENDERED = {("child", "male"): "son", ("child", "female"): "daughter",
            ("sibling", "male"): "brother", ("sibling", "female"): "sister",
            ("spouse", "male"): "husband", ("spouse", "female"): "wife"}
PARENT = {"male": "father", "female": "mother"}


def family_facts(x: str, relation: str, y: str, sex: dict) -> list[str]:
    """``y is the <relation> of x`` as facts the library's rules speak of, both ways: the relation named by
    y's sex (a male child is a son), and the edge back, which Wikidata itself declares -- father and mother
    are the inverse of child, sibling and spouse are symmetric (P1696) -- named by x's sex. `sex` maps a
    person to "male" / "female"; an unknown sex keeps the property's word, and an unknown parent's sex
    drops the edge back (no word for it is a rule's)."""
    r = rel(relation)
    out = [f"{y} is the {GENDERED.get((r, sex.get(y)), r)} of {x}"]
    if r in ("father", "mother"):
        out.append(f"{x} is the {GENDERED.get(('child', sex.get(x)), 'child')} of {y}")
    elif r == "child":
        if sex.get(x) in PARENT:
            out.append(f"{x} is the {PARENT[sex[x]]} of {y}")
    elif r in ("sibling", "spouse"):
        out.append(f"{x} is the {GENDERED.get((r, sex.get(x)), r)} of {y}")
    return out
MAX_PATHS = 16                     # more paths than this is refused, never sampled


def rel(s: str | None) -> str:
    return normalize(s or "")


@dataclass(frozen=True)
class Rule:
    """``the SECOND of the FIRST of x is the CONCLUSION of x`` (walk order)."""
    first: str
    second: str
    conclusion: str

    @property
    def premise(self) -> tuple[str, str]:
        return (self.first, self.second)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.first, self.second, self.conclusion)

    def label(self) -> str:
        """The rule step's role in a chain program."""
        return f"{self.first} then {self.second}"

    def __str__(self) -> str:
        return f"the {self.second} of the {self.first} is the {self.conclusion}"


@dataclass
class Episode:
    """A chain of relations (walk order) and the relation it composes to, stated by gold or an asker.
    `source` names who stated it when that matters: episodes from one source count once toward a rule's
    support (one asker cannot put a rule in alone); None counts every episode (a bench's gold)."""
    id: str
    relations: list[str]
    conclusion: str
    source: str | None = None


@dataclass
class Derivation:
    status: str                                        # derived | split | no_rule | empty
    conclusion: str | None = None
    steps: list[Rule] = field(default_factory=list)    # one derivation of `conclusion`, in application order
    conclusions: dict[str, list[Rule]] = field(default_factory=dict)   # every conclusion found -> one derivation


# ── the library ────────────────────────────────────────────────────────────────────────────────────────
class Library:
    """The adopted rules, replayed from an append-only ledger (``skills.jsonl``). Without a path it lives
    in memory. ``with_rule`` is a trial copy for the gate and never writes."""

    def __init__(self, path: str | pathlib.Path | None = None, entails: dict | None = None) -> None:
        self.path = pathlib.Path(path) if path else None
        if entails is None and self.path is not None and self.path.with_name("skills_taxonomy.json").exists():
            entails = json.loads(self.path.with_name("skills_taxonomy.json").read_text(encoding="utf-8"))
        #: term -> the terms it entails (itself included); a term not listed entails only itself
        self.entails: dict[str, frozenset] = {rel(t): frozenset({rel(t)} | {rel(s) for s in sups})
                                              for t, sups in (entails or {}).items()}
        self.rules: dict[tuple[str, str], dict] = {}           # premise -> {rule, support, evidence, night}
        self.retired: dict[tuple[str, str, str], dict] = {}    # rule key -> {reason, evidence, night}
        self.discounted: dict[str, dict] = {}                   # episode id -> {reason, night}: the host's word
        self.helpers: dict[str, dict] = {}                      # CubeLang helpers (reasoning/helpers.py): name -> line
        self.retired_helpers: dict[str, dict] = {}
        self.inverses: dict[tuple[str, str], dict] = {}         # (relation, sex of the one it is TO) -> line
        self.retired_inverses: dict[tuple[str, str, str], dict] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def __len__(self) -> int:
        return len(self.rules)

    def __contains__(self, rule: Rule) -> bool:
        e = self.rules.get(rule.premise)
        return e is not None and e["rule"] == rule

    def conclusion(self, first: str, second: str) -> str | None:
        e = self.rules.get((first, second))
        return e["rule"].conclusion if e else None

    def support(self, rule: Rule) -> int:
        e = self.rules.get(rule.premise)
        return int(e.get("support") or 0) if e else 0

    def _apply(self, r: dict) -> None:
        if r.get("kind") == "inverse":
            key = (r["relation"], r["sex"])
            if r.get("action") == "adopt" and (*key, r["inverse"]) not in self.retired_inverses:
                self.inverses[key] = r
            elif r.get("action") == "retire":
                self.retired_inverses[(*key, r["inverse"])] = r
                if (self.inverses.get(key) or {}).get("inverse") == r["inverse"]:
                    del self.inverses[key]
            return
        if r.get("kind") == "helper":
            if r.get("action") == "adopt" and r["name"] not in self.retired_helpers:
                self.helpers[r["name"]] = r
            elif r.get("action") == "retire":
                self.retired_helpers[r["name"]] = r
                self.helpers.pop(r["name"], None)
            return
        if r.get("action") == "discount" and r.get("kind") == "episode":
            self.discounted[r["episode"]] = {"reason": r.get("reason"), "night": r.get("night")}
            return
        if r.get("kind", "rule") != "rule":
            return                                             # a kind this build does not read
        rule = Rule(r["first"], r["second"], r["conclusion"])
        if r.get("action") == "adopt":
            if rule.key not in self.retired:
                self.rules[rule.premise] = {"rule": rule, "support": r.get("support"),
                                            "evidence": list(r.get("evidence") or []), "night": r.get("night")}
        elif r.get("action") == "retire":
            self.retired[rule.key] = {"reason": r.get("reason"), "evidence": list(r.get("evidence") or []),
                                      "night": r.get("night")}
            if rule in self:
                del self.rules[rule.premise]

    def _write(self, action: str, rule: Rule, night: str, **detail) -> dict:
        line = {"action": action, "kind": "rule", "first": rule.first, "second": rule.second,
                "conclusion": rule.conclusion, "night": night,
                "ts": _dt.datetime.now().isoformat(timespec="seconds"), **detail}
        self._apply(line)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return line

    def adopt(self, rule: Rule, support: int, evidence: list[str], night: str, **detail) -> dict:
        return self._write("adopt", rule, night, support=int(support), evidence=list(evidence)[:5], **detail)

    def retire(self, rule: Rule, reason: str, evidence: list[str], night: str, **detail) -> dict:
        return self._write("retire", rule, night, reason=reason, evidence=list(evidence)[:5], **detail)

    def _append(self, line: dict) -> dict:
        self._apply(line)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return line

    def adopt_helper(self, shape, support: int, night: str, **detail) -> dict:
        """A CubeLang helper (`helpers.Shape`) the gate adopted: its source is on the ledger, so a program
        that calls it can always be given its definition (`helpers.with_helpers`)."""
        if shape.name in self.retired_helpers:
            return {}
        return self._append({"action": "adopt", "kind": "helper", "name": shape.name, "source": shape.source(),
                             "op1": shape.op1, "op2": shape.op2, "inner_left": shape.inner_left,
                             "consts": [list(c) for c in shape.consts], "support": int(support), "night": night,
                             "ts": _dt.datetime.now().isoformat(timespec="seconds"), **detail})

    def inverse_of(self, relation: str, sex: str | None) -> str | None:
        """What A is to B, when B is the `relation` of A and A's sex is `sex`: the adopted inverse rule for
        exactly that relation, or None. Not through the terms it entails: the first cut did, and on Wikidata
        (exp_r35) the taxonomy's own gaps turned a grand-nephew into an 'ancestor' -- the hierarchy decides
        what does not CONTRADICT, it is not good enough to decide what is TRUE."""
        if not sex:
            return None
        hit = self.inverses.get((rel(relation), sex))
        return hit["inverse"] if hit else None

    def adopt_inverse(self, relation: str, sex: str, inverse: str, support: int, evidence: list[str], night: str,
                      **detail) -> dict:
        return self._append({"action": "adopt", "kind": "inverse", "relation": rel(relation), "sex": sex,
                             "inverse": rel(inverse), "support": int(support), "evidence": list(evidence)[:5],
                             "night": night, "ts": _dt.datetime.now().isoformat(timespec="seconds"), **detail})

    def retire_inverse(self, relation: str, sex: str, inverse: str, reason: str, night: str) -> dict:
        return self._append({"action": "retire", "kind": "inverse", "relation": rel(relation), "sex": sex,
                             "inverse": rel(inverse), "reason": reason, "night": night,
                             "ts": _dt.datetime.now().isoformat(timespec="seconds")})

    def retire_helper(self, name: str, reason: str, night: str) -> dict:
        return self._append({"action": "retire", "kind": "helper", "name": name, "reason": reason, "night": night,
                             "ts": _dt.datetime.now().isoformat(timespec="seconds")})

    def discount(self, episode_id: str, reason: str, night: str, **detail) -> dict:
        """The HOST says an episode is not evidence (a label error, most often). It stays on the record and
        on the ledger with the reason; the gate stops reading it."""
        line = {"action": "discount", "kind": "episode", "episode": episode_id, "reason": reason, "night": night,
                "ts": _dt.datetime.now().isoformat(timespec="seconds"), **detail}
        self._apply(line)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return line

    def sup(self, term: str) -> frozenset:
        return self.entails.get(term) or frozenset({term})

    def comparable(self, a: str, b: str) -> bool:
        """Neither contradicts the other: one entails the other (equal terms included)."""
        return a == b or a in self.sup(b) or b in self.sup(a)

    def finest(self, terms) -> str | None:
        """The one term every other entails, when the terms are pairwise comparable; else None."""
        terms = list(dict.fromkeys(terms))
        if any(not self.comparable(a, b) for i, a in enumerate(terms) for b in terms[i + 1:]):
            return None
        return max(terms, key=lambda t: len(self.sup(t))) if terms else None

    def with_rule(self, rule: Rule) -> "Library":
        trial = Library()
        trial.entails = self.entails
        trial.rules = dict(self.rules)
        trial.retired = self.retired
        trial.discounted = self.discounted
        trial.inverses = self.inverses
        trial.rules[rule.premise] = {"rule": rule, "support": 0, "evidence": [], "night": None}
        return trial

    def table(self) -> list[dict]:
        return [{"rule": str(e["rule"]), "support": e.get("support"), "night": e.get("night")}
                for _p, e in sorted(self.rules.items())]


# ── composing a chain ──────────────────────────────────────────────────────────────────────────────────
def _cells(rels: list[str], lib: Library) -> tuple[list[list[dict]], bool]:
    """CYK over the chain: cell[i][j] maps every relation span i..j composes to, to one derivation of it.
    The flag says some span composed to two relations."""
    n = len(rels)
    cell: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]
    for i, r in enumerate(rels):
        cell[i][i] = {r: ()}
    split = False
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            out: dict = {}
            for k in range(i, j):
                for x, sx in cell[i][k].items():
                    for y, sy in cell[k + 1][j].items():
                        c = lib.conclusion(x, y)
                        if c is not None and c not in out:
                            out[c] = sx + sy + (Rule(x, y, c),)
            cell[i][j] = out
            split = split or (len(out) > 1 and lib.finest(out) is None)
    return cell, split


def derive(relations: list[str], lib: Library) -> Derivation:
    """The relation a chain composes to, over every bracketing. `split` when bracketings disagree anywhere
    in the chain (a span that composes to two relations is a bad rule somewhere, and nothing built on it is
    spoken); `no_rule` when no bracketing completes."""
    rels = [rel(r) for r in relations]
    if not rels:
        return Derivation("empty")
    cell, split = _cells(rels, lib)
    top = cell[0][len(rels) - 1]
    if not top:
        return Derivation("no_rule")
    found = {c: list(s) for c, s in top.items()}
    best = lib.finest(found)
    if split or best is None:
        return Derivation("split", conclusions=found)
    return Derivation("derived", best, found[best], found)


def _labels(rels: list[str], lib: Library) -> set[str]:
    """Every relation any span of the chain composes to: a rule (X, Y) can only change this chain's
    derivation when both X and Y are among them."""
    cell, _ = _cells(rels, lib)
    return {c for row in cell for d in row for c in d}


def _conclusion(said: Counter, lib: Library, min_support: int = 2) -> str | None:
    """The conclusion a premise's instances support: one of the terms they STATED, contradicted by none of
    them, the finest that the most of them confirm. None when two stated terms contradict each other."""
    terms = list(said)
    if any(not lib.comparable(a, b) for i, a in enumerate(terms) for b in terms[i + 1:]):
        return None
    confirm = {c: sum(n for t, n in said.items() if c in lib.sup(t)) for c in terms}
    return max(terms, key=lambda c: (confirm[c] >= min_support, len(lib.sup(c)), confirm[c], c))


def _premises(rels: list[str], lib: Library) -> set[tuple[str, str]]:
    """The (X, Y) premises one episode shows: every split into two spans the library composes."""
    out = set()
    for p in range(1, len(rels)):
        left, right = derive(rels[:p], lib), derive(rels[p:], lib)
        if left.status == "derived" and right.status == "derived":
            out.add((left.conclusion, right.conclusion))
    return out


def _wrong(d: Derivation, stated: str, lib: Library | None = None) -> bool:
    """A split, or a relation the stated one contradicts (with a hierarchy: neither entails the other)."""
    if d.status == "split":
        return True
    if d.status != "derived":
        return False
    return not lib.comparable(d.conclusion, stated) if lib is not None else d.conclusion != stated


# ── the gate ───────────────────────────────────────────────────────────────────────────────────────────
def reverify(episodes: list[Episode], lib: Library) -> list[tuple[Episode, Derivation]]:
    """Every episode the library derives wrongly: a different relation from the one it states, or two."""
    out = []
    for e in episodes:
        d = derive(e.relations, lib)
        if _wrong(d, rel(e.conclusion), lib):
            out.append((e, d))
    return out


def _retire_contradicted(eps: list[Episode], lib: Library, night: str, git: str | None) -> list[dict]:
    """Until no episode on record is derived wrongly: from each wrong derivation retire the weakest rule it
    used (least support; ties all go). Weakest first, because a rule that dozens of episodes state is less
    likely to be the one at fault than one that two do -- and the loop repeats until the record is clean."""
    retired = []
    while True:
        progress = False
        for e, d in reverify(eps, lib):
            stated = rel(e.conclusion)
            used = {s for c, steps in d.conclusions.items() if not lib.comparable(c, stated) for s in steps} or \
                   {s for steps in d.conclusions.values() for s in steps}
            used = [s for s in used if s in lib]
            if not used:
                continue
            low = min(lib.support(s) for s in used)
            said = d.conclusion if d.status == "derived" else sorted(d.conclusions)
            for s in used:
                if lib.support(s) == low and s in lib:
                    retired.append(lib.retire(s, reason=f"episode {e.id} states {stated}; the library derived {said}",
                                              evidence=[e.id], night=night, git=git))
                    progress = True
        if not progress:
            return retired


def mine(episodes: list[Episode], lib: Library, night: str, min_support: int = 2, max_rounds: int = 20,
         git: str | None = None) -> dict:
    """Adopt every rule the episodes support and nothing they contradict; retire what they now contradict.
    Idempotent: mining the same episodes into the same library adopts nothing new."""
    # a single relation composes nothing: an episode needs two hops to say anything about a rule. An episode
    # the host discounted is on the record and not evidence
    eps = [Episode(e.id, [rel(r) for r in e.relations], rel(e.conclusion), e.source) for e in episodes
           if len(e.relations) >= 2 and e.conclusion and e.id not in lib.discounted]
    retired = _retire_contradicted(eps, lib, night, git)
    labels = {e.id: _labels(e.relations, lib) for e in eps}
    adopted: list[dict] = []
    rejected: dict[tuple[str, str], dict] = {}
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        seen: dict[tuple[str, str], Counter] = defaultdict(Counter)
        sources: dict[tuple[str, str], set] = defaultdict(set)
        evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e in eps:
            if len(e.relations) < 2:
                continue
            for prem in _premises(e.relations, lib):
                if prem in lib.rules:
                    continue
                seen[prem][e.conclusion] += 1
                sources[prem].add((e.source or e.id, e.conclusion))
                if len(evidence[prem]) < 5:
                    evidence[prem].append(e.id)
        candidates = []
        for prem, said in seen.items():
            c = _conclusion(said, lib, min_support)
            if c is None:
                rejected[prem] = {"why": "conflict", "conclusions": dict(said)}
                continue
            # distinct sources whose stated term entails the conclusion: the ones that CONFIRM it
            n = len({s for s, t in sources[prem] if c in lib.sup(t)})
            if (prem[0], prem[1], c) in lib.retired:
                rejected[prem] = {"why": "retired", "conclusion": c}
            elif n < min_support:
                rejected[prem] = {"why": "support", "conclusion": c, "support": n}
            else:
                candidates.append((n, Rule(prem[0], prem[1], c)))
        candidates.sort(key=lambda nr: (-nr[0], nr[1].first, nr[1].second))
        new = 0
        for n, rule in candidates:
            trial = lib.with_rule(rule)
            touched = [e for e in eps if rule.first in labels[e.id] and rule.second in labels[e.id]]
            bad = next(((e, d) for e in touched for d in [derive(e.relations, trial)] if _wrong(d, e.conclusion, trial)),
                       None)
            if bad is not None:
                e, d = bad
                rejected[rule.premise] = {"why": "counterexample", "conclusion": rule.conclusion, "episode": e.id,
                                          "states": e.conclusion,
                                          "derived": d.conclusion if d.status == "derived" else sorted(d.conclusions)}
                continue
            adopted.append(lib.adopt(rule, n, evidence[rule.premise], night, git=git))
            rejected.pop(rule.premise, None)
            new += 1
            for e in touched:
                labels[e.id] = _labels(e.relations, lib)
        if not new:
            break
    return {"episodes": len(eps), "rounds": rounds, "adopted": adopted, "retired": retired,
            "rejected": {f"{a} then {b}": v for (a, b), v in sorted(rejected.items())}, "rules": len(lib)}


# ── using the library ──────────────────────────────────────────────────────────────────────────────────
def fact_text(t: Triple) -> str:
    return f"{t.obj} is the {t.rel} of {t.subj}"


def certify(facts: list[Triple], steps: list[Rule], run_fn, tau_vm: dict[int, float] | None = None,
            chunk: int = 2, extra: list[Triple] = ()) -> tuple[bool, list[str], str]:
    """The facts and the rule steps as ONE chain program, two bindings to a frame, each recovered and
    checked exactly like a walked hop (`pipeline._check_chain`). -> (ok, failed clauses, program source)."""
    from .pipeline import HopTrace, _check_chain

    tau_vm = tau_vm or TAU_BUNDLE
    triples = list(facts) + [Triple(obj=s.conclusion, rel=s.label(), subj=s.first) for s in steps] + list(extra)
    trace = [HopTrace(query="", fact=fact_text(t), triple=t, ret_score=1.0,
                      source="lookup" if i < len(facts) else "rule") for i, t in enumerate(triples)]
    bundle = min(chunk, len(triples)) if chunk > 0 else len(triples)
    ok, failed, source, _ctrl = _check_chain(SimpleNamespace(relations=[t.rel for t in triples]), triples, trace,
                                             run_fn, tau_vm.get(bundle, min(tau_vm.values())), chunk)
    return ok, failed, source


def relate(paths: list[list[Triple]], lib: Library, run_fn, tau_vm: dict[int, float] | None = None,
           chunk: int = 2, overflow: bool = False) -> dict:
    """How is B related to A, from the paths of facts that lead from A to B (walk order). Speaks one
    relation only if every path that composes agrees, and every one of them certifies in the VM.

    -> {answer, verified, reason, paths: [{relations, facts, status, conclusion, steps, failed}]}.
    Reasons: no_path, relation_paths_overflow, no_rule, rule_split, vm_verify_failed."""
    out = {"answer": None, "verified": False, "reason": None, "paths": []}
    if overflow:
        out["reason"] = "relation_paths_overflow"
        return out
    if not paths:
        out["reason"] = "no_path"
        return out
    for p in paths:
        d = derive([t.rel for t in p], lib)
        entry = {"relations": [t.rel for t in p], "facts": [fact_text(t) for t in p], "status": d.status,
                 "conclusion": d.conclusion, "steps": [str(s) for s in d.steps]}
        if d.status == "split":
            entry["conclusions"] = sorted(d.conclusions)
        elif d.status == "derived":
            ok, failed, _src = certify(p, d.steps, run_fn, tau_vm, chunk)
            entry["certified"], entry["failed"] = ok, failed
            if not ok:
                entry["status"] = "vm_failed"
        out["paths"].append(entry)
    said = {e["conclusion"] for e in out["paths"] if e["status"] == "derived"}
    best = lib.finest(said)
    split = (len(said) > 1 and best is None) or any(e["status"] == "split" for e in out["paths"])
    unverified = any(e["status"] == "vm_failed" for e in out["paths"])
    if split:
        out["reason"] = "rule_split"
    elif unverified:
        out["reason"] = "vm_verify_failed"
    elif said:
        out["answer"], out["verified"] = best, True
    else:
        out["reason"] = "no_rule"
    return out


# ── the host's word on a contested rule ────────────────────────────────────────────────────────────────
def host_adopt(episodes: list[Episode], lib: Library, premise: tuple[str, str], conclusion: str, reason: str,
               night: str, git: str | None = None) -> dict:
    """The HOST adopts a rule the gate kept out, and discounts the episodes that stated otherwise -- the loop
    never does this (invariant 2). Refused unless, with the minority discounted, EVERY other episode on
    record still derives its own relation or nothing: the host overrides the minority, never the record.

    -> {adopted, discounted: [episode ids]} or {refused: why}."""
    first, second, conclusion = rel(premise[0]), rel(premise[1]), rel(conclusion)
    rule = Rule(first, second, conclusion)
    if rule.key in lib.retired:
        return {"refused": f"{rule} was retired: {lib.retired[rule.key].get('reason')}"}
    eps = [Episode(e.id, [rel(r) for r in e.relations], rel(e.conclusion), e.source) for e in episodes
           if len(e.relations) >= 2 and e.conclusion and e.id not in lib.discounted]
    minority = [e for e in eps if not lib.comparable(e.conclusion, conclusion)
                and (first, second) in _premises(e.relations, lib)]
    support = [e for e in eps if conclusion in lib.sup(e.conclusion) and (first, second) in _premises(e.relations, lib)]
    if not support:
        return {"refused": f"no episode on record supports {rule}"}
    trial = lib.with_rule(rule)
    drop = {e.id for e in minority}
    bad = next(((e, d) for e in eps if e.id not in drop for d in [derive(e.relations, trial)]
                if _wrong(d, e.conclusion, trial)), None)
    if bad is not None:
        e, d = bad
        return {"refused": f"episode {e.id} states {e.conclusion} and would derive "
                           f"{d.conclusion if d.status == 'derived' else sorted(d.conclusions)}: not a minority "
                           f"of this premise, so the host cannot discount it here"}
    for e in minority:
        lib.discount(e.id, reason=f"host: {reason} (against {rule})", night=night, git=git)
    line = lib.adopt(rule, len({e.source or e.id for e in support}), [e.id for e in support[:5]], night,
                     git=git, host=True, reason=reason, discounted=len(minority))
    return {"adopted": line, "discounted": sorted(drop)}


def main(argv=None) -> None:
    """``python -m cubbyllm.reasoning.skills contested --sleep <dir>`` lists what the night kept out;
    ``... adopt --sleep <dir> --premise "husband then father" --conclusion "father in law" --reason "..."``
    is the host's word on one of them."""
    import argparse
    ap = argparse.ArgumentParser(description="the skill library, for the host")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contested")
    c.add_argument("--sleep", required=True)
    a = sub.add_parser("adopt")
    a.add_argument("--sleep", required=True)
    a.add_argument("--premise", required=True, help='"first then second"')
    a.add_argument("--conclusion", required=True)
    a.add_argument("--reason", required=True, help="why the minority is not evidence (it goes on the ledger)")
    a.add_argument("--night", default=_dt.date.today().isoformat())
    args = ap.parse_args(argv)
    d = pathlib.Path(args.sleep)
    if args.cmd == "contested":
        p = d / "skills_contested.jsonl"
        for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
            print(line)
        return
    first, _, second = args.premise.partition(" then ")
    eps = []
    p = d / "skill_episodes.jsonl"
    for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if line.strip():
            e = json.loads(line)
            eps.append(Episode(e["id"], e["relations"], e["conclusion"], e.get("source")))
    print(json.dumps(host_adopt(eps, Library(d / "skills.jsonl"), (first, second), args.conclusion, args.reason,
                                args.night), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()


# ── the other way round: inverse rules ─────────────────────────────────────────────────────────────────
@dataclass
class InverseEpisode:
    """B is the `relation` of A, A's sex is `sex`, and A is the `inverse` of B -- both stated, by two
    statements of the source (Wikidata's relative statements both ways, or a father stated on the child and
    the child on the father), never one of them derived from the other."""
    id: str
    relation: str
    sex: str
    inverse: str
    source: str | None = None


def mine_inverses(episodes: list[InverseEpisode], lib: Library, night: str, min_support: int = 2,
                  git: str | None = None) -> dict:
    """The same gate as a composition rule's, for (relation, sex) -> inverse: every episode of the key states
    a term comparable with the conclusion, and enough distinct sources confirm it; an adopted inverse an
    episode contradicts is retired."""
    by: dict[tuple[str, str], Counter] = defaultdict(Counter)
    srcs: dict[tuple[str, str], set] = defaultdict(set)
    ev: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in episodes:
        if not e.sex or not e.relation or not e.inverse:
            continue
        key = (rel(e.relation), e.sex)
        by[key][rel(e.inverse)] += 1
        srcs[key].add((e.source or e.id, rel(e.inverse)))
        if len(ev[key]) < 5:
            ev[key].append(e.id)
    adopted, retired, rejected = [], [], {}
    for key, said in sorted(by.items()):
        have = lib.inverses.get(key)
        if have is not None:
            bad = [t for t in said if not lib.comparable(t, have["inverse"])]
            if bad:
                retired.append(lib.retire_inverse(*key, have["inverse"], reason=f"stated as {bad[0]}", night=night,
                                                  ))
            continue
        c = _conclusion(said, lib, min_support)
        if c is None:
            rejected[f"{key[0]} ({key[1]})"] = {"why": "conflict", "conclusions": dict(said)}
            continue
        n = len({s for s, t in srcs[key] if c in lib.sup(t)})
        if (*key, c) in lib.retired_inverses:
            rejected[f"{key[0]} ({key[1]})"] = {"why": "retired", "conclusion": c}
        elif n < min_support:
            rejected[f"{key[0]} ({key[1]})"] = {"why": "support", "conclusion": c, "support": n}
        else:
            adopted.append(lib.adopt_inverse(key[0], key[1], c, n, ev[key], night, git=git))
    return {"adopted": adopted, "retired": retired, "rejected": rejected, "inverses": len(lib.inverses)}


def relate_back(back: list[list[Triple]], sex_of_b: tuple[str, Triple] | None, lib: Library, run_fn,
                tau_vm: dict[int, float] | None = None, chunk: int = 2) -> dict:
    """How is B related to A when the store only leads from B to A: each path composes to what A is to B,
    the inverse rule for B's sex turns that round, and the VM certifies the facts, the rule steps, B's sex and
    the inverse step together. Speaks only if every composing path agrees -- `relate`'s rule."""
    out = {"answer": None, "verified": False, "reason": None, "paths": [], "direction": "inverse"}
    if not back:
        out["reason"] = "no_path"
        return out
    if sex_of_b is None:
        out["reason"] = "no_rule"
        out["needs"] = "the sex of the person asked about, for the inverse"
        return out
    sex, sex_fact = sex_of_b
    for p in back:
        d = derive([t.rel for t in p], lib)
        entry = {"relations": [t.rel for t in p], "facts": [fact_text(t) for t in p], "status": d.status,
                 "conclusion": None, "steps": [str(s) for s in d.steps], "via": d.conclusion}
        if d.status == "derived":
            inv = lib.inverse_of(d.conclusion, sex)
            if inv is None:
                entry["status"] = "no_rule"
            else:
                step = Triple(obj=inv, rel=f"inverse of {d.conclusion} for {sex}", subj=d.conclusion)
                ok, failed, _src = certify(list(p) + [sex_fact], d.steps, run_fn, tau_vm, chunk, extra=[step])
                entry.update(conclusion=inv, certified=ok, failed=failed,
                             steps=entry["steps"] + [f"the inverse of {d.conclusion} for a {sex} person is {inv}"])
                if not ok:
                    entry["status"] = "vm_failed"
        elif d.status == "split":
            entry["conclusions"] = sorted(d.conclusions)
        out["paths"].append(entry)
    said = {e["conclusion"] for e in out["paths"] if e["status"] == "derived"}
    best = lib.finest(said)
    split = (len(said) > 1 and best is None) or any(e["status"] == "split" for e in out["paths"])
    unverified = any(e["status"] == "vm_failed" for e in out["paths"])
    if split:
        out["reason"] = "rule_split"
    elif unverified:
        out["reason"] = "vm_verify_failed"
    elif said:
        out["answer"], out["verified"] = best, True
    else:
        out["reason"] = "no_rule"
    return out
