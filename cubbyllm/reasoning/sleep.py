"""sleep -- the nightly cycle: the day's questions replayed, its refusals routed, what it learned made durable.

Wired: STANDALONE (a batch job, ``python -m cubbyllm.reasoning.sleep``; nothing on the serve path imports
it). Nothing it writes changes what the loop may SPEAK: an episode only proposes a plan, a consolidated fact
is looked up and verified like any other, a queue item is a request for a host decision.

Why a sleep. By day the loop answers or refuses, one question at a time, and forgets: the ask loop's
history lived in memory, the facts it fetched lived in the running store, and a refusal was a status line.
At night the day becomes material:

    collect      the day's records (the ask loop's history, or a bench's rows)
    replay       every certified chain becomes a hippocampal episode (a plan candidate for tomorrow);
                 an episode whose chain lost a fact is retired, never deleted
    consolidate  facts the gate accepted from a TRUSTED source go to a durable, append-only log the
                 server loads at boot -- what was learned today is still known tomorrow
    triage       every refusal is routed by its reason to the queue for the level that must change:
                 emitter curriculum, relation wordings, clarify, source gaps, VM review. Items merge
                 across nights; an item seen night after night is the need monitor's signal
    skills       relation asks with a relation stated by gold or the asker become episodes; the skill
                 library (reasoning/skills.py) adopts every composition rule the episodes support and
                 none they contradict, and retires what a new episode contradicts. Every past episode
                 is re-derived before a rule gets in: the gate is the whole record, not tonight
    audit        a spoken answer contradicted by gold or by the asker is a DEFECT (the kill line);
                 every outcome is delivered to the striatum as a reward, from the gate or the audit
    report       what changed, what recurs, and the refusal mix against the previous night

The seven invariants hold by construction: no model is called here, so none judges (1, 6); every action
is a host decision on the record (2); a refusal stays a refusal (3); episodes, facts and queue items are
retired with a reason, never deleted, and every night records the git revision and the input's hash (4);
it runs on the CPU in seconds (5); stdlib only (7).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass, field

from ..core.protocols import Wiring
from .hippocampus import Hippocampus
from .planner import normalize
from .skills import Episode, Library, mine
from .striatum import Striatum, question_shape

# reason -> (queue, what has to change). A reason not listed lands in "unrouted", and the report says so:
# the table grows by decision, never by default.
ROUTES: dict[str, tuple[str, str]] = {
    "plan_does_not_cover_question": ("emitter_curriculum", "the emitter's plan leaves words of the question uncovered"),
    "no_plan": ("emitter_curriculum", "the emitter produced no plan for this shape"),
    "emit_error": ("emitter_curriculum", "the emitter failed outright"),
    "unparseable": ("emitter_curriculum", "the emitter's program did not parse"),
    "unknown_relation": ("relation_wordings", "no held relation answers to the plan's wording"),
    "ambiguous_relation": ("clarify", "several held relations fit the wording"),
    "ambiguous_hop": ("clarify", "a multi-valued hop: the question needs a qualifier or a choice"),
    "retrieval_exhausted": ("source_gaps", "no fact reached the next hop, even after the fetch"),
    "latent_only": ("latent_audit", "only the latent tier had a chain; held back, never spoken"),
    "answer_type_mismatch": ("vm_review", "the walk ended on the wrong kind of answer"),
    "vm_verify_failed": ("vm_review", "the VM did not verify the chain the walk built"),
    # the ask loop's profile / date / introspection paths (standin/ask.py)
    "ambiguous_entity": ("clarify", "the name fits several items; the loop asked which one"),
    "date_too_broad": ("clarify", "too much is dated in the range to answer in one breath"),
    "entity_unresolved": ("entity_resolution", "the source found no item for the name"),
    "profile_empty": ("source_gaps", "the item was found and holds nothing the ask wants"),
    "profile_unrecovered": ("vm_review", "facts were held but none recovered above the profile threshold"),
    "no_events_for_date": ("capability_gaps", "nothing indexes events by date"),
    "no_self_record": ("capability_gaps", "the loop keeps no record of that subject about itself"),
    # the relation ask (reasoning/skills.py): "how is B related to A?"
    "no_path": ("source_gaps", "no chain of held facts leads from the one to the other"),
    "no_rule": ("skill_gaps", "a chain joins them but no adopted rule composes it; gold or the asker can teach it"),
    "rule_split": ("skill_review", "the rules compose the chain (or its paths) to two relations: a rule is wrong"),
    "relation_paths_overflow": ("capability_gaps", "more paths between them than the relation ask follows"),
}
# reasons that come WITH a spoken answer: a profile or an introspection answered line by line, each line
# recovered by the VM. Not refusals.
ANSWERED = {"profile", "self"}
QUEUES = sorted({q for q, _ in ROUTES.values()} | {"unrouted", "defects"})
SPOKEN_OK = ("correct", "near")                 # audit matches that are not defects ("near" = a coarser answer)


def _sha(*parts: str, n: int = 16) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:n]


def _reason_key(reason: str | None) -> str | None:
    return None if not reason else reason.split(":", 1)[0].strip()


def answered(r: "DayRecord") -> bool:
    """Spoken: a VM-verified chain, or a profile / introspection answer whose lines the VM recovered."""
    return r.verified or (_reason_key(r.reason) in ANSWERED and bool(r.answer))


@dataclass
class DayRecord:
    """One question of the day, whatever produced it."""
    id: str
    question: str
    verified: bool
    answer: str | None = None
    reason: str | None = None
    plan: list[str] | None = None
    seed: str | None = None
    chain: list[str] = field(default_factory=list)
    learned: list[dict] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    gold: str | None = None
    match: str | None = None                    # an audit's verdict on a spoken answer
    feedback: str | None = None                 # the asker's: "wrong" | "right"
    run: str = ""
    ts: str | None = None
    kind: str | None = None                     # "relation" for a relation ask (reasoning/skills.py)
    asker: str | None = None                    # who gave the feedback, when someone did
    paths: list[list[str]] = field(default_factory=list)   # a relation ask's paths, as relation chains


def load_day(path: str | pathlib.Path) -> list[DayRecord]:
    """The ask loop's history (``.jsonl``, one record a line) or a search-and-learn bench's rows
    (``.json`` with ``rows`` and ``verified``, exp_r11's format)."""
    p = pathlib.Path(path)
    run = p.stem
    out: list[DayRecord] = []
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        # an asker's feedback is its own line (AskLoop.feedback), folded into the record it names; the last word wins
        said = {r["feedback_for"]: r for r in rows if "feedback_for" in r}
        for r in rows:
            if "feedback_for" in r:
                continue
            fb = said.get(r.get("id"))
            if fb:
                r = dict(r, feedback=fb.get("verdict") or r.get("feedback"), gold=fb.get("relation") or r.get("gold"),
                         asker=fb.get("asker") or "asker")
            q = r.get("question") or r.get("q") or ""
            out.append(DayRecord(
                id=r.get("id") or _sha(run, q, str(r.get("ts", ""))), question=q, verified=bool(r.get("verified")),
                answer=r.get("answer"), reason=r.get("reason"), plan=r.get("plan"), seed=r.get("seed"),
                chain=list(r.get("trace") or []), learned=list(r.get("learned") or []),
                unknown=list(r.get("unknown") or (r.get("refused") or {}).get("unknown_relations") or []),
                gold=r.get("gold"), match=r.get("match"), feedback=r.get("feedback"), run=run, ts=r.get("ts"),
                kind=r.get("kind"), asker=r.get("asker"),
                # an answer turned round by an inverse rule is not an episode of its paths' composition
                paths=[list(p) for p in r.get("paths") or []] if r.get("direction") != "inverse" else []))
        return out
    d = json.loads(p.read_text(encoding="utf-8"))
    spoken = {v["q"]: v for v in d.get("verified", []) if isinstance(v, dict) and "q" in v}
    for r in d.get("rows", []):
        q = r.get("q", "")
        v = spoken.get(q, {})
        said = v if r.get("verified") else {}
        out.append(DayRecord(
            id=_sha(run, q), question=q, verified=bool(r.get("verified")), answer=r.get("answer"),
            reason=None if r.get("verified") else (r.get("final") or r.get("first") or ("no_plan" if not r.get("plan") else None)),
            plan=r.get("plan"), seed=r.get("seed"), chain=list(said.get("trace") or []),
            learned=[{"fact": f, "status": "accepted", "source": d.get("source", "")} for f in said.get("learned", [])],
            unknown=list(r.get("unknown") or []), gold=r.get("gold") or v.get("gold"), match=v.get("match"), run=run))
    return out


def git_revision(repo: str | pathlib.Path | None = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Ledger:
    """Append-only jsonl; a line already on the record (same night, phase, action, id) is not written twice,
    so a re-run of the same night leaves the ledger as it was."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.seen: set[tuple] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.seen.add((r.get("night"), r.get("phase"), r.get("action"), r.get("id")))

    def write(self, night: str, phase: str, action: str, id_: str, **detail) -> bool:
        key = (night, phase, action, id_)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"night": night, "phase": phase, "action": action, "id": id_,
                                "ts": _dt.datetime.now().isoformat(timespec="seconds"), **detail},
                               ensure_ascii=False) + "\n")
        return True


# ── replay ──────────────────────────────────────────────────────────────────────────────────────────────
def replay(records: list[DayRecord], hip: Hippocampus, night: str, ledger: Ledger, contains=None) -> dict:
    """Certified chains -> episodes (deduplicated by question and chain); with ``contains`` (fact -> bool, the
    store as it stands), every live episode whose chain lost a fact is retired with the reason."""
    have = {(normalize(e.question), tuple(e.chain)) for e in hip.episodes}
    written = 0
    for r in records:
        if not (r.verified and r.chain and r.plan and r.seed and r.answer):
            continue
        if r.kind == "relation":
            continue                             # a composed relation, not a walk: its episodes are the skill record's
        key = (normalize(r.question), tuple(r.chain))
        if key in have:
            continue
        have.add(key)
        hip.write(r.question, r.plan, r.seed, r.chain, r.answer,
                  provenance={"night": night, "record": r.id, "run": r.run, "ts": r.ts})
        ledger.write(night, "replay", "episode", _sha(*key[:1], *key[1]), question=r.question, record=r.id)
        written += 1
    stale = 0
    if contains is not None:
        for e in hip.episodes:
            if e.retired:
                continue
            gone = [f for f in e.chain if not contains(f)]
            if gone:
                e.retired = True
                e.provenance["retired"] = {"night": night, "reason": "stale: a chain fact left the store", "facts": gone}
                ledger.write(night, "replay", "retire_episode", _sha(normalize(e.question), *e.chain),
                             question=e.question, facts=gone)
                stale += 1
    return {"episodes_written": written, "episodes_stale": stale, "episodes_live": len(hip)}


# ── consolidate ─────────────────────────────────────────────────────────────────────────────────────────
def trusted_source(source: str, trusted: set[str]) -> bool:
    """A provenance whose facts may be made durable. The web (2026-09-24) is trusted only as TWO or
    more independent sites ('web:a.org+web:b.com') -- the same bar it has to clear to be spoken."""
    s = (source or "").lower()
    if not s or "latent" in s:
        return False
    head = s.split()[0]
    if head.startswith("web:"):
        return "web" in trusted and len({x for x in head.split("+") if x.startswith("web:")}) >= 2
    return head in trusted


def load_facts(path: pathlib.Path) -> dict[str, dict]:
    """The durable fact log: fact key -> its record; a later ``{"retire": fact}`` line retires it."""
    live: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "retire" in r:
                live.pop(" ".join(r["retire"].split()), None)
            else:
                live[" ".join(r["fact"].split())] = r
    return live


def consolidate(records: list[DayRecord], path: pathlib.Path, night: str, ledger: Ledger,
                trusted: set[str]) -> dict:
    """Facts the gate ACCEPTED from a trusted source, appended once each with their provenance. A latent
    source's fact, a duplicate, a contradiction or an unparseable line never enters."""
    live = load_facts(path)
    added = skipped_untrusted = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            for p in r.learned:
                if p.get("status") != "accepted":
                    continue
                if not trusted_source(p.get("source", ""), trusted):
                    skipped_untrusted += 1
                    continue
                key = " ".join(str(p.get("fact", "")).split())
                if not key or key in live:
                    continue
                rec = {"fact": key, "rel": p.get("rel"), "source": p.get("source"), "entity": p.get("entity"),
                       "fetched_at": p.get("fetched_at"), "snapshot_before": p.get("snapshot_before"),
                       "time": p.get("time"), "record": r.id, "night": night}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                live[key] = rec
                ledger.write(night, "consolidate", "fact", _sha(key), fact=key, source=p.get("source"), record=r.id)
                added += 1
    return {"facts_added": added, "facts_live": len(live), "facts_untrusted_skipped": skipped_untrusted}


# ── triage ──────────────────────────────────────────────────────────────────────────────────────────────
def route(r: DayRecord) -> list[tuple[str, str]]:
    """(queue, key) for one refused or defective record; a record may feed several keys (one per unknown
    wording)."""
    rk = _reason_key(r.reason)
    if answered(r) or not rk:
        return []
    queue = ROUTES.get(rk, ("unrouted", ""))[0]
    shape = question_shape(r.question)
    if queue == "relation_wordings":
        words = r.unknown or (r.plan or [])[:1]
        return [(queue, normalize(w)) for w in words if w] or [(queue, "?")]
    if queue == "source_gaps":
        return [(queue, f"{normalize(r.seed or '?')} | {normalize((r.plan or ['?'])[-1])}")]
    if queue == "clarify":
        return [(queue, f"{' / '.join(normalize(x) for x in (r.plan or ['?']))} | {shape}")]
    if queue == "unrouted":
        return [(queue, rk)]
    return [(queue, f"{rk} | {shape}")]


def is_defect(r: DayRecord) -> bool:
    if not r.verified:
        return False
    if r.feedback == "wrong":
        return True
    if r.kind == "relation" and r.gold and r.answer:
        return normalize(r.answer) != normalize(r.gold)     # a spoken relation gold disagrees with
    return r.match is not None and r.match not in SPOKEN_OK


def triage(records: list[DayRecord], qdir: pathlib.Path, night: str, ledger: Ledger) -> dict:
    """Route every refusal (and every defect) to its queue; merge into the queue files by night, so a
    re-run of a night overwrites that night's counts instead of adding to them."""
    fresh: dict[str, dict[str, list[DayRecord]]] = {}
    for r in records:
        pairs = [("defects", f"{normalize(r.question)}")] if is_defect(r) else route(r)
        for q, key in pairs:
            fresh.setdefault(q, {}).setdefault(key, []).append(r)
    out = {}
    qdir.mkdir(parents=True, exist_ok=True)
    for q in QUEUES:
        path = qdir / f"{q}.jsonl"
        items: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    it = json.loads(line)
                    items[it["id"]] = it
        for key, rs in fresh.get(q, {}).items():
            iid = _sha(q, key)
            it = items.get(iid) or {"id": iid, "queue": q, "key": key, "status": "open", "by_night": {},
                                    "examples": [], "first_seen": night}
            it["by_night"][night] = len(rs)
            it["count"] = sum(it["by_night"].values())
            it["nights"] = len(it["by_night"])
            it["last_seen"] = max(it.get("last_seen", night), night)
            ex = [{"q": r.question, "record": r.id, "answer": r.answer, "gold": r.gold, "chain": r.chain[:3]}
                  for r in rs[:5]]
            seen_q = {e["q"] for e in ex}
            it["examples"] = (ex + [e for e in it["examples"] if e["q"] not in seen_q])[:5]
            items[iid] = it
            if ledger.write(night, "triage", "item", iid, queue=q, key=key, n=len(rs)):
                pass
        with path.open("w", encoding="utf-8") as f:
            for it in sorted(items.values(), key=lambda i: (-i.get("count", 0), i["key"])):
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        out[q] = {"items": len(items), "tonight": sum(len(v) for v in fresh.get(q, {}).values()),
                  "top": [(i["key"], i["count"], i["nights"], i["status"])
                          for i in sorted(items.values(), key=lambda i: -i.get("count", 0))[:8]]}
    return out


# ── audit ───────────────────────────────────────────────────────────────────────────────────────────────
def audit(records: list[DayRecord], st: Striatum, night: str, ledger: Ledger, proposer: str = "emitter") -> dict:
    """Every outcome to the striatum: certified +1, refused 0, a defect -5 (source 'audit'). Returns the
    counts and the shapes the proposer is worst at, which is where tomorrow's curriculum should go."""
    n = {"certified": 0, "refused": 0, "wrong": 0}
    for r in records:
        outcome = "wrong" if is_defect(r) else ("certified" if answered(r) else "refused")
        if ledger.write(night, "audit", outcome, r.id, question=r.question):
            st.reward(r.question, proposer, outcome, source="audit" if outcome == "wrong" else "gate")
        n[outcome] += 1
    # lowest expectation first; among equals the most-tried, where a flat 0 means "refused every time"
    keys = sorted((k for k in st.expected if k.startswith(proposer + "|")),
                  key=lambda k: (st.expected[k], -st.n.get(k, 0)))[:8]
    return {**n, "worst_shapes": [(k.split("|", 1)[-1], round(st.expected[k], 3), st.n.get(k, 0)) for k in keys]}


# ── skills ──────────────────────────────────────────────────────────────────────────────────────────────
def episodes_of(records: list[DayRecord]) -> list[Episode]:
    """Relation asks whose relation someone OTHER than the loop stated: a bench's gold, or an asker (who said
    a spoken relation was right, or stated the relation). Every path of the ask is an episode -- each must
    compose to it. An asker's episode carries the asker as its source: the library counts distinct sources,
    so one asker cannot put a rule in alone."""
    out = []
    for r in records:
        if r.kind != "relation" or not r.paths:
            continue
        stated = r.gold or (r.answer if r.verified and r.feedback == "right" else None)
        if not stated:
            continue
        for i, p in enumerate(r.paths):
            if len(p) >= 2:
                out.append(Episode(f"{r.id}#{i}", list(p), stated, source=f"asker:{r.asker}" if r.asker else None))
    return out


def skills_night(records: list[DayRecord], out: pathlib.Path, night: str, ledger: Ledger,
                 git: str | None = None, min_support: int = 2) -> dict:
    """Tonight's episodes join the record (``skill_episodes.jsonl``, append-only, deduplicated); the library
    (``skills.jsonl``) is then mined against the WHOLE record -- a rule gets in only if every episode ever
    kept still derives its own relation or nothing."""
    ep_path = out / "skill_episodes.jsonl"
    kept: dict[str, Episode] = {}
    if ep_path.exists():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                kept[e["id"]] = Episode(e["id"], e["relations"], e["conclusion"], source=e.get("source"))
    fresh = [e for e in episodes_of(records) if e.id not in kept]
    if fresh:
        with ep_path.open("a", encoding="utf-8") as f:
            for e in fresh:
                f.write(json.dumps({"id": e.id, "relations": e.relations, "conclusion": e.conclusion,
                                    "source": e.source, "night": night}, ensure_ascii=False) + "\n")
                kept[e.id] = e
    # the world's relation words at different grain (`skills_taxonomy.json`: term -> the terms it entails),
    # when the world has one; without it every term entails only itself
    lib = Library(out / "skills.jsonl")          # reads skills_taxonomy.json beside it, when there is one
    rep = mine(list(kept.values()), lib, night, min_support=min_support, git=git)
    # contested premises: kept out by a conflict or a counterexample. A view for the host, rewritten each
    # night, most one-sided first -- 114 episodes against 2 is a different question from 50 against 50, and
    # the host (never the loop) decides whether the minority is label noise
    contested = []
    for prem, v in rep["rejected"].items():
        if v["why"] == "conflict":
            counts = v["conclusions"]
            top = max(counts, key=counts.get)
            contested.append({"premise": prem, "why": "conflict", "majority": top, "counts": counts,
                              "share": round(counts[top] / sum(counts.values()), 3)})
        elif v["why"] == "counterexample":
            contested.append({"premise": prem, "why": "counterexample", "majority": v["conclusion"],
                              "episode": v["episode"], "states": v["states"], "derived": v["derived"], "share": None})
    contested.sort(key=lambda c: (c["share"] is None, -(c["share"] or 0), c["premise"]))
    (out / "skills_contested.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in contested),
                                                encoding="utf-8")
    for line in rep["adopted"]:
        ledger.write(night, "skills", "adopt", _sha(line["first"], line["second"], line["conclusion"]),
                     rule=f"{line['first']} then {line['second']} -> {line['conclusion']}", support=line["support"])
    for line in rep["retired"]:
        ledger.write(night, "skills", "retire", _sha(line["first"], line["second"], line["conclusion"]),
                     rule=f"{line['first']} then {line['second']} -> {line['conclusion']}", reason=line["reason"])
    return {"episodes_new": len(fresh), "episodes_total": len(kept), "rounds": rep["rounds"],
            "adopted": len(rep["adopted"]), "retired": len(rep["retired"]), "rules": rep["rules"],
            "rejected": _count(v["why"] for v in rep["rejected"].values()), "contested": contested[:8]}


# ── the night ───────────────────────────────────────────────────────────────────────────────────────────
def sleep(day: list[str | pathlib.Path], out: str | pathlib.Path, night: str | None = None,
          trusted: set[str] | None = None, keep_episodes: int | None = None, contains=None,
          repo: str | pathlib.Path | None = None) -> dict:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records: list[DayRecord] = []
    digests = []
    for p in day:
        records += load_day(p)
        digests.append(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16])
    night = night or _dt.date.today().isoformat()
    ledger = Ledger(out / "ledger.jsonl")
    ledger.write(night, "collect", "night", _sha(night, *digests), inputs=[str(p) for p in day],
                 input_sha256=digests, records=len(records), git=git_revision(repo))

    hip_path, st_path = out / "hippocampus.jsonl", out / "striatum.json"
    hip = Hippocampus.load(hip_path) if hip_path.exists() else Hippocampus()
    st = Striatum.load(st_path) if st_path.exists() else Striatum()

    rep = {"night": night, "records": len(records), "verified": sum(answered(r) for r in records)}
    rep["replay"] = replay(records, hip, night, ledger, contains)
    if keep_episodes:
        for e in hip.consolidate(keep_episodes):
            e.provenance["retired"] = {"night": night, "reason": f"consolidate: beyond the {keep_episodes} most useful"}
            ledger.write(night, "replay", "retire_episode", _sha(normalize(e.question), *e.chain),
                         question=e.question, reason="consolidate")
        rep["replay"]["episodes_live"] = len(hip)
    rep["consolidate"] = consolidate(records, out / "learned_facts.jsonl", night, ledger,
                                     trusted or {"wikidata", "web"})
    rep["triage"] = triage(records, out / "queues", night, ledger)
    rep["audit"] = audit(records, st, night, ledger)
    rep["skills"] = skills_night(records, out, night, ledger, git=git_revision(repo))
    rep["refusals"] = dict(sorted(_count(_reason_key(r.reason) or "?" for r in records if not answered(r)).items(),
                                  key=lambda kv: -kv[1]))
    # what the NIGHT did, from the ledger -- so re-running a night reports the same night, not "nothing new"
    def _n(phase, action):
        return sum(1 for k in ledger.seen if k[0] == night and k[1] == phase and k[2] == action)
    rep["replay"]["episodes_written"] = _n("replay", "episode")
    rep["replay"]["episodes_stale"] = _n("replay", "retire_episode")
    rep["consolidate"]["facts_added"] = _n("consolidate", "fact")
    rep["skills"]["adopted"] = _n("skills", "adopt")
    rep["skills"]["retired"] = _n("skills", "retire")
    hip.save(hip_path)
    st.save(st_path)

    ndir = out / "nights" / night
    ndir.mkdir(parents=True, exist_ok=True)
    prev = _previous(out / "nights", night)
    (ndir / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    (ndir / "report.md").write_text(render(rep, prev), encoding="utf-8")
    return rep


def _count(it) -> dict:
    c: dict = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


def _previous(ndir: pathlib.Path, night: str) -> dict | None:
    earlier = sorted(p.name for p in ndir.iterdir() if p.is_dir() and p.name < night) if ndir.exists() else []
    if not earlier:
        return None
    f = ndir / earlier[-1] / "summary.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def render(rep: dict, prev: dict | None) -> str:
    n = max(rep["records"], 1)
    L = [f"# Sleep — night {rep['night']}", "",
         f"{rep['records']} questions, {rep['verified']} answered through the gate "
         f"({100 * rep['verified'] / n:.1f}%), **{rep['audit']['wrong']} defects** (spoken and wrong).", ""]
    if rep["audit"]["wrong"]:
        L += ["**Kill line crossed:** see `queues/defects.jsonl`; every chain fact there is a retirement candidate.", ""]
    L += ["## Refusals by reason", "", "| reason | tonight | share | previous night |", "|---|---:|---:|---:|"]
    pr = (prev or {}).get("refusals", {})
    pn = max((prev or {}).get("records", 0), 1)
    for k, v in rep["refusals"].items():
        was = f"{pr[k]} ({100 * pr[k] / pn:.1f}%)" if k in pr else "—"
        L.append(f"| {k} | {v} | {100 * v / n:.1f}% | {was} |")
    r, c = rep["replay"], rep["consolidate"]
    L += ["", "## What the day became", "",
          f"- **Episodes:** {r['episodes_written']} certified chains remembered, {r['episodes_stale']} retired as "
          f"stale; {r['episodes_live']} live. Tomorrow they are plan candidates, through the same gate.",
          f"- **Facts:** {c['facts_added']} accepted facts made durable ({c['facts_live']} in the log); "
          f"{c['facts_untrusted_skipped']} skipped as untrusted or latent."]
    s = rep.get("skills")
    if s and (s["episodes_total"] or s["rules"]):
        L.append(f"- **Skills:** {s['adopted']} composition rules adopted and {s['retired']} retired tonight; "
                 f"{s['rules']} in the library, gated against {s['episodes_total']} episodes "
                 f"({s['episodes_new']} new).")
        for c in s.get("contested") or []:
            said = (", ".join(f"{k} ×{n}" for k, n in sorted(c["counts"].items(), key=lambda kv: -kv[1]))
                    if c["why"] == "conflict" else f"{c['majority']}, but episode {c['episode']} states {c['states']}")
            L.append(f"  - contested, kept out: `{c['premise']}` → {said}")
    L += ["", "## Queues", ""]
    for q, v in rep["triage"].items():
        if not v["items"]:
            continue
        L.append(f"**{q}** — {v['tonight']} tonight, {v['items']} open items")
        L.append("")
        for key, cnt, nights, status in v["top"]:
            flag = " · recurring" if nights > 1 else ""
            L.append(f"- `{key}` ×{cnt} ({nights} night{'s' if nights > 1 else ''}{flag}, {status})")
        L.append("")
    if rep["audit"]["worst_shapes"]:
        L += ["## Where the emitter is weakest", "",
              "The striatum's expected reward per question shape (+1 certified, 0 refused, -5 wrong), "
              "lowest first; the curriculum's first candidates.", ""]
        L += [f"- `{s}`: {v:+.3f} over {n} tries" for s, v, n in rep["audit"]["worst_shapes"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="The nightly sleep cycle over the day's ask-loop records.")
    ap.add_argument("--day", nargs="+", required=True, help="ask history .jsonl and/or bench .json files")
    ap.add_argument("--out", required=True, help="the sleep directory (state, queues, ledger, nightly reports)")
    ap.add_argument("--night", default=None, help="the night's id (default: today)")
    ap.add_argument("--trusted", nargs="*", default=["wikidata", "web"],
                    help="sources whose accepted facts are kept ('web' = facts two or more independent sites stated)")
    ap.add_argument("--keep-episodes", type=int, default=None)
    a = ap.parse_args(argv)
    rep = sleep(a.day, a.out, night=a.night, trusted=set(a.trusted), keep_episodes=a.keep_episodes)
    print(render(rep, None))


if __name__ == "__main__":
    main()


__wiring__ = Wiring.STANDALONE
