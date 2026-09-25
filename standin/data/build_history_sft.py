"""Talk records from the history graph (standin/history_graph.py): what caused an event, what it led to, how one
event led to another, which came first, what a change to an event reaches -- in the VM's facts-block form, the
graph's links written as lines (`Battle of Fort Sumter — caused: American Civil War`).

    python standin/data/build_history_sft.py [--graph standin/data/out/history_graph.jsonl] [--seed 0]

Families (every answer templated from the lines the block shows; each passes build_ground_sft.check):
  hist_cause       "What led to X?"             -> each cause the block gives, by its link ("A caused X.")
  hist_effect      "What did X lead to?"        -> each effect the block gives
  hist_chain       "How did A lead to C?"       -> the path, hop by hop ("A caused B, and B led to C.")
  hist_downstream  "What would change if X had not happened?" -> everything the block has following from X
  hist_part        "What was X part of?"        -> "X was part of Y."
  hist_order       "Which came first, X or Y?"  -> the earlier, with both dates
  hist_when / hist_where / hist_who             -> the event's date, place, participants
  hist_fact        a fact about a person or place -> the sentence that states it
  hist_absent      a cause, effect or date question with the lines that answer it taken out -> "The facts don't say."
Split by the asked event's name hash (build_ground_sft.split_of, held = 10%); a train block never holds a held event.
Events only the newspaper reports (build_nyt_events.py) are at most half of the when/where/who/order families, so
the books' history is not crowded out; each family keeps at most --max-per-family records (seeded).
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, random, re, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_ground_sft import ABSENT, check, norm, render, split_of  # noqa: E402
from history_graph import FORWARD, HistoryGraph, year_text  # noqa: E402

SOURCE = "history graph (standin/data/build_history_graph.py over the local history books)"
CAUSE_Q = ["What led to {x}?", "What caused {x}?", "What brought about {x}?"]
EFFECT_Q = ["What did {x} lead to?", "What came of {x}?", "What were the consequences of {x}?"]
WHEN_Q = ["When did {x} happen?", "When was {x}?", "What is the date of {x}?"]
WHERE_Q = ["Where did {x} take place?", "Where was {x}?"]
WHO_Q = ["Who took part in {x}?", "Who was involved in {x}?"]


def says(a: str, rel: str, b: str) -> str:
    """One link as a clause, a before b in the sentence."""
    a, b = the(a), the(b)
    return {"caused": f"{a} caused {b}", "contributed to": f"{a} contributed to {b}",
            "precursor of": f"{a} set the stage for {b}", "response to": f"{a} was a response to {b}",
            "part of": f"{a} was part of {b}", "ended": f"{a} ended {b}"}[rel]


_THE = re.compile(r"^[A-Z][a-z]+ (?:of|at|on|in|to|from|for|against|with|by|between|over)\b")


def the(name: str) -> str:
    """'Fall of Richmond' -> 'the Fall of Richmond' in running text; 'Lee's Retreat', 'Pickett's Charge' stay bare."""
    return f"the {name}" if _THE.match(name) else name


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def join(clauses: list[str]) -> str:
    body = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return body[:1].upper() + body[1:] + "."


def listing(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" and {xs[-1]}"


def quotable(w: dict | None, context: bool = False) -> bool:
    """A date the talk data may show and answer with: one a book stated (or, as a line, the year its passage is
    about) -- never one the graph inferred from the book's running year or a larger event."""
    return bool(w) and w.get("basis", "stated") in (("stated", "context") if context else ("stated",))


class Blocks:
    def __init__(self, g: HistoryGraph, rng: random.Random):
        self.g, self.rng = g, rng

    def name(self, i):
        return self.g.events[i].name

    def own(self, i, date=True, where=True, who=True) -> list[list[str]]:
        ev, out = self.g.events[i], []
        if date and quotable(ev.when, context=True):
            w = ev.when
            if w.get("day"):
                out.append([ev.name, "date", w["day"]])
            elif w["y0"] == w["y1"]:
                out.append([ev.name, "year", year_text(w["y0"])])
            else:
                out += [[ev.name, "start year", year_text(w["y0"])], [ev.name, "end year", year_text(w["y1"])]]
        if where:
            out += [[ev.name, "location", x] for x in ev.where[:2]]
        if who:
            out += [[ev.name, "participant", x] for x in ev.who[:4]]
        return out

    def link(self, a, rel, b) -> list[str]:
        return [self.name(a), rel, self.name(b)]

    def when_says(self, i) -> tuple[str, list[str]] | None:
        ev = self.g.events[i]
        if not quotable(ev.when):
            return None
        w = ev.when
        if w.get("day"):
            return cap(f"{the(ev.name)} took place on {w['day']}."), [w["day"]]
        if w["y0"] == w["y1"]:
            return cap(f"{the(ev.name)} took place in {year_text(w['y0'])}."), [year_text(w["y0"])]
        return cap(f"{the(ev.name)} ran from {year_text(w['y0'])} to {year_text(w['y1'])}."), [year_text(w["y0"]), year_text(w["y1"])]


MONTHS = "January February March April May June July August September October November December".split()
# a person or place fact as a question and the sentence that answers it (build_history_graph.FACT_RELATIONS)
FACT_SAY = {"birth date": ("When was {e} born?", "{e} was born {on} {v}."),
            "death date": ("When did {e} die?", "{e} died {on} {v}."),
            "birthplace": ("Where was {e} born?", "{e} was born in {v}."),
            "place of death": ("Where did {e} die?", "{e} died in {v}."),
            "position held": ("What position did {e} hold?", "{e} held the position of {v}."),
            "spouse": ("Who was {e}'s spouse?", "{e}'s spouse was {v}."),
            "parent": ("Who was a parent of {e}?", "{v} was a parent of {e}."),
            "child": ("Who was a child of {e}?", "{v} was a child of {e}."),
            "sibling": ("Who was a sibling of {e}?", "{v} was a sibling of {e}."),
            "predecessor": ("Who came before {e}?", "{e}'s predecessor was {v}."),
            "successor": ("Who came after {e}?", "{e}'s successor was {v}."),
            "founder": ("Who founded {e}?", "{e} was founded by {v}."),
            "capital": ("What was the capital of {e}?", "The capital of {e} was {v}."),
            "author": ("Who wrote {e}?", "{e} was written by {v}."),
            "member of": ("What was {e} a member of?", "{e} was a member of {v}."),
            "occupation": ("What was {e}'s occupation?", "{e}'s occupation was {v}."),
            "nationality": ("What was {e}'s nationality?", "{e}'s nationality was {v}."),
            "ruler": ("Who ruled {e}?", "{e} was ruled by {v}."),
            "located in": ("Where is {e}?", "{e} is in {v}.")}


def day_words(iso: str) -> str:
    """'1863-07-01' -> 'July 1, 1863' (the facts line keeps the VM's ISO form; the answer reads it out)."""
    y, m, d = iso.split("-")
    return f"{MONTHS[int(m) - 1]} {int(d)}, {int(y)}"


NEWS_SHARE = 0.5          # at most this share of a when/where/who/order family asks about a newspaper-only event


def origin(ev) -> str:
    """'news' when only the newspaper reports the event (build_nyt_events.py), else 'book'."""
    return "news" if ev.sources and all(s[0] == "nyt" for s in ev.sources) else "book"


def build(g: HistoryGraph, rng: random.Random, absent_share: float, per_event: int):
    B = Blocks(g, rng)
    funnel, kept = collections.Counter(), []
    split = {i: split_of(ev.name) for i, ev in g.events.items()}
    dated = [i for i, ev in g.events.items() if quotable(ev.when)]
    dated_in = collections.defaultdict(list)                  # split -> its dated events: a distractor stays in split
    for j in dated:
        dated_in[split[j]].append(j)

    def distractor(i, n=1):
        pool = dated_in[split[i]]
        pick = [j for j in rng.sample(pool, min(n + 1, len(pool))) if j != i][:n]
        return [x for j in pick for x in B.own(j, where=False, who=False)]

    def keep(fam, i, q, a, facts, gold, forbid, asked, task="ask", extra=None):
        subjects = {norm(f[0]) for f in facts}
        if split[i] == "train" and any(split_of(s) != "train" for s in {f[0] for f in facts}):
            funnel[f"{fam}:mixed_split"] += 1; return
        facts = [list(f) for f in facts]
        rng.shuffle(facts)
        it = {"family": fam, "task": task, "entity": B.name(i), "relation": asked, "facts": facts, "gold": gold,
              "forbid": forbid, "split": split[i], "origin": origin(g.events[i]), **(extra or {})}
        reason = check(it, q, a)
        if reason:
            funnel[f"{fam}:{reason}"] += 1; return
        funnel[f"kept:{fam}"] += 1
        kept.append(dict(it, question=q, answer=a))
        del subjects

    for i in g.events:
        x = B.name(i)
        own = B.own(i)
        causes = [(a, r) for a, r in g.into(i, FORWARD)] + [(b, "response to*") for r, b in g.out(i, ("response to",))]
        effects = [(r, b) for r, b in g.out(i, FORWARD)] + [("response to*", a) for a, r in g.into(i, ("response to",))]
        if causes:
            lines = own + [B.link(a, r, i) if r != "response to*" else B.link(i, "response to", a) for a, r in causes]
            lines += [f for a, _ in causes for f in B.own(a, where=False, who=False)] + distractor(i)
            clauses = [says(B.name(a), r, x) if r != "response to*" else says(x, "response to", B.name(a)) for a, r in causes]
            keep("hist_cause", i, rng.choice(CAUSE_Q).format(x=the(x)), join(clauses), lines, [B.name(a) for a, _ in causes],
                 [], "cause")
        if effects:
            lines = own + [B.link(i, r, b) if r != "response to*" else B.link(b, "response to", i) for r, b in effects]
            lines += [f for _, b in effects for f in B.own(b, where=False, who=False)] + distractor(i)
            clauses = [says(x, r, B.name(b)) if r != "response to*" else says(B.name(b), "response to", x) for r, b in effects]
            keep("hist_effect", i, rng.choice(EFFECT_Q).format(x=the(x)), join(clauses), lines, [B.name(b) for _, b in effects],
                 [], "effect")
        down = g.downstream(i, limit=64)                     # a record uses at most 6 of them
        if len(down) >= 2:
            lines, clauses, seen = list(own), [], {i}
            for cur in [i] + down:
                for r, b in g.out(cur, FORWARD):
                    if b in down and (cur, r, b) not in seen:
                        seen.add((cur, r, b)); lines.append(B.link(cur, r, b)); clauses.append(says(B.name(cur), r, B.name(b)))
                for a, r in g.into(cur, ("response to", "part of")):
                    if a in down and (a, r, cur) not in seen:
                        seen.add((a, r, cur)); lines.append(B.link(a, r, cur)); clauses.append(says(B.name(a), r, B.name(cur)))
            if clauses and len(clauses) <= 6:
                keep("hist_downstream", i, f"What would change if {the(x)} had not happened?",
                     join(clauses)[:-1] + "; all of this would change without it.",
                     lines, [B.name(d) for d in down[:1]], [], "downstream")
        for c in [next((d for d in down[1:] if g.chain(i, d)), None)]:
            path = g.chain(i, c) if c else None
            if path and 3 <= len(path) <= 4:
                lines, clauses = list(own), []
                for a, b in zip(path, path[1:]):
                    r = next((r for r, t in g.out(a, FORWARD) if t == b), None)
                    if r is None:
                        r = "response to"; lines.append(B.link(b, r, a)); clauses.append(says(B.name(b), r, B.name(a)))
                    else:
                        lines.append(B.link(a, r, b)); clauses.append(says(B.name(a), r, B.name(b)))
                keep("hist_chain", i, f"How did {the(x)} lead to {the(B.name(c))}?", join(clauses), lines + distractor(i),
                     [B.name(c)], [], "chain", {"hops": len(path) - 1})
        for r, b in g.out(i, ("part of",)):
            keep("hist_part", i, f"What was {the(x)} part of?", join([says(x, "part of", B.name(b))]),
                 own + [B.link(i, r, b)] + B.own(b, where=False, who=False), [B.name(b)], [], "part of")
        w = B.when_says(i)
        if w:
            sent, vals = w
            if g.events[i].when.get("day"):
                sent = sent.replace(vals[0], day_words(vals[0]))
            keep("hist_when", i, rng.choice(WHEN_Q).format(x=the(x)), sent, own + distractor(i), vals, [], "date")
            other = rng.choice(dated) if dated else None
            if other and other != i and not (g.events[other].when["y0"] <= g.events[i].when["y1"]
                                             and g.events[i].when["y0"] <= g.events[other].when["y1"]):
                first, second = sorted([i, other], key=lambda j: g.events[j].when["y0"])
                s1, s2 = g.events[first].span(), g.events[second].span()
                pair = [x, B.name(other)]
                rng.shuffle(pair)
                keep("hist_order", i, f"Which came first, {the(pair[0])} or {the(pair[1])}?",
                     cap(f"{the(B.name(first))} came first ({s1}); {the(B.name(second))} came later ({s2})."),
                     B.own(i, where=False, who=False) + B.own(other, where=False, who=False), [s1], [], "order")
        if g.events[i].where and not any("," in p for p in g.events[i].where[:1]):
            keep("hist_where", i, rng.choice(WHERE_Q).format(x=the(x)), cap(f"{the(x)} took place in {g.events[i].where[0]}."),
                 own, [g.events[i].where[0]], [], "location")
        if g.events[i].who:
            who = g.events[i].who[:4]
            keep("hist_who", i, rng.choice(WHO_Q).format(x=the(x)),
                 f"The participant{'s' if len(who) > 1 else ''} in {the(x)} {'were' if len(who) > 1 else 'was'} {listing(who)}.",
                 own, who, [], "participant")
    # the newspaper reports far more dated events than the books: it may not crowd the books out of these families
    for fam in ("hist_when", "hist_where", "hist_who", "hist_order"):
        news = [k for k, r in enumerate(kept) if r["family"] == fam and r["origin"] == "news"]
        books = sum(1 for r in kept if r["family"] == fam and r["origin"] == "book")
        room = int(books * NEWS_SHARE / (1 - NEWS_SHARE))
        if len(news) > room:
            drop = set(rng.sample(news, len(news) - room))
            kept = [r for k, r in enumerate(kept) if k not in drop]
            funnel[f"{fam}:news_share"] += len(drop)
    fact_keys = list(g.facts)
    for (e, r, v), _src in list(g.facts.items()):
        if r not in FACT_SAY:
            funnel["hist_fact:relation"] += 1; continue
        s = split_of(e)
        it = {"family": "hist_fact", "task": "ask", "entity": e, "relation": r,
              "facts": [[e, r, v]] + [[e2, r2, v2] for (e2, r2, v2) in rng.sample(fact_keys, min(2, len(fact_keys)))
                                      if e2 != e and split_of(e2) == s],
              "gold": [v], "forbid": [], "split": s, "origin": "book"}
        qt, at = FACT_SAY[r]
        on = "on" if any(m.lower() in v.lower() for m in MONTHS) else "in"
        q, a = qt.format(e=e, v=v), at.format(e=e, v=v, on=on)
        a = a[:1].upper() + a[1:]
        reason = check(it, q, a)
        funnel[f"hist_fact:{reason}" if reason else "kept:hist_fact"] += 1
        if not reason:
            kept.append(dict(it, question=q, answer=a))
    # absent: the lines that answer it taken out, the rest kept
    asks = [r for r in kept if r["family"] in ("hist_cause", "hist_effect", "hist_when", "hist_fact")]
    absent = []
    for r in rng.sample(asks, int(absent_share * len(asks))):
        gold = {norm(x) for x in r["gold"]}
        if r["family"] == "hist_cause":
            f = [x for x in r["facts"] if not (norm(x[2]) == norm(r["entity"]) and x[1] in FORWARD)
                 and not (norm(x[0]) == norm(r["entity"]) and x[1] == "response to")]
        elif r["family"] == "hist_effect":
            f = [x for x in r["facts"] if not (norm(x[0]) == norm(r["entity"]) and x[1] in FORWARD)
                 and not (norm(x[2]) == norm(r["entity"]) and x[1] == "response to")]
        elif r["family"] == "hist_when":
            f = [x for x in r["facts"] if not (norm(x[0]) == norm(r["entity"]) and x[1] in ("date", "year", "start year", "end year"))]
        else:
            f = [x for x in r["facts"] if not (norm(x[0]) == norm(r["entity"]) and x[1] == r["relation"])]
        f = [x for x in f if norm(x[2]) not in gold]
        if not f or any(g_ and f" {g_} " in f" {norm(x[2])} " for x in f for g_ in gold):
            funnel["hist_absent:answer_left"] += 1; continue
        absent.append(dict(r, family="hist_absent", from_family=r["family"], facts=f, answer=ABSENT, gold=None,
                           forbid=list(r["gold"]) + r["forbid"]))
    return kept + absent, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default=str(ROOT / "standin" / "data" / "out" / "history_graph.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent-share", type=float, default=0.3)
    ap.add_argument("--per-event", type=int, default=1)
    ap.add_argument("--max-per-family", type=int, default=50000, help="records kept a family (seeded sample); 0 = all")
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "history_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "history_sft.manifest.json"))
    args = ap.parse_args()
    t0 = time.time()
    g = HistoryGraph.load(args.graph)
    records, funnel = build(g, random.Random(args.seed), args.absent_share, args.per_event)
    out, seen = [], set()
    for r in records:
        rid = hashlib.sha1(f"{r['family']}|{r['entity']}|{r['question']}|{render(r['facts'], r['question'])}".encode()).hexdigest()[:16]
        if rid in seen:
            continue
        seen.add(rid)
        out.append({"id": f"hist:{rid}", "family": r["family"], "split": r["split"], "prompt": render(r["facts"], r["question"]),
                    "answer": r["answer"], "entity": r["entity"], "relation": r["relation"], "gold": r["gold"],
                    "forbid": r["forbid"], "origin": r.get("origin", "book"), "source": SOURCE,
                    **({"from_family": r["from_family"]} if "from_family" in r else {})})
    if args.max_per_family:                                   # a mix draws thousands a family: the file need not hold millions
        by_fam, cap_rng = collections.defaultdict(list), random.Random(args.seed + 1)
        for r in out:
            by_fam[r["family"]].append(r)
        capped = {f: len(rs) for f, rs in by_fam.items() if len(rs) > args.max_per_family}
        out = [r for f, rs in by_fam.items()
               for r in (cap_rng.sample(rs, args.max_per_family) if f in capped else rs)]
        funnel.update({f"capped:{f}": n - args.max_per_family for f, n in capped.items()})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = collections.Counter(f"{r['family']}:{r['split']}" for r in out)
    manifest = {"source": SOURCE, "graph": pathlib.Path(args.graph).name, "events": len(g.events), "links": len(g.links),
                "facts": len(g.facts), "records": len(out), "by_family": dict(sorted(fams.items())),
                "by_origin": dict(collections.Counter(f"{r['family']}:{r['origin']}" for r in out).most_common()),
                "funnel": dict(funnel.most_common()), "wall_s": round(time.time() - t0, 1)}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
