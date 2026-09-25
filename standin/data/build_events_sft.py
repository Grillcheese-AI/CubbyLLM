"""Dated facts for the talk adapter (H-E6 families over history): years, regions and participants of
historical events, in the VM's facts-block form. The wikikg families rarely ask for a year; a timeline
answer is a date, and binding the right date to the right event is where a small model slips.

    python standin/data/build_events_sft.py --src <datasets dir> [--seed 0]

Source (listed for the owner's disclosure): the local datasets drive's
`domains/verified_facts/historical_facts.jsonl` and `domains/temporal_timelines_events/train_augmented.jsonl`
(the same ~4k events, the second with periods added), merged by title and start year. Each event gives
`title`, `year_start`/`year_end` (negative = BC), `region`, `actors`, `categories`. The event's `text` and
`impact` prose is NOT used: every answer is built from the lines the block shows.

Lines: `<title> — year: 1879` (or `start year` / `end year`), `region`, one `participant` per actor, and
one `category`. Families, the build_ground_sft contract:
  event_relation  when / where / who, about one event -> the sentence that states it
  event_bind      2-3 events' dates in one block -> the asked one's, none of the others'
  event_counter   the asked event carries another event's date -> the date the facts give
  event_profile   "Tell me about X." -> one sentence from its year, region and participants
  event_absent    a relation or bind record with the asked lines taken out -> "The facts don't say."
Every record passes build_ground_sft.check. Split by title hash (held = 10%).
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_ground_sft import ABSENT, check, norm, render, split_of  # noqa: E402

SOURCE = "datasets drive: domains/verified_facts/historical_facts.jsonl + temporal_timelines_events/train_augmented.jsonl"
FILES = ("domains/verified_facts/historical_facts.jsonl", "domains/temporal_timelines_events/train_augmented.jsonl")
WHEN = ["When did {t} happen?", "What year was {t}?", "When was {t}?", "In what year did {t} take place?",
        "What is the date of {t}?"]
WHERE = ["Where did {t} take place?", "In which region did {t} happen?", "Where was {t}?"]
WHO = ["Who was involved in {t}?", "Who took part in {t}?", "Which people are linked to {t}?"]
ABOUT = ["Tell me about {t}.", "What was {t}?", "What do you know about {t}?"]


def year(y) -> str:
    y = int(y)
    return f"{-y} BC" if y < 0 else str(y)


def listing(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" and {xs[-1]}"


def load(src: pathlib.Path):
    events, seen = [], set()
    for rel in FILES:
        for line in open(src / rel, encoding="utf-8"):
            try:
                r = json.loads(line, strict=False)
            except Exception:
                continue
            t, y0 = " ".join(str(r.get("title") or "").split()), r.get("year_start")
            if not t or y0 is None or (norm(t), y0) in seen:
                continue
            seen.add((norm(t), y0))
            actors = [" ".join(str(a).split()) for a in (r.get("actors") or []) if str(a).strip()][:4]
            cats = [str(c) for c in (r.get("categories") or []) if str(c).strip()]
            events.append({"title": t, "y0": int(y0), "y1": int(r["year_end"]) if r.get("year_end") is not None else None,
                           "region": " ".join(str(r.get("region") or "").split()), "actors": actors,
                           "category": cats[0] if cats else ""})
    return events


def lines(ev) -> list[list[str]]:
    t, out = ev["title"], []
    if ev["y1"] is not None and ev["y1"] != ev["y0"]:
        out += [[t, "start year", year(ev["y0"])], [t, "end year", year(ev["y1"])]]
    else:
        out.append([t, "year", year(ev["y0"])])
    if ev["region"]:
        out.append([t, "region", ev["region"]])
    out += [[t, "participant", a] for a in ev["actors"]]
    if ev["category"]:
        out.append([t, "category", ev["category"]])
    return out


def when(ev, ys=None) -> tuple[str, list[str]]:
    """The date sentence and its values. ys overrides the event's own (the counter family)."""
    y0, y1 = ys or (year(ev["y0"]), year(ev["y1"]) if ev["y1"] is not None and ev["y1"] != ev["y0"] else None)
    if y1:
        return f"{ev['title']} ran from {y0} to {y1}.", [y0, y1]
    return f"{ev['title']} happened in {y0}.", [y0]


DATE_RELS = ("year", "start year", "end year")


def build(events, rng: random.Random, absent_share: float, counter_share: float, bind_per_event: int):
    funnel, kept = collections.Counter(), []
    split = {ev["title"]: split_of(ev["title"]) for ev in events}
    by_split = collections.defaultdict(list)
    for ev in events:
        by_split[split[ev["title"]]].append(ev)

    def keep(fam, ev, q, a, facts, gold, forbid, asked, task="ask"):
        it = {"family": fam, "task": task, "entity": ev["title"], "relation": asked, "facts": [list(x) for x in facts],
              "gold": gold, "forbid": forbid, "split": split[ev["title"]]}
        reason = check(it, q, a)
        if reason:
            funnel[f"{fam}:{reason}"] += 1; return
        funnel[f"kept:{fam}"] += 1
        kept.append(dict(it, question=q, answer=a))

    def other(ev, n=1, cond=lambda o: True):
        pool, out = by_split[split[ev["title"]]], []
        for _ in range(40):
            o = rng.choice(pool)
            if o is not ev and norm(o["title"]) != norm(ev["title"]) and cond(o) and all(o is not x for x in out):
                out.append(o)
            if len(out) == n:
                break
        return out

    for ev in events:
        t, own = ev["title"], lines(ev)
        extra = [x for o in (other(ev) if rng.random() < 0.3 else []) for x in lines(o)]
        block = own + extra
        # when
        a, ys = when(ev)
        keep("event_relation", ev, rng.choice(WHEN).format(t=t), a, rng.sample(block, len(block)), ys, [], "year")
        # where (a region that reads as a place)
        if ev["region"] and "global" not in ev["region"].lower() and "," not in ev["region"]:
            keep("event_relation", ev, rng.choice(WHERE).format(t=t), f"{t} took place in {ev['region']}.",
                 rng.sample(block, len(block)), [ev["region"]], [], "region")
        # who
        if ev["actors"]:
            a = (f"The participant in {t} was {ev['actors'][0]}." if len(ev["actors"]) == 1 else
                 f"The participants in {t} were {listing(ev['actors'])}.")
            keep("event_relation", ev, rng.choice(WHO).format(t=t), a, rng.sample(block, len(block)), ev["actors"],
                 [], "participant")
        # profile: its date, region and participants, nothing else
        d, _ = when(ev)
        body = d[:-1]
        if ev["region"] and "global" not in ev["region"].lower() and "," not in ev["region"]:
            body += f" in {ev['region']}"
        if ev["actors"]:
            body += f", and its participants {'was' if len(ev['actors']) == 1 else 'were'} {listing(ev['actors'])}"
        keep("event_profile", ev, rng.choice(ABOUT).format(t=t), body + ".", rng.sample(own, len(own)), None, [],
             None, task="profile")
        # bind: 2-3 events' dates, the question about one
        for _ in range(bind_per_event):
            mine = set(when(ev)[1])
            partners = other(ev, rng.choice([1, 1, 2]), lambda o: not (set(when(o)[1]) & mine))
            if not partners:
                funnel["event_bind:no_partner"] += 1; continue
            blk = [x for x in own if x[1] in DATE_RELS or rng.random() < 0.4]
            for o in partners:
                blk += [x for x in lines(o) if x[1] in DATE_RELS or rng.random() < 0.3]
            rng.shuffle(blk)
            a, ys = when(ev)
            forbid = [y for o in partners for y in when(o)[1]] + [o["title"] for o in partners]
            keep("event_bind", ev, rng.choice(WHEN).format(t=t), a, blk, ys, forbid, "year")
        # counter: another event's single year in place of this one's
        if ev["y1"] is None and rng.random() < counter_share:
            donor = other(ev, 1, lambda o: o["y1"] is None and abs(o["y0"] - ev["y0"]) > 30)
            if donor:
                fake, real = year(donor[0]["y0"]), year(ev["y0"])
                blk = [[t, "year", fake] if x[1] == "year" else x for x in own]
                keep("event_counter", ev, rng.choice(WHEN).format(t=t), f"{t} happened in {fake}.",
                     rng.sample(blk, len(blk)), [fake], [real], "year")
    # absent: the asked lines out (all date lines for a date question), the rest kept
    asks = [r for r in kept if r["task"] == "ask" and r["family"] != "event_counter"]
    absent = []
    for r in rng.sample(asks, int(absent_share * len(asks))):
        drop = DATE_RELS if r["relation"] in DATE_RELS else (r["relation"],)
        gold = {norm(g) for g in r["gold"]}
        f = [x for x in r["facts"] if not (x[0] == r["entity"] and x[1] in drop) and norm(x[2]) not in gold]
        if not f or any(g and f" {g} " in f" {norm(x[2])} " for x in f for g in gold):
            funnel["event_absent:answer_left"] += 1; continue
        absent.append(dict(r, family="event_absent", from_family=r["family"], facts=f, answer=ABSENT, gold=None,
                           forbid=list(r["gold"]) + r["forbid"]))
    return kept + absent, funnel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the datasets folder holding domains/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent-share", type=float, default=0.3)
    ap.add_argument("--counter-share", type=float, default=0.6)
    ap.add_argument("--bind-per-event", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "standin" / "data" / "out" / "events_sft.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "validation" / "logs" / "events_sft.manifest.json"))
    args = ap.parse_args()
    t0 = time.time()
    events = load(pathlib.Path(args.src))
    records, funnel = build(events, random.Random(args.seed), args.absent_share, args.counter_share, args.bind_per_event)
    out, ids = [], collections.Counter()
    for r in records:
        rid = hashlib.sha1(f"{r['family']}|{r['entity']}|{r['question']}|{r['prompt'] if 'prompt' in r else ''}|"
                           f"{render(r['facts'], r['question'])}".encode()).hexdigest()[:16]
        ids[rid] += 1
        if ids[rid] > 1:
            continue
        out.append({"id": f"event:{rid}", "family": r["family"], "split": r["split"],
                    "prompt": render(r["facts"], r["question"]), "answer": r["answer"], "entity": r["entity"],
                    "relation": r["relation"], "gold": r["gold"], "forbid": r["forbid"], "source": SOURCE,
                    **({"from_family": r["from_family"]} if "from_family" in r else {})})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams = collections.Counter(f"{r['family']}:{r['split']}" for r in out)
    manifest = {"source": SOURCE, "events": len(events), "records": len(out), "by_family": dict(sorted(fams.items())),
                "funnel": dict(funnel.most_common()), "wall_s": round(time.time() - t0, 1)}
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
