"""The New York Times archive -> dated events for the history graph (standin/history_graph.py). Every article
carries its publication day; the events it reports get their day from it, and the book events of the same name
get theirs through the graph's merge (a book says "the attack on Pearl Harbor", the paper dates it 1941-12-07).

    python standin/data/build_nyt_events.py --src <folder of yyyy_m.json> [--months 3] [--front-page] [--workers 64]
        [--model qwen/qwen3-235b-a22b-2507] [--max-cost 50] [--tag _pilot]

Source (listed for the owner's disclosure): the local datasets drive's `domains/temporal_timelines_events`
monthly New York Times archive files (`<year>_<month>.json`, 1851-2024, ~2M articles: headline, abstract, lead
paragraph, publication date, the paper's own keywords for people, organisations and places). Opinion, letters,
corrections, notices, listings and media pages are skipped. A model reads the articles in batches and names the
event each reports; dataset building only, never serving. The host keeps an event only if:
  - its name's proper nouns are the article's (headline, abstract or lead),
  - its date is the publication day or the day before (what a daily paper reports), or a date the article states
    (its numbers in the article), never later than the publication day,
  - its places and people are the article's (text or the paper's keywords).
Kept: the event, its day, places, people, kind and the article id -- never the article's text.
"""
from __future__ import annotations

import argparse, collections, concurrent.futures as cf, datetime as dt, glob, hashlib, json, os, pathlib, random, re, sys
import threading, time, urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from build_history_graph import _SENTENCE, _s, has, is_name, norm, proper_words, reasoning_of  # noqa: E402

SOURCE = "datasets drive: domains/temporal_timelines_events/<year>_<month>.json (New York Times archive)"
SKIP = {"Op-Ed", "Letter", "Editorial", "Correction", "Paid Death Notice", "Statistics", "Schedule", "Slideshow", "Video",
        "Interactive Feature", "Review", "Recipe", "Quote", "List", "briefing", "Web Log", "Blog", "Question",
        "Text", "Audio", "Sidebar", "Addendum", "An Appraisal", "Caption", "Chronology", "Marriage Announcement",
        "Paid Notice", "Special Report", "Series", "Glossary", "Editors' Note"}
SYSTEM = """You read numbered newspaper articles (publication date, headline, summary) and name the news event each reports.
Reply with ONLY one JSON object: {"items": [...]}, one entry per article that reports an event:
  {"i": the article's number,
   "name": the event as a historian would name it -- a capitalised noun phrase built from the article's own names
     ("Sinking of the Titanic", "Assassination of President McKinley", "Opening of the Brooklyn Bridge", "Election of
     Grover Cleveland", "Death of Babe Ruth"), never a sentence and never the headline copied,
   "when": the day the event happened as YYYY-MM-DD: the publication day, or the day before when the article reports
     "yesterday"/"last night", or a date the article states; never a day after publication,
   "where": [places, as the article writes them], "who": [people and organisations that took part, as written],
   "kind": one lowercase word (battle, war, election, death, disaster, crime, law, treaty, strike, launch, discovery,
     sport, business, protest, trial, speech, appointment, other)}
Skip articles that report no event (announcements, advice, sports results of minor games, notices, listings, essays).
Only what the article says; no outside knowledge."""


def iter_articles(src: str, months: int, front_page: bool, seed: int):
    files = sorted(glob.glob(os.path.join(src, "[0-9][0-9][0-9][0-9]_*.json")))
    if months:
        files = sorted(random.Random(seed).sample(files, min(months, len(files))))
    for f in files:
        try:
            docs = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for d in docs:
            if str(d.get("type_of_material") or "") in SKIP:
                continue
            if front_page and str(d.get("print_page")) != "1":
                continue
            pub = str(d.get("pub_date") or "")[:10]
            head = _s((d.get("headline") or {}).get("main"))
            text = " ".join(x for x in (_s(d.get("abstract")), _s(d.get("lead_paragraph"))) if x)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pub) or not head:
                continue
            kw = collections.defaultdict(list)
            for k in d.get("keywords") or []:
                if k.get("name") in ("persons", "organizations", "glocations", "subject"):
                    kw[k["name"]].append(_s(k.get("value")))
            yield {"id": str(d.get("_id") or d.get("uri") or hashlib.sha1((pub + head).encode()).hexdigest()),
                   "pub": pub, "head": head, "text": text[:700], "page": str(d.get("print_page") or ""),
                   "people": kw["persons"], "orgs": kw["organizations"], "places": kw["glocations"], "file": os.path.basename(f)}


def check_items(reply: str, batch: list[dict], funnel: collections.Counter) -> list[dict]:
    s = reply.strip()
    i, j = s.find("{"), s.rfind("}")
    try:
        items = json.loads(s[i:j + 1]).get("items") or [] if i >= 0 and j > i else []
    except Exception:
        items = []
    out = []
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("i"), int) or not 0 <= it["i"] < len(batch):
            funnel["item:bad_index"] += 1; continue
        a = batch[it["i"]]
        hay = norm(" ".join([a["head"], a["text"]] + a["people"] + a["orgs"] + a["places"]))
        name = _s(it.get("name"))
        if not name or not name[:1].isupper() or not is_name(name) or _SENTENCE.match(name) or len(name) > 100:
            funnel["item:not_a_name"] += 1; continue
        pw = proper_words(name)
        if not pw or not all(has(hay, w) for w in pw):
            funnel["item:name_not_grounded"] += 1; continue
        when = _s(it.get("when"))
        pub = dt.date.fromisoformat(a["pub"])
        try:
            day = dt.date.fromisoformat(when)
        except ValueError:
            funnel["item:bad_date"] += 1; continue
        text_n = norm(a["head"] + " " + a["text"])
        stated = str(day.day) in text_n.split() and (str(day.year) in text_n.split() or day.year == pub.year)
        if day > pub or not ((pub - day).days <= 1 or stated):
            funnel["item:date_not_supported"] += 1; continue
        where = [x for x in map(_s, it.get("where") or []) if x and has(hay, x)][:3]
        who = [x for x in map(_s, it.get("who") or []) if x and has(hay, x) and is_name(x)][:6]
        kind = _s(it.get("kind")).lower()
        out.append({"name": name, "day": day.isoformat(), "pub": a["pub"], "where": where, "who": who,
                    "kind": kind if re.fullmatch(r"[a-z]{2,20}", kind) else "other", "article": a["id"], "file": a["file"],
                    "front_page": a["page"] == "1"})
        funnel["item:kept"] += 1
    return out


def prompt_for(batch: list[dict]) -> str:
    rows = []
    for n, a in enumerate(batch):
        rows.append(f"[{n}] published {a['pub']} | {a['head']}" + (f" | {a['text']}" if a["text"] else ""))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="the folder holding the <year>_<month>.json files")
    ap.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--months", type=int, default=0, help="a pilot: this many months (seeded); 0 = all")
    ap.add_argument("--front-page", action="store_true", help="page-one articles only")
    ap.add_argument("--limit", type=int, default=0, help="a pilot: at most this many articles")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-cost", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--reasoning", default="", help="a hybrid model's thinking: off | low | medium | high ('' = default)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from openrouter import OpenRouterProposer
    t0 = time.time()
    arts = list(iter_articles(args.src, args.months, args.front_page, args.seed))
    if args.limit:
        arts = arts[:args.limit]
    batches = [arts[i:i + args.batch] for i in range(0, len(arts), args.batch)]
    random.Random(args.seed).shuffle(batches)          # a run stopped by --max-cost covers every decade, not the first ones
    print(f"{len(arts):,} articles in {len(batches):,} calls ({time.time() - t0:.0f}s)", flush=True)
    local, lock, stop = threading.local(), threading.Lock(), threading.Event()
    usage, funnel = collections.Counter(), collections.Counter()

    def run(batch):
        if not hasattr(local, "llm"):
            local.llm = OpenRouterProposer(args.model, system=SYSTEM, max_tokens=4000, reasoning=reasoning_of(args.reasoning))
        llm = local.llm
        llm.offline = args.offline or stop.is_set()
        before, reply = dict(llm.usage), ""
        for attempt in range(6):
            try:
                reply = llm.chat(prompt_for(batch))
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 402, 403):
                    stop.set(); llm.offline = True
                    break
                time.sleep(min(60, 3 * 2 ** attempt))
            except Exception:                                        # noqa: BLE001 -- timeouts, resets: retried
                time.sleep(min(60, 3 * 2 ** attempt))
        local_funnel = collections.Counter()
        events = check_items(reply, batch, local_funnel)
        with lock:
            for k in ("prompt_tokens", "completion_tokens", "cost"):
                usage[k] += llm.usage[k] - before[k]
            if usage["cost"] > args.max_cost:
                stop.set()
            funnel.update(local_funnel)
            funnel["articles"] += len(batch)
        return events

    out, done = [], 0
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for events in pool.map(run, batches):
            out += events
            done += 1
            if done % 200 == 0 or done == len(batches):
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done:,}/{len(batches):,} calls, {len(out):,} events, ${usage['cost']:.2f}, "
                      f"eta {(len(batches) - done) / max(rate, 1e-9) / 60:.0f} min{' [STOPPED]' if stop.is_set() else ''}", flush=True)
    path = ROOT / "standin" / "data" / "out" / f"nyt_events{args.tag}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    kinds = collections.Counter(e["kind"] for e in out)
    decades = collections.Counter(e["day"][:3] + "0s" for e in out)
    manifest = {"source": SOURCE, "model": args.model, "articles": len(arts), "calls": len(batches), "events": len(out),
                "front_page_only": args.front_page, "funnel": dict(funnel.most_common()), "usage": dict(usage),
                "stopped": stop.is_set(), "kinds": dict(kinds.most_common()), "decades": dict(sorted(decades.items())),
                "sample": [f"{e['day']}  {e['name']}  ({', '.join(e['where'][:1] + e['who'][:2])})" for e in out[:30]],
                "wall_s": round(time.time() - t0, 1)}
    (ROOT / "validation" / "logs" / f"nyt_events{args.tag}.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("articles", "events", "funnel", "usage", "wall_s")}, indent=1))
    print("\n".join(manifest["sample"]))


if __name__ == "__main__":
    main()
