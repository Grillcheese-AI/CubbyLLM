"""History books -> the history graph (standin/history_graph.py): events in time, linked by cause and precursor,
plus plain facts about the people and places in them. The base of history the worlds branch from; the talk
adapter's history families are built from it (talk_v2.1).

    python standin/data/build_history_graph.py --src <books dir> [--src <another>] [--books 4] [--workers 48]
        [--model qwen/qwen3-235b-a22b-2507] [--max-cost 60] [--tag _pilot]

Sources (listed for the owner's disclosure): the local datasets drive's `domains/history` (~1,080 history books
as text: Project Gutenberg texts with their `<id>_metadata.json`, and books converted from epub/pdf) and
`domains/verified_facts` (~4,300 books on prehistory, the ancient and classical world and the Middle Ages, as
text). A model reads each passage and lists what it states; dataset building only, never serving (Nick,
2026-09-12). Nothing it writes is kept unless the host finds it in the passage:
  event  its name occurs in the passage and has a capitalised word; its quote occurs in the passage; its date's
         numbers occur in the quote (else the event is kept undated); each place and participant occurs in the
         passage (else that one is dropped)
  link   both ends are events kept from the same passage; the relation is one of history_graph.RELATIONS; the
         quote occurs in the passage and names both ends (a word of each); and the graph refuses a link whose
         dates contradict it (a cause dated after its effect)
  fact   entity and quote in the passage, value in the quote (a date: its numbers), entity a name, not the value
Kept: the event, link or fact and where it came from (book, passage number) -- never the passage or the quote.

Books are deduplicated first (both folders hold books twice: .txt and .epub.txt, "(Retail)" and plain):
content-defined 8-word shingles over a window of each book, one kept of any pair sharing over a third. A passage
(~3,000 words, cut at paragraph ends) goes to the model only if it holds at least two dates or date words --
a dictionary page, a poem or a theory chapter does not. Replies are cached by (model, prompt), so a stopped run
resumes for free; `--max-cost` stops calling the model past the budget.
"""
from __future__ import annotations

import argparse, collections, concurrent.futures as cf, hashlib, json, os, pathlib, random, re, sys, threading, time
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "standin", ROOT / "standin" / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__wiring__ = "STANDALONE"

from history_graph import RELATIONS, HistoryGraph, key, parse_time  # noqa: E402

NEWS_COLUMN_DAYS = 5          # a newspaper "event" name printed on this many days is a column title

SOURCE = "datasets drive: domains/history + domains/verified_facts (history books as text)"
FACT_RELATIONS = ("birth date, death date, birthplace, place of death, position held, spouse, parent, child, sibling, "
                  "predecessor, successor, founder, capital, author, member of, occupation, nationality, ruler, located in")
SYSTEM = f"""You read one passage of a history book and write what it states as a small history graph.
Reply with ONLY one JSON object: {{"events": [...], "links": [...], "facts": [...]}}.

events: the historical events the passage tells of -- battles, wars, campaigns, sieges, treaties, reigns, revolutions,
  rebellions, elections, councils, foundings, laws, voyages, discoveries, deaths of rulers.
  {{"name": the name historians use for it ("Battle of Resaca", "Atlanta Campaign", "Siege of Alesia", "Treaty of Verdun"),
      even where the passage words it differently ("the fight at Resaca") -- but its proper nouns must be the passage's.
      Always a capitalised noun phrase, never a sentence: "Approval of the Thirteenth Amendment", not "Thirteenth
      Amendment approved"; "Sherman's Capture of Savannah", not "Sherman reaches Savannah".
      Skip small actions with no such name (a march, a meeting, an order), unless the passage says one caused a named event,
    "when": its date or start WITH the year ("July 1, 1863", "1066", "44 BC", "the 1790s"). Every event needs one:
      if the passage dates this event, give that date; if it does not, give the year the passage's own context
      places it in (the year the passage is telling about, as the passage writes it). Add "BC" where the passage
      counts years BC even if it drops the letters ("133" in a passage about 133 BC -> "133 BC").
      "" only if the passage gives no year at all,
    "basis": "stated" if the passage dates this event itself, "context" if the year comes from the passage's context,
    "end": its end date as the passage states it, or "",
    "where": [places as written], "who": [people, peoples, states or armies that took part, as written],
    "kind": one lowercase word (battle, war, campaign, siege, treaty, reign, revolution, election, council, founding, law, voyage, discovery, death, other),
    "q": one contiguous span copied exactly from the passage (no "..."), at most 20 words, that names the event}}
links: how one event bears on another, ONLY where the passage says so ("led to", "provoked", "as a result of", "in response to", "paved the way for", "was part of", "ended").
  {{"from": a name from your events list, "rel": exactly one of: {", ".join(RELATIONS)},
    "to": a name from your events list, "q": one contiguous span copied exactly, at most 30 words, that says it}}
  Meanings: caused / contributed to / precursor of -- "from" came first and brought about or set up "to";
  response to -- "from" was a response to "to"; part of -- "from" was part of "to"; ended -- "from" ended "to".
facts: other facts about named people, places and states: {{"e": the entity by its fullest name in the passage
  ("Andrew Johnson", not "Johnson"), "r": exactly one of: {FACT_RELATIONS}, "v": one value exactly as written,
  "q": one contiguous span copied exactly, at most 20 words}}.

Rules: only what the passage states as true -- no opinions, hypotheses, rumours, disputed claims or outside knowledge.
Every "q" is one contiguous span copied character for character from the passage, never shortened with "...".
At most 15 events, 15 links and 15 facts; prefer the events that carry dates and links.
A table of contents, index, bibliography, or a passage with nothing of this kind gives {{"events": [], "links": [], "facts": []}}.
A passage in French or another language: relation and kind in English; names, dates and places exactly as written."""

_STOP = set("the of and a an in on at to for by with from de la le les du des et his her its their".split())
# the words a historian's name for an event adds to the passage's proper nouns ("Battle of", "Siege of", "Treaty of")
_GENERIC = set("battle siege war wars treaty campaign council congress revolution rebellion revolt uprising massacre conquest "
               "fall sack invasion assassination coronation death founding foundation reign peace edict act crisis expedition "
               "voyage election first second third fourth great battles sieges civil holy crusade dynasty empire kingdom "
               "republic period era age affair incident raid march retreat defeat victory surrender capture destruction "
               "rise collapse division partition union restoration reformation council synod diet".split())
_DATEWORD = re.compile(r"\b(\d{3,4}|\d{1,4}\s*(?:bc|b\.c\.|bce|ad|a\.d\.|ce)|centur(?:y|ies)|siècle|january|february|march|"
                       r"april|may|june|july|august|september|october|november|december|janvier|février|avril|juin|"
                       r"juillet|août|septembre|octobre|novembre|décembre)\b", re.I)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def has(hay_n: str, needle: str) -> bool:
    n = norm(needle)
    return bool(n) and f" {n} " in f" {hay_n} "


# ---------------------------------------------------------------- books
def list_books(srcs: list[str]) -> list[tuple[str, str]]:
    out = []
    for src in srcs:
        for root, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith(".txt"):
                    p = os.path.join(root, f)
                    out.append((p, os.path.relpath(p, src).replace(os.sep, "/")))
    return sorted(out)


def book_meta(path: str, rel: str) -> dict:
    meta_path = path[:-4] + "_metadata.json"
    if os.path.exists(meta_path):
        try:
            m = json.load(open(meta_path, encoding="utf-8", errors="replace"))
            return {"title": m.get("title") or rel, "author": "; ".join(a.get("name", "") for a in m.get("authors") or []),
                    "lang": (m.get("languages") or ["en"])[0], "file": rel}
        except Exception:
            pass
    stem = re.sub(r"\s*\(v\d\.\d\)\.epub$|\.pdf$|__k\d+$", "", os.path.basename(path)[:-4]).replace("_", " ").strip()
    return {"title": stem, "author": "", "lang": "", "file": rel}


def clean(text: str) -> str:
    s, e = text.find("*** START OF"), text.find("*** END OF")
    if s >= 0:
        text = text[text.find("\n", s) + 1: e if e > s else None]
    return text.replace("\r\n", "\n")


def shingles(text: str) -> set:
    w = norm(text[20_000:140_000]).split()                      # a window past the front matter
    out = set()
    for i in range(max(0, len(w) - 8)):
        h = hashlib.blake2b(" ".join(w[i:i + 8]).encode(), digest_size=8).digest()
        if h[0] < 8:                                            # content-defined 1/32 sample: no alignment needed
            out.add(h)
    return out


def dedupe(books: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    sigs, sizes = {}, {}
    for p, rel in books:
        t = open(p, encoding="utf-8", errors="replace").read()
        sigs[rel], sizes[rel] = shingles(clean(t)), len(t)
    index = collections.defaultdict(set)
    for rel, s in sigs.items():
        for h in s:
            index[h].add(rel)
    dropped, pairs = set(), []
    for rel in sorted(sigs, key=lambda r: -sizes[r]):
        if rel in dropped or not sigs[rel]:
            continue
        shared = collections.Counter(q for h in sigs[rel] for q in index[h] if q != rel and q not in dropped)
        for q, c in shared.items():
            if c > len(sigs[q]) / 3 or c > len(sigs[rel]) / 3:
                dropped.add(q); pairs.append((rel, q))
    return [b for b in books if b[1] not in dropped], pairs


def _pieces(para: str, words: int):
    """A paragraph longer than a passage (a converted pdf can run pages together) in passage-sized pieces."""
    w = para.split()
    if len(w) <= words:
        yield para
        return
    for i in range(0, len(w), words):
        yield " ".join(w[i:i + words])


def passages(text: str, words: int = 3000) -> list[str]:
    out, cur, n = [], [], 0
    for block in re.split(r"\n\s*\n", text):
        for para in _pieces(" ".join(block.split()), words):
            if para:
                cur.append(para); n += len(para.split())
            if n >= words:
                out.append("\n\n".join(cur)); cur, n = [], 0
    if n >= 200:
        out.append("\n\n".join(cur))
    return out


# ---------------------------------------------------------------- the host's checks
def parse(reply: str) -> dict:
    s = reply.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        d = json.loads(s[i:j + 1])
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def _s(x) -> str:
    return " ".join(str(x or "").split())


def is_name(s: str) -> bool:
    return any(w[:1].isupper() for w in s.split())


# a subject then a verb: 'Sherman reaches Savannah', 'Thirteenth Amendment approved', 'Crassus raises private army'
_SENTENCE = re.compile(r"^(?:[A-Z][\w'’.\-]*\s+){1,4}(?!of\b|and\b|in\b|at\b|on\b|to\b|for\b|the\b|with\b|against\b)"
                       r"[a-z]+(?:ed|es|s)\b")
_BCWORD = re.compile(r"\b(?:B\.?\s?C\.?(?:E\.?)?|BCE|av\.?\s*J\.?-?C\.?)(?!\w)", re.I)
_BC_YEARS = re.compile(r"\b\d{1,4}\s*(?:B\.\s?C\.|BC\b|BCE\b|av\.?\s*J\.?-?C\.?)")
_AD_YEARS = re.compile(r"\b(?:A\.\s?D\.|AD|CE)\s*\d{1,4}\b|\b\d{1,4}\s*(?:A\.\s?D\.|AD\b|CE\b|ap\.?\s*J\.?-?C\.?)")
_BARE_YEAR = re.compile(r"^\s*(?:c\.\s*|circa\s+)?\d{1,3}(?:\s*[-–]\s*\d{1,3})?\s*$", re.I)
# a person-fact needs the person's full name: 'Johnson' alone is Andrew or Lyndon
_PERSON_RELS = {"birth date", "death date", "birthplace", "place of death", "position held", "spouse", "parent", "child",
                "sibling", "occupation", "nationality", "member of"}


# the same meaning in other words (a model's paraphrase of the six); anything else is refused
_REL_SAME = {"led to": "caused", "resulted in": "caused", "triggered": "caused", "provoked": "caused", "cause": "caused",
             "causes": "caused", "brought about": "caused", "contributed": "contributed to", "contributes to": "contributed to",
             "paved the way for": "precursor of", "precursor to": "precursor of", "set the stage for": "precursor of",
             "in response to": "response to", "responded to": "response to", "was part of": "part of",
             "is part of": "part of", "ends": "ended", "brought an end to": "ended", "concluded": "ended"}


def proper_words(name: str) -> list[str]:
    """The name's own proper nouns: capitalised words that are not the generic part of an event name."""
    out = []
    for w in re.findall(r"[^\W\d_][\w'’-]*", name):
        n = norm(w)
        if w[:1].isupper() and n and n not in _GENERIC and n not in _STOP and len(n) > 1:
            out.append(n)
    return out


def content_words(name: str) -> list[str]:
    return [w for w in norm(name).split() if len(w) > 2 and w not in _STOP] or norm(name).split()


def book_era(text: str) -> str:
    """'bc' for a book that counts years BC (Rubicon: '88 BC' often, 'AD' rarely), where a bare '107' is 107 BC;
    'ad' otherwise."""
    bc, ad = len(_BC_YEARS.findall(text)), len(_AD_YEARS.findall(text))
    return "bc" if bc >= 5 and bc > 3 * ad else "ad"


def check_passage(d: dict, text: str, funnel: collections.Counter, era: str = "ad") -> dict:
    """The model's graph for one passage -> what the host could find in the passage. `era`: the book's (book_era)."""
    pn = norm(text)
    events, names = [], set()
    for e in d.get("events") or []:
        if not isinstance(e, dict):
            continue
        name, q = _s(e.get("name")), _s(e.get("q"))
        if not name or len(name) > 100 or not is_name(name) or not name[:1].isupper():
            funnel["event:not_a_name"] += 1; continue
        if _SENTENCE.match(name):                                    # 'Sherman reaches Savannah' is a sentence, not a name
            funnel["event:sentence_not_a_name"] += 1; continue
        pw = proper_words(name)
        if not pw:
            funnel["event:no_proper_noun"] += 1; continue
        if not all(has(pn, w) for w in pw):                          # 'Battle of Resaca': 'Resaca' is the passage's
            funnel["event:name_not_grounded"] += 1; continue
        if not q or not has(pn, q):
            funnel["event:quote_not_in_passage"] += 1; continue
        qn = norm(q)
        at = f" {pn} ".find(f" {qn} ")
        if not any(has(pn[max(0, at - 300): at + len(qn) + 300], w) for w in pw):   # the name where the quote is
            funnel["event:quote_misses_name"] += 1; continue
        when, end = _s(e.get("when")), _s(e.get("end"))
        basis = "context" if _s(e.get("basis")).lower() == "context" else "stated"
        near = pn[max(0, at - 800): at + len(qn) + 800]              # a stated date: near where the event is told
        hint = ""                                                    # a year the passage does not hold: a hint only
        for label, hay in (("when", near if basis == "stated" else pn), ("end", pn)):   # a context year: in the passage
            val = when if label == "when" else end
            digits = re.findall(r"\d+", val)
            if val and not (has(hay, val) or (digits and all(f" {x} " in f" {hay} " for x in digits))):
                funnel[f"event:{label}_not_found"] += 1
                if label == "when":
                    hint = f"{when} BC" if era == "bc" and _BARE_YEAR.match(when) else when
                    when = ""
                else:
                    end = ""
            elif val and _BCWORD.search(val) and not _BCWORD.search(text) and era != "bc":   # 'BC' where the book counts BC
                funnel[f"event:{label}_bc_not_in_passage"] += 1
                when, end = (_BCWORD.sub("", when).strip(), end) if label == "when" else (when, _BCWORD.sub("", end).strip())
            elif val and era == "bc" and _BARE_YEAR.match(val):     # '88' in a book that counts BC and drops the letters
                funnel[f"event:{label}_read_as_bc"] += 1
                when, end = (f"{when} BC", end) if label == "when" else (when, f"{end} BC")
        where = [x for x in map(_s, e.get("where") or []) if x and has(pn, x)][:4]
        who = [x for x in map(_s, e.get("who") or []) if x and has(pn, x) and is_name(x)][:8]
        kind = _s(e.get("kind")).lower()
        events.append({"name": name, "when": when, "end": end, "basis": basis, "where": where, "who": who,
                       "kind": kind if re.fullmatch(r"[a-z]{2,20}", kind) else "other", **({"hint": hint} if hint else {})})
        funnel[f"event:dated_{basis}" if when else "event:undated"] += 1
        names.add(name)
        funnel["event:kept"] += 1
    links = []
    for l in d.get("links") or []:
        if not isinstance(l, dict):
            continue
        a, rel, b, q = _s(l.get("from")), _s(l.get("rel")).lower(), _s(l.get("to")), _s(l.get("q"))
        rel = _REL_SAME.get(rel, rel)
        if rel not in RELATIONS:
            funnel["link:relation"] += 1; continue
        if a not in names or b not in names or a == b:
            funnel["link:end_not_an_event"] += 1; continue
        if not q or not has(pn, q):
            funnel["link:quote_not_in_passage"] += 1; continue
        qn = f" {norm(q)} "
        if not any(f" {w} " in qn for w in content_words(a)) or not any(f" {w} " in qn for w in content_words(b)):
            funnel["link:quote_misses_an_end"] += 1; continue
        links.append({"from": a, "rel": rel, "to": b})
        funnel["link:kept"] += 1
    facts = []
    for f in d.get("facts") or []:
        if not isinstance(f, dict):
            continue
        e, r, v, q = _s(f.get("e")), _s(f.get("r")), _s(f.get("v")), _s(f.get("q"))
        if not (e and r and v and q) or len(e) > 80 or len(v) > 80 or len(r.split()) > 5 or r != r.lower():
            funnel["fact:shape"] += 1; continue
        if r not in FACT_RELATIONS.split(", "):
            funnel["fact:relation"] += 1; continue
        if r in _PERSON_RELS and len(e.split()) < 2:
            funnel["fact:short_name"] += 1; continue
        if not is_name(e) or norm(e) == norm(v):
            funnel["fact:entity"] += 1; continue
        if not has(pn, q) or not has(pn, e):
            funnel["fact:not_in_passage"] += 1; continue
        digits = re.findall(r"\d+", v)
        if not has(norm(q), v) and not (digits and all(f" {x} " in f" {norm(q)} " for x in digits)):
            funnel["fact:value_not_in_quote"] += 1; continue
        facts.append({"e": e, "r": r, "v": v})
        funnel["fact:kept"] += 1
    return {"events": events, "links": links, "facts": facts}


def reasoning_of(level: str) -> dict | None:
    """OpenRouter's reasoning control for a hybrid instant/thinking model: 'off' asks for the instant answer."""
    if not level:
        return None
    return {"enabled": False} if level == "off" else {"effort": level}


def when_of(ev: dict) -> dict | None:
    a, b = parse_time(ev["when"]), parse_time(ev["end"])
    basis = ev.get("basis", "stated")
    if a and b:
        w = {"y0": min(a["y0"], b["y0"]), "y1": max(a["y1"], b["y1"]), "day": None, "approx": a["approx"] or b["approx"]}
    else:
        w = a or b
    if w:
        w["basis"] = basis
        if basis == "context":
            w["approx"], w["day"] = True, None
    return w


def passage_years(extracted: list[tuple[str, int, dict]], back: int = 12, ahead: int = 2) -> dict:
    """(book, passage) -> the year the book is telling about there: the commonest year its events carry (ties: the
    later -- a narrative looks back more than ahead), else the last such year within `back` passages before, else the
    next within `ahead` after."""
    own: dict[tuple, int] = {}
    by_book = collections.defaultdict(list)
    for book, k, d in extracted:
        by_book[book].append(k)
        ys = collections.Counter(w["y0"] for ev in d["events"] if (w := when_of(ev)) and w["y1"] - w["y0"] <= 5)
        if ys:
            own[(book, k)] = max(ys, key=lambda y: (ys[y], y))
    out = {}
    for book, ks in by_book.items():
        for k in ks:
            y = own.get((book, k))
            if y is None:
                y = next((own[(book, j)] for j in range(k - 1, k - back - 1, -1) if (book, j) in own), None)
            if y is None:
                y = next((own[(book, j)] for j in range(k + 1, k + ahead + 1) if (book, j) in own), None)
            if y is not None:
                out[(book, k)] = y
    return out


def assemble(extracted: list[tuple[str, int, dict]], news: list[dict] = ()) -> HistoryGraph:
    """The graph from the books' checked passages and, first, the newspaper's dated events
    (build_nyt_events.py): a day the paper reports is a stated date, so a book event of the same name takes it."""
    g = HistoryGraph()
    days = collections.defaultdict(set)                  # a name the paper prints on many days is a column, not
    for e in news:                                       # an event: "Obituary", "Losses by Fire", "Marine Intelligence"
        days[key(e["name"])].add(e["day"])
    for e in news:
        w = parse_time(e["day"])
        if not w or not w["day"]:
            g.refused["news:bad_day"] += 1; continue
        if len(days[key(e["name"])]) >= NEWS_COLUMN_DAYS:
            g.refused["news:column_name"] += 1; continue
        if not e["where"] and not e["who"]:              # "Court Appoints Receiver": nowhere, no one
            g.refused["news:no_place_or_party"] += 1; continue
        g.add_event(e["name"], {"y0": w["y0"], "y1": w["y0"], "day": w["day"], "approx": False, "basis": "stated"},
                    e["where"], e["who"], e["kind"], ("nyt", e["article"]))
    hints = {}
    for book, k, d in extracted:
        src = (book, k)
        ids = {}
        for ev in d["events"]:
            ids[ev["name"]] = g.add_event(ev["name"], when_of(ev), ev["where"], ev["who"], ev["kind"], src)
            if ev.get("hint") and (h := parse_time(ev["hint"])):
                hints.setdefault(ids[ev["name"]], h)
        for l in d["links"]:
            g.add_link(ids[l["from"]], l["rel"], ids[l["to"]], src)
        for f in d["facts"]:
            g.add_fact(f["e"], f["r"], f["v"], src)
    g.prune_links("time_contradicts_merged")                  # a merge gave an end a stronger date
    g.dates = g.infer_dates(passage_years(extracted), hints)   # after the merge: inferred dates never merge
    g.prune_links("time_contradicts_dated")                   # a link to an undated end, checked with its date
    return g


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", action="append", required=True, help="a folder of books (.txt); repeatable")
    ap.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--books", type=int, default=0, help="a pilot: this many books from each folder (seeded); 0 = all")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--words", type=int, default=3000)
    ap.add_argument("--max-cost", type=float, default=60.0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dry", action="store_true", help="count books and passages, call nothing")
    ap.add_argument("--reasoning", default="", help="a hybrid model's thinking: off | low | medium | high ('' = the model's default)")
    ap.add_argument("--assemble", nargs="*", default=None, metavar="EXTRACT",
                    help="no reading: build the graph from these history_extract*.jsonl files (and --nyt)")
    ap.add_argument("--nyt", default="", help="nyt_events*.jsonl to merge in (build_nyt_events.py)")
    ap.add_argument("--reuse", default="", help="a history_extract*.jsonl (same --src) whose read passages are kept as "
                                                 "they are and not sent again -- a run moved to another model mid-way")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from openrouter import OpenRouterProposer
    t0 = time.time()
    out_dir = ROOT / "standin" / "data" / "out"
    hold = out_dir / f"history_graph{args.tag}.hold"
    if hold.exists():                                   # a queued run the owner moved elsewhere: do nothing
        print(f"held: {hold.name} exists ({hold.read_text(encoding='utf-8').strip()})")
        return
    if args.assemble is not None:
        extracted = [(d["book"], d["passage"], d) for path in args.assemble
                     for d in map(json.loads, open(path, encoding="utf-8"))]
        news = [json.loads(l) for l in open(args.nyt, encoding="utf-8")] if args.nyt else []
        g = assemble(extracted, news)
        g.save(out_dir / f"history_graph{args.tag}.jsonl")
        stats = {"passages": len(extracted), "news_events": len(news), "events": len(g.events), "links": len(g.links),
                 "facts": len(g.facts), "dated_by": dict(g.dates), "links_refused": dict(g.refused),
                 "events_with_news_day": sum(1 for e in g.events.values() if any(s[0] == "nyt" for s in e.sources)
                                             and any(s[0] != "nyt" for s in e.sources))}
        (ROOT / "validation" / "logs" / f"history_graph{args.tag}.assemble.json").write_text(json.dumps(stats, indent=1),
                                                                                           encoding="utf-8")
        print(json.dumps(stats, indent=1))
        return
    books = []
    for src in args.src:                                        # a pilot draws --books from each folder
        these = list_books([src])
        books += sorted(random.Random(0).sample(these, min(args.books, len(these)))) if args.books else these
    kept, pairs = dedupe(books)
    print(f"{len(books)} books, {len(kept)} after dedupe ({time.time() - t0:.0f}s)", flush=True)
    jobs, metas, funnel = [], {}, collections.Counter()
    for p, rel in kept:
        bid = hashlib.sha1(rel.encode()).hexdigest()[:12]
        metas[bid] = book_meta(p, rel)
        book = clean(open(p, encoding="utf-8", errors="replace").read())
        metas[bid]["era"] = book_era(book)
        for k, text in enumerate(passages(book, args.words)):
            funnel["passages"] += 1
            if len(_DATEWORD.findall(text)) < 2:
                funnel["passages_no_dates"] += 1; continue
            jobs.append((bid, k, text))
    reused = []
    if args.reuse:
        prior = {(d["book"], d["passage"]): d for d in map(json.loads, open(args.reuse, encoding="utf-8"))
                 if d.get("events") or d.get("links") or d.get("facts")}
        reused = [(b, k, {x: y for x, y in prior[(b, k)].items() if x not in ("book", "passage")})
                  for b, k, _ in jobs if (b, k) in prior]
        jobs = [j for j in jobs if (j[0], j[1]) not in prior]
        funnel["passages_reused"] = len(reused)
        print(f"{len(reused):,} passages kept from {pathlib.Path(args.reuse).name}", flush=True)
    words = sum(len(t.split()) for _, _, t in jobs)
    print(f"{funnel['passages']:,} passages, {len(jobs):,} with dates to read, {words / 1e6:.1f}M words", flush=True)
    if args.dry:
        print(json.dumps({"books": len(books), "books_kept": len(kept), "duplicates": len(pairs),
                          "passages": funnel["passages"], "passages_sent": len(jobs), "words_sent": words}))
        return

    local, lock, stop = threading.local(), threading.Lock(), threading.Event()
    usage = collections.Counter()

    def run(job):
        bid, k, text = job
        if not hasattr(local, "llm"):
            local.llm = OpenRouterProposer(args.model, system=SYSTEM, max_tokens=6000, reasoning=reasoning_of(args.reasoning))
        llm = local.llm
        llm.offline = args.offline or stop.is_set()
        before, reply = dict(llm.usage), ""
        for attempt in range(6):
            try:
                reply = llm.chat(f"Book: {metas[bid]['title']}\n\nPassage:\n{text}")
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 402, 403):
                    stop.set(); llm.offline = True
                    with lock:
                        usage[f"error_{e.code}"] += 1
                    break
                time.sleep(min(60, 3 * 2 ** attempt))
            except Exception as e:                                   # noqa: BLE001 -- timeouts, resets: retried
                with lock:
                    usage[f"retry_{type(e).__name__}"] += 1
                time.sleep(min(60, 3 * 2 ** attempt))
        local_funnel = collections.Counter()
        d = check_passage(parse(reply), text, local_funnel, metas[bid]["era"])
        with lock:
            for key_ in ("prompt_tokens", "completion_tokens", "cost"):
                usage[key_] += llm.usage[key_] - before[key_]
            if usage["cost"] > args.max_cost:
                stop.set()
            funnel.update(local_funnel)
            funnel["replies" if reply else "no_reply"] += 1
        return bid, k, d

    extracted, done = list(reused), 0
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for bid, k, d in pool.map(run, jobs):
            extracted.append((bid, k, d))
            done += 1
            if done % 200 == 0 or done == len(jobs):
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done:,}/{len(jobs):,} passages, events {funnel['event:kept']:,}, links {funnel['link:kept']:,}, "
                      f"facts {funnel['fact:kept']:,}, ${usage['cost']:.2f}, eta {(len(jobs) - done) / max(rate, 1e-9) / 60:.0f} min"
                      f"{' [STOPPED: cached only]' if stop.is_set() else ''}", flush=True)
    with open(out_dir / f"history_extract{args.tag}.jsonl", "w", encoding="utf-8") as f:
        for bid, k, d in extracted:
            f.write(json.dumps({"book": bid, "passage": k, **d}, ensure_ascii=False) + "\n")
    g = assemble(extracted)
    g.save(out_dir / f"history_graph{args.tag}.jsonl")
    rels = collections.Counter(r for (_, r, _) in g.links)
    dated = sum(1 for e in g.events.values() if e.when)
    sample = [f"{g.events[a].name} ({g.events[a].span()}) --{r}--> {g.events[b].name} ({g.events[b].span()})"
              for (a, r, b) in list(g.links)[:25]]
    manifest = {"source": SOURCE, "model": args.model, "reused_from": pathlib.Path(args.reuse).name if args.reuse else "",
                "passages_reused": len(reused), "books": len(books), "books_kept": len(kept),
                "duplicates_dropped": pairs, "books_read": {b: m for b, m in metas.items()},
                "passages": funnel["passages"], "passages_sent": len(jobs),
                "graph": {"events": len(g.events), "events_dated": dated, "dated_by": dict(g.dates), "links": len(g.links),
                          "links_by_relation": dict(rels), "links_refused": dict(g.refused), "facts": len(g.facts)},
                "funnel": dict(funnel.most_common()), "usage": dict(usage), "stopped": stop.is_set(),
                "sample_links": sample, "wall_s": round(time.time() - t0, 1)}
    (ROOT / "validation" / "logs" / f"history_graph{args.tag}.manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("books_kept", "passages_sent", "graph", "usage", "stopped", "wall_s")}, indent=1))
    print("\n".join(sample))


if __name__ == "__main__":
    main()
