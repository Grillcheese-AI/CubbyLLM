"""standin/news_events.py -- events by day (order 0024): the archive's news items about the same event grouped across
outlets, by Cubby's own reading (no model).

One item reached through several feeds (a topic feed, a region feed) is one item (by link). Two items are about the
same event when they are dated within `WITHIN_DAYS` of each other and share two names with their names and key words
overlapping by `THRESHOLD` (Jaccard), or one name with `THRESHOLD_ONE`; two groups merge when at least `LINKED` of
the pairs between them are such pairs (average linkage: a single link does not chain). A name carried by more than
`COMMON` of the items (Canada, Québec, the UN) does not count as a name in common. Names are capitalised words not starting a sentence
(a headline's first word is capitalised because it starts it), folded (case, accents); key words are the other words
of 5+ letters, minus the stop words of English and French. Names carry across languages ("Gaza" in an English and a
French headline); key words do not, so a cross-language pair needs more names in common.

Only items of kind `news` are grouped: an office's statement and an organisation's analysis are kept as what was said,
never read as the event (`news_feeds.json`). A group stays **reported** however many outlets carry it: agreement among
newsrooms is syndication, not corroboration (`news_source.py`).

    python standin/news_events.py --root <archive> [--from 2026-10-22] [--to 2026-10-29] [--sample 100]
-> counts (committable) and, with `--sample`, a judging file of random groups (local: it holds headlines).

Wired: STANDALONE (`validation`; the reply path reads groups once 0024's bench is built).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import re
import sys
import unicodedata

__wiring__ = "STANDALONE"

HERE = pathlib.Path(__file__).resolve().parent
WITHIN_DAYS = 2
THRESHOLD = 0.15         # with two names in common
THRESHOLD_ONE = 0.3
LINKED = 0.5
COMMON = 0.02            # a name in more than this share of the items (Canada, Québec, ONU) is not evidence of one event      # with one: "Canada" alone chained a state visit to an election dispute (2026-10-22)
STOP = frozenset("""the a an and or but for nor with from into onto over under after before about amid among between
    this that these those their there here where when what which while whose would could should says said will been
    being have has had more most than then they them very also only just into other some such
    le la les un une des du de et ou mais pour avec sans dans sur sous apres avant entre selon cette ces leur leurs
    dont quand alors aussi plus moins tres sont etait etre avoir fait faire comme tout tous toute toutes pendant
    contre depuis encore lors veut doit peut nouveau nouvelle nouvelles annonce""".split())


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


_WORD = re.compile(r"[^\W\d_][\w'’-]*")


def capitals(text: str) -> float | None:
    """The share of a headline's words (4+ letters, after the first) that are capitalised; None when too short."""
    ws = [w for w in _WORD.findall(text or "")[1:] if len(w) >= 4 and fold(w) not in STOP]
    return sum(w[:1].isupper() for w in ws) / len(ws) if len(ws) >= 4 else None


def title_case_feeds(rows: list[dict], min_items: int = 5, bar: float = 0.8) -> set[str]:
    """Feeds whose house style capitalises most words (mean share >= `bar` over `min_items`+ headlines): one name-dense
    headline in sentence case reaches 0.8 too, a feed's habit does not."""
    by: dict[str, list[float]] = {}
    for r in rows:
        c = capitals(r.get("title", ""))
        if c is not None:
            by.setdefault(r["feed"], []).append(c)
    return {f for f, cs in by.items() if len(cs) >= min_items and sum(cs) / len(cs) >= bar}


def features(title: str, summary: str = "", title_case: bool = False) -> tuple[set, set]:
    """(names, key words) of an item's headline and summary. A title-cased headline gives key words only (one
    publisher's house style joined Nepal's floods to Venezuela's earthquake by their capitals, 2026-10-22)."""
    names, words = set(), set()
    for text, caps in ((title, not title_case), (summary, True)):
        for sent in re.split(r"(?<=[.!?:;])\s+|\s+[-–—|]\s+", text or ""):
            toks = _WORD.findall(sent)
            for i, t in enumerate(toks):
                f = fold(t.strip("'’-"))
                if len(f) < 2 or f in STOP:
                    continue
                if caps and i > 0 and (t[:1].isupper() or (t.isupper() and len(t) >= 2)):
                    names.add(f)
                elif len(f) >= 5:
                    words.add(f[:-1] if f.endswith("s") else f)
    return names, words - names


def _day(r: dict) -> dt.date:
    return dt.date.fromisoformat(r.get("published") or r["first_seen"][:10])


def group(rows: list[dict], within_days: int = WITHIN_DAYS, threshold: float = THRESHOLD, kinds=None) -> list[list[dict]]:
    """Groups of news items, biggest first; an item with no match is a group of one. `kinds`: feed -> kind (the
    manifest's, which wins over a row's: rows archived before kinds existed carry none)."""
    items, by_link = [], {}
    for r in rows:
        if (kinds or {}).get(r["feed"], r.get("kind", "news")) != "news":
            continue
        k = r.get("link") or r["key"]
        if k in by_link:
            by_link[k].setdefault("feeds", [by_link[k]["feed"]]).append(r["feed"])
            continue
        by_link[k] = r = dict(r)
        items.append(r)
    styled = title_case_feeds(items)
    feats = [features(r.get("title", ""), r.get("summary", ""), r["feed"] in styled) for r in items]
    by_name: dict[str, list[int]] = {}
    for i, (names, _) in enumerate(feats):
        for n in names:
            by_name.setdefault(n, []).append(i)
    days = [_day(r) for r in items]
    common = {n for n, js in by_name.items() if len(js) > max(3, COMMON * len(items))}
    links = []
    for i, (ni, wi) in enumerate(feats):
        seen = set()
        for n in ni:
            for j in by_name[n]:
                if j <= i or j in seen:
                    continue
                seen.add(j)
                if abs((days[i] - days[j]).days) > within_days:
                    continue
                nj, wj = feats[j]
                a, b = ni | wi, nj | wj
                shared, jac = len((ni & nj) - common), len(a & b) / len(a | b)
                if (shared >= 2 and jac >= threshold) or (shared == 1 and jac >= THRESHOLD_ONE):
                    links.append((jac, i, j))
    # Average linkage, strongest pairs first: two groups merge only when at least `LINKED` of the pairs between them
    # are links. Single links chained a state visit, an election dispute and a speech by their one shared name.
    linked = {(i, j) for _, i, j in links}
    member = {i: [i] for i in range(len(items))}
    for _, i, j in sorted(links, reverse=True):
        a, b = member[i], member[j]
        if a is b:
            continue
        n = sum((min(x, y), max(x, y)) in linked for x in a for y in b)
        if n >= LINKED * len(a) * len(b):
            a.extend(b)
            for y in b:
                member[y] = a
    out = {id(g): [items[i] for i in g] for g in member.values()}
    return sorted(out.values(), key=lambda g: (-len(g), min(_day(r) for r in g)))


def load(root, start: str | None = None, end: str | None = None) -> list[dict]:
    rows = []
    for f in sorted(pathlib.Path(root).glob("????-??-??.jsonl")):
        rows += [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    return [r for r in rows if (not start or _day(r).isoformat() >= start) and (not end or _day(r).isoformat() <= end)]


def outlets(group_rows: list[dict], feeds: list[dict]) -> set[str]:
    host = {f["id"]: (f.get("publisher_hosts") or [f["id"]])[0] for f in feeds}
    return {host.get(r["feed"], r["feed"]) for r in group_rows}


def summary(groups: list[list[dict]], feeds: list[dict]) -> dict:
    multi = [g for g in groups if len(g) > 1]
    return {"items": sum(map(len, groups)), "groups": len(groups), "groups_of_2_plus": len(multi),
            "items_in_groups_of_2_plus": sum(map(len, multi)),
            "groups_across_outlets": sum(len(outlets(g, feeds)) > 1 for g in multi),
            "biggest": [len(g) for g in groups[:10]], "within_days": WITHIN_DAYS, "threshold": THRESHOLD,
            "threshold_one_name": THRESHOLD_ONE, "linked": LINKED, "common": COMMON}


def main() -> None:
    sys.path.insert(0, str(HERE))
    from news_archive import feeds as manifest_feeds
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--sample", type=int, default=0, help="write N random groups of 2+ for judging (local file)")
    ap.add_argument("--sample-out", default=str(HERE.parent / "validation" / "logs" / "_news_groups_sample.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fs = manifest_feeds()
    groups = group(load(a.root, a.start, a.end), kinds={f["id"]: f.get("kind", "news") for f in fs})
    print(json.dumps(summary(groups, fs), indent=1))
    if a.sample:
        multi = [g for g in groups if len(g) > 1]
        pick = random.Random(a.seed).sample(multi, min(a.sample, len(multi)))
        json.dump([{"n": len(g), "outlets": sorted(outlets(g, fs)), "same_event": None,
                    "items": [{"feed": r["feed"], "day": _day(r).isoformat(), "title": r["title"]} for r in g]}
                   for g in pick], open(a.sample_out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{len(pick)} groups to judge -> {a.sample_out}")


if __name__ == "__main__":
    main()
