"""exp_leak_scan -- order 0016's acceptance: answers as a regular user receives them, scanned for their sources.

The ask loop as served -- the wiki world, the books before an offline Wikidata (its cache only: no network),
questions asked as people ask (`question_forms`), the CubeLang VM -- answers until it has spoken `--answers`
times: from the store (facts the world holds), from the books (the books bench's facts, taken out of the world),
from the lookup (people the Wikidata cache knows), and profiles ("Who was X?"). Each record goes through
`audience.for_user`, and every text a user reads (reply, prose, statement, answer, needs, clarify, scope) is
scanned for:
- a source name: every witness the store's provenance names after the run, the authors and titles of every book
  any record cites, `audience.SOURCE_NAMES`;
- a URL or a domain;
- a run of 8 words any stored passage holds (the books' index).
Also: the user's JSON carries no provenance field; every spoken answer has a sealed chain under audit
(`audience.audit_chain`); and how often the guard changed a reply at all (the behaviour otherwise unchanged).

The log keeps questions, answers and counts; no passage, title or author.

    CUBBY_BOOKS_INDEX=<index.sqlite> python validation/exp_leak_scan.py [--answers 200] [--bench <exp_books_bench json>]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import audience as A  # noqa: E402
import exp_books_bench as eb  # noqa: E402
import wikikg as wk  # noqa: E402
from book_source import BookSource  # noqa: E402
from books import BookIndex  # noqa: E402
from question_forms import Forms, TemplateEmitter  # noqa: E402

TEXT_FIELDS = ("reply", "prose", "statement", "answer", "needs", "clarify", "scope")


def cached_people(n: int, rng) -> list[str]:
    """Labels the Wikidata cache found by search (the lookup's own entities), people-shaped: two or more
    capitalised words."""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "standin", "data", "out", "wikidata_cache", "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in (d.get("search") or [])[:1] if isinstance(d, dict) else []:
            lab = s.get("label") or ""
            if re.fullmatch(r"[A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+", lab):
                out.append(lab)
    out = sorted(set(out))
    rng.shuffle(out)
    return out[:n]


def spoke(r: dict) -> bool:
    """Spoken: a verified chain, or a profile / introspection whose lines the VM recovered (`sleep.answered`)."""
    return bool(r.get("verified") or (r.get("reason") in ("profile", "self") and r.get("answer")))


def spoken_text(user: dict) -> str:
    parts = []
    for f in TEXT_FIELDS:
        v = user.get(f)
        if v:
            parts.append(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=os.environ.get("CUBBY_BOOKS_INDEX"))
    ap.add_argument("--answers", type=int, default=200)
    ap.add_argument("--bench", default=None, help="the books bench's log (its questions are asked too)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", f"exp_leak_scan_{_dt.date.today().isoformat()}.json"))
    a = ap.parse_args()
    t0, rng = time.time(), random.Random(a.seed)
    from ask import AskLoop
    from sources import WikidataSource
    idx = BookIndex(a.index)
    pairs = eb.gold_pairs()
    bench_q = []
    if a.bench and os.path.exists(a.bench):
        rows = json.load(open(a.bench, encoding="utf-8"))["bench"]["rows"]
        bench_q = [(r["question"], "books", (r["subject"], r["relation"])) for r in rows]
    out_keys = {k for _q, _w, k in bench_q}
    triples = [(s, r, o) for s, r, o in wk.load_triples()
               if not ((wk.decamel(s), eb.REL.get(r)) in out_keys or (r == "CAPITAL_OF" and (wk.decamel(o), "capital") in out_keys))]
    world = wk.wiki_world(triples=triples)
    loop = AskLoop(TemplateEmitter(), world=world, source=WikidataSource(offline=True, use_choices=True), lexicon=True,
                   books=BookSource(idx))
    forms = Forms()
    held = [k for k in pairs if k not in out_keys and k[1] in eb.REL.values()]
    rng.shuffle(held)
    store_q = [(forms.render(rel, s, eb.TENSE.get(rel, "past"))[0], "store", (s, rel)) for s, rel in held[:600]]
    people = cached_people(120, rng)
    lookup_q = [(q, "lookup", (p, rel)) for p in people for rel, q in
                (("date of birth", f"When was {p} born?"), ("place of birth", f"Where was {p} born?"))][:150]
    profile_q = [(f"Who was {s}?", "profile", (s, None)) for s, _r in held[600:700]] + \
                [(f"Who was {p}?", "profile", (p, None)) for p in people[:40]]
    queue = bench_q + [x for grp in zip(store_q, lookup_q, profile_q) for x in grp] + store_q[len(profile_q):]
    recs = []
    for q, where, key in queue:
        if sum(spoke(r) for _w, r in recs) >= a.answers and len(recs) >= len(bench_q):
            break
        try:
            recs.append((where, loop.ask(q)))
        except Exception as e:                                   # noqa: BLE001 -- a crash is a finding, not a pass
            recs.append((where, {"question": q, "verified": False, "reason": f"error: {type(e).__name__}: {e}"[:200]}))
    names = set(A.SOURCE_NAMES)
    for v in (getattr(world, "provenance", None) or {}).values():
        names |= A.source_names({"learned": [{"source": v}]})
    for _w, r in recs:
        names |= A.source_names(r)
    names = {n for n in names if len(n) >= 4}
    guard = A.LeakGuard(index=idx)
    rows, leaks, quotes, urls, fields, sealed, guarded = [], 0, 0, 0, 0, 0, 0
    for where, r in recs:
        user = A.for_user(r)
        text = spoken_text(user)
        own = {str(r.get("answer") or "").lower()} if "profile" not in r else set()
        hit = [n for n in sorted(names) if n not in own and re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", text, re.I)]
        url = [m.group(0) for m in A._URL.finditer(text)]
        quo = guard.quotes(text)
        bad_fields = sorted(set(user) & set(A.PROVENANCE_FIELDS + A.INTERNAL_FIELDS))
        ok_chain = None
        if spoke(r) and r.get("id"):
            chain = A.audit_chain(loop, r["id"])
            ok_chain = bool(chain and chain["sealed"] and chain["facts"] is not None)
            sealed += ok_chain
        leaks += bool(hit); urls += bool(url); quotes += bool(quo); fields += bool(bad_fields); guarded += bool(r.get("guard"))
        rows.append({"where": where, "question": r.get("question"), "verified": spoke(r), "tier": r.get("tier"),
                     "answer": r.get("answer"), "reason": r.get("reason"), "source_names": hit, "urls": url,
                     "quotes": len(quo), "provenance_fields": bad_fields, "audit_chain": ok_chain, "guard": bool(r.get("guard"))})
    spoken = [x for x in rows if x["verified"]]
    by = collections.Counter(x["where"] for x in spoken)
    rep = {"args": {k: v for k, v in vars(a).items() if k not in ("index", "out")}, "asked": len(rows), "spoken": len(spoken),
           "spoken_by_kind": dict(by), "names_scanned": len(names),
           "replies_with_a_source_name": leaks, "replies_with_a_url": urls, "replies_with_an_8_word_quote": quotes,
           "user_json_with_provenance_fields": fields, "spoken_with_a_sealed_audit_chain": sealed,
           "audit_coverage": round(sealed / max(1, len(spoken)), 3), "guard_changed_a_reply": guarded,
           "refused": dict(collections.Counter(x["reason"] for x in rows if not x["verified"])),
           "wall_s": round(time.time() - t0, 1), "rows": rows}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(spoken)}/{len(rows)} spoken {dict(by)}; source names {leaks}, urls {urls}, 8-word quotes {quotes}, "
          f"provenance fields {fields}; audit {sealed}/{len(spoken)}; guard changed {guarded} [{rep['wall_s']}s]\nwrote {a.out}")


if __name__ == "__main__":
    main()
