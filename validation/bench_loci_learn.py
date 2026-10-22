"""bench_loci_learn -- order 0007's battery: a question the trunk cannot answer is looked up once from Wikidata, and
the next time -- in a new process, with the store persisted -- the trunk answers it through the loci path.

100 one-hop questions, asked as people ask (`question_forms`, the set's own templates): each (entity, relation) is
stated on Wikidata (the hdc world's Wikidata5M facts pick them), absent from the world store, its entity outside
the wiki world, its relation one the 450M's shard addressed (`loci_reader.WIKIDATA`).

    --phase 1   ask 1: the reader reads first (a miss), the loop learns from Wikidata only (no web, cached), the
                reader reads again; spoken answers checked against what was learned; the facts the gate
                admitted written to --store; Wikidata's calls counted.
    --phase 2   ask 2, a new process: --store loaded (`AskLoop.load_learned`); every source call counted (0 for
                the facts learned in ask 1); the reader's answer against the stored value; and 30 creative asks
                (EN/FR) route away from the facts with the reader on, as with it off, and call no source.

    CUBBY_LOCI_CKPT=<run 5 final> CUBBY_HDC_WORLD=<hdc_world.sqlite> python validation/bench_loci_learn.py --phase 1 --src <hdc>
    ... python validation/bench_loci_learn.py --phase 2 --src <hdc>
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data"), HERE, os.path.join(ROOT, "standin", "tests"),
          os.path.join(ROOT, "tests", "brain")):                 # test_serve's fixtures (moved there in order 0031)
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.reasoning.planner import normalize  # noqa: E402

DATE = _dt.date.today().isoformat()
OUT = os.path.join(HERE, "logs", f"bench_loci_learn_{DATE}.json")
STORE = os.path.join(ROOT, "standin", "data", "out", "bench_loci_learned.jsonl")
CREATIVE = ["can you write a poem about berlin?", "write me a short story about a dragon", "imagine a city made of glass",
            "tell me a joke about cats", "compose a haiku about the sea", "what would you do on mars?",
            "invent a new holiday", "describe a dream you might have", "make up a riddle", "sing me a song about rain",
            "write a limerick about a teacher", "imagine you are a tree", "draw me a picture with words",
            "what is your favourite colour?", "how do you feel today?", "write a letter to the moon",
            "écris un poème sur paris", "raconte-moi une histoire de pirates", "imagine une ville sous la mer",
            "invente une chanson sur le printemps", "fais-moi rire", "décris un rêve étrange", "écris un haïku sur la neige",
            "que ferais-tu sur la lune ?", "quelle est ta couleur préférée ?", "imagine que tu es un oiseau",
            "write a sonnet about autumn", "pretend you are a pirate", "tell me a bedtime story", "dream up a new sport"]


def battery(src, hdc, wiki_names, n, rng):
    from exp_hdc_bench import set_forms
    import loci_reader as lr
    forms = set_forms(src, hdc)
    rels = [r for r in lr.WIKIDATA if normalize(r) in hdc.relations() and forms.render(r, "X", "past")]
    rows = hdc.db.execute(f"SELECT subj, rel, obj FROM facts WHERE rel IN ({','.join('?' * len(rels))}) ORDER BY rowid",
                          rels).fetchall()
    rows = rng.sample(rows, min(len(rows), n * 30))              # seeded: both phases ask the same questions
    out, seen = [], set()
    for s, r, o in rows:
        if len(out) >= n or normalize(s) in wiki_names or (s, r) in seen:
            continue
        seen.add((s, r))
        out.append({"entity": s, "relation": r, "hdc_value": o, "question": forms.render(r, s, "past")[0]})
    return out, forms


class Counted:
    """The source, every call counted per entity."""

    def __init__(self, inner) -> None:
        self.inner, self.name, self.asked = inner, inner.name, collections.Counter()

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def facts(self, entity, **kw):
        self.asked[normalize(entity)] += 1
        return self.inner.facts(entity, **kw)


def make_loop(forms, source):
    import wikikg as wk
    from ask import AskLoop
    from loci_reader import LociReader
    from question_forms import TemplateEmitter
    loop = AskLoop(TemplateEmitter(forms), world=wk.wiki_world(), source=source, lexicon=True, loci=LociReader())
    loop.resolvers.append(forms)
    return loop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    import wikikg as wk
    from hdc_store import HdcWorld
    from sources import WikidataSource
    from cubby_serve.sources.http import wikidata as _wikidata_fetch  # noqa: E402 -- the online fetch (0031)
    t0, hdc = time.time(), HdcWorld()
    wiki = wk.wiki_world()
    names = set(wiki.index._by_subj) | set(wiki.index._by_obj)
    qs, forms = battery(a.src, hdc, names, a.n, random.Random(a.seed))
    if a.phase == 1:
        wd = WikidataSource(offline=False, use_choices=True, fetch=_wikidata_fetch())
        src = Counted(wd)
        loop = make_loop(forms, src)
        rows, learned = [], []
        for q in qs:
            t = time.time()
            rec = loop.ask(q["question"])
            first = (rec.get("loci") or [{}])[0]
            got = [l for l in rec.get("learned") or [] if l.get("status") == "accepted"]
            learned += got
            rows.append({**q, "first_miss": bool(first.get("miss")), "path": rec.get("path"), "verified": bool(rec["verified"]),
                         "answer": rec.get("answer"), "reason": rec.get("reason"), "learned": len(got),
                         "reads": [x.get("answer") for x in rec.get("loci") or []],
                         "stored": (rec.get("refused") or {}).get("stored") if isinstance(rec.get("refused"), dict) else None,
                         "wrong": bool(rec["verified"] and normalize(rec.get("answer") or "") not in
                                       {normalize(l["fact"].split(" is the ", 1)[0]) for l in got} | {normalize(q["hdc_value"])}),
                         "wall_s": round(time.time() - t, 1)})
        with open(a.store, "w", encoding="utf-8") as f:
            for l in learned:
                f.write(json.dumps({"fact": l["fact"], "source": l.get("source") or "wikidata"}, ensure_ascii=False) + "\n")
        rep = {"phase1": {"asked": len(rows), "first_read_miss": sum(r["first_miss"] for r in rows),
                          "learned_any": sum(r["learned"] > 0 for r in rows), "spoken": sum(r["verified"] for r in rows),
                          "wrong_spoken": sum(r["wrong"] for r in rows), "paths": dict(collections.Counter(r["path"] for r in rows)),
                          "refused": dict(collections.Counter(r["reason"] for r in rows if not r["verified"])),
                          "wikidata_http_calls": wd.calls, "facts_learned": len(learned),
                          "wall_s_per_question": round(sum(r["wall_s"] for r in rows) / max(1, len(rows)), 1), "rows": rows}}
    else:
        rep = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
        p1 = {r["question"]: r for r in rep.get("phase1", {}).get("rows", [])}
        src = Counted(WikidataSource(offline=True, use_choices=True))
        loop = make_loop(forms, src)
        n_loaded = loop.load_learned(a.store)
        rows = []
        for q in qs:
            before = sum(src.asked.values())
            t = time.time()
            rec = loop.ask(q["question"])
            stored = {normalize(t_.obj) for _f, t_ in loop.world.index._by_subj.get(normalize(q["entity"]), [])}
            rows.append({**q, "learned_in_ask_1": bool(p1.get(q["question"], {}).get("learned")), "path": rec.get("path"),
                         "verified": bool(rec["verified"]), "answer": rec.get("answer"), "reason": rec.get("reason"),
                         "reads": [x.get("answer") for x in rec.get("loci") or []],
                         "source_calls": sum(src.asked.values()) - before,
                         "equals_stored": bool(rec["verified"] and normalize(rec.get("answer") or "") in stored),
                         "wall_s": round(time.time() - t, 1)})
        learned = [r for r in rows if r["learned_in_ask_1"]]
        import serve as sv
        from test_serve import STORE as SSTORE, ChainEmitter, overlap_retriever
        s = sv.CubbyServe(ChainEmitter(), overlap_retriever, SSTORE, route_tau=0.0)
        reads0 = loop.loci.calls
        creative = [{"ask": c, "needs_facts": s.needs_facts(c)[0], "route": s.route(c).get("cortex")} for c in CREATIVE]
        rep["phase2"] = {"loaded": n_loaded, "asked": len(rows), "learned_in_ask_1": len(learned),
                         "source_calls_on_learned": sum(r["source_calls"] for r in learned),
                         "loci_answered_learned": sum(r["path"] == "loci" and r["verified"] for r in learned),
                         "equals_stored_on_learned": round(sum(r["equals_stored"] for r in learned) / max(1, len(learned)), 3),
                         "wrong_spoken": sum(r["verified"] and not r["equals_stored"] for r in rows),
                         "refused": dict(collections.Counter(r["reason"] for r in rows if not r["verified"])),
                         "wall_s_per_question": round(sum(r["wall_s"] for r in rows) / max(1, len(rows)), 1),
                         "creative": {"asked": len(creative), "routed_to_facts": sum(c["needs_facts"] for c in creative),
                                      "reader_reads": loop.loci.calls - reads0, "routes": dict(collections.Counter(c["route"] for c in creative))},
                         "rows": rows}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v for k, v in rep.items()},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
