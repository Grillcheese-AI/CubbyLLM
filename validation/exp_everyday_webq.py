"""exp_everyday_webq -- everyday questions as a scoreboard (2026-10-24, Nick: "how far are we from a useful model?").

The questions people ask (WebQuestions, filtered to the USA, Canada and Europe: 2,216; `validation/builders/
build_webq_bench.py`) and the real stack's 20 (`validation/stack_answers.py`), through the ask loop as the stack mounts
it (the wiki world, the hdc world behind it, the Wikidata source offline, the lexicon), in two arms:

  --everyday off   the loop as today
  --everyday on    the host reads everyday wording first (`cubbyllm/reasoning/everyday.py`, CB_EVERYDAY)

and one of two emitters for whatever the host does not read: `none` (no model: what the host alone answers) or `450m`
(the programs seat of the 450M on grilly2, as `cubby_serve` loads it). Every spoken answer is checked against the
question's listed answers by a normalizer, not a judge; what does not match is listed for a person to read (WebQuestions'
answers are Freebase's of 2013: a team, a mayor or a currency may have changed since, and a "where" has many right
answers), so `mismatch` is not `wrong` until read.

    python validation/exp_everyday_webq.py --everyday on --emitter none --out validation/logs/everyday_webq_on_none_<date>.json

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import re
import sys
import time
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

__wiring__ = "STANDALONE"


class NoModel:
    """No model in the programs seat: every question the host does not read gets no plan."""
    name = "none"

    def emit(self, prompt: str, max_new_tokens: int = 300, **kw) -> str:
        return "no plan\n"


def model_450m(export: str, adapter: str, world):
    from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk
    from cubbyllm.reasoning.slots import SlotEmitter
    from cubby_serve.build import slot_index
    trunk = Cubby450mTrunk(export, os.path.join(export, "tokenizer.json"), {"programs": adapter})
    em = SlotEmitter(Cubby450mEmitter(trunk), index=None)
    em.index = slot_index(world)
    return em


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(the|a|an|of|language|time zone|timezone|university|college)\b", " ", s)
    return " ".join(re.findall(r"[a-z0-9]+", s))


def matches(answer: str, golds: list[str]) -> bool:
    """A spoken answer agrees with a listed one: the same words, one inside the other, or the same year."""
    a = _norm(answer)
    if not a:
        return False
    for g in golds:
        n = _norm(g)
        if not n:
            continue
        if a == n or (len(a) > 3 and a in n) or (len(n) > 3 and n in a):
            return True
        ya, yg = re.findall(r"\b(1\d{3}|20\d{2})\b", a), re.findall(r"\b(1\d{3}|20\d{2})\b", n)
        if ya and yg and ya[0] == yg[0]:
            return True
    return False


def load(path: str | None, stack: bool) -> list[dict]:
    if stack:
        from validation.stack_answers import QUESTIONS
        return [{"problem": q, "answers": [], "region": "stack"} for q in QUESTIONS]
    from cubbyllm.local_paths import expand
    path = path or expand("$CUBBY_DATASETS/webquestions/webq_na_eu.csv")
    return [{"problem": r["problem"], "answers": [x.strip() for x in r["answers"].split("|")], "region": r["region"]}
            for r in csv.DictReader(open(path, encoding="utf-8"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--everyday", choices=("on", "off"), default="on")
    ap.add_argument("--emitter", choices=("none", "450m"), default="none")
    ap.add_argument("--export", default=os.environ.get("CUBBY_450M_EXPORT"),
                    help="the 450M grilly export (default $CUBBY_450M_EXPORT); --emitter 450m only")
    ap.add_argument("--adapter", default=os.environ.get("CUBBY_450M_EMITTER"),
                    help="the programs adapter (default $CUBBY_450M_EMITTER); --emitter 450m only")
    ap.add_argument("--questions", default=None, help="a csv (problem, answers); default WebQuestions NA/EU")
    ap.add_argument("--stack", action="store_true", help="the real stack's 20 questions instead")
    ap.add_argument("--parsed-only", action="store_true", help="only the questions the everyday table reads")
    ap.add_argument("--online", action="store_true",
                    help="the Wikidata source live, as the server mounts it (a miss is looked up once and cached), "
                         "instead of its cache only")
    ap.add_argument("--source", choices=("cache", "off"), default="cache",
                    help="the Wikidata source from its cache (the stack offline), or no source at all (order 0037, item 3: "
                         "a snapshot's changing value stands, as of its year, only when the source gives none)")
    ap.add_argument("--tense", choices=("on", "off"), default="on",
                    help="the tense check (order 0012: 'who rules X?' on a reign that ended is not answered); on, as "
                         "the server runs since 2026-10-24 (Nick: on always); off is the stack as recorded before")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.emitter == "450m" and not (a.export and a.adapter):
        ap.error("--emitter 450m needs --export and --adapter (or $CUBBY_450M_EXPORT and $CUBBY_450M_EMITTER)")

    from cubbyllm.worlds import wikikg as wk
    from cubbyllm.worlds.hdc_store import HdcWorld, Stores
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.everyday import EverydayForms
    rows = load(a.questions, a.stack)
    forms = EverydayForms()
    if a.parsed_only:
        rows = [r for r in rows if forms.parse(r["problem"]) is not None]
    if a.n:
        rows = rows[:a.n]
    t0 = time.perf_counter()
    wk.ensure_data(("triplets",))
    wiki = wk.wiki_world()
    world = Stores(wiki, HdcWorld())
    em = NoModel() if a.emitter == "none" else model_450m(a.export, a.adapter, wiki)   # the names, as the server's index
    source = "wikidata-offline"
    if a.online:                                         # as `cubby_serve.cli` builds it for `--ask wikidata`
        from cubbyllm.sources.wikidata import WikidataSource
        from cubby_serve.sources.http import wikidata as wikidata_fetch
        source = WikidataSource(offline=False, use_choices=True, fetch=wikidata_fetch())
    if a.source == "off":
        from cubbyllm.reasoning.loci_reader import NoSource
        source = NoSource()
    loop = AskLoop(em, world=world, source=source, lexicon=True, history_path=None,
                   everyday=a.everyday == "on", tense=a.tense == "on")
    print(f"loaded in {time.perf_counter() - t0:.0f}s; {len(rows)} questions", flush=True)
    out, c = [], collections.Counter()
    for i, r in enumerate(rows):
        q = r["problem"]
        p = forms.parse(q)
        t1 = time.perf_counter()
        try:
            rec = loop.ask(q)
        except Exception as e:                           # noqa: BLE001 -- a crash is a row, not the end of the run
            rec = {"answer": None, "verified": False, "reason": f"error: {str(e)[:160]}"}
        said = rec.get("reply") or rec.get("answer")
        ok = matches(rec.get("answer") or "", r["answers"]) if rec.get("verified") and r["answers"] else None
        row = {"question": q, "region": r["region"], "golds": r["answers"][:5], "form": p[1] if p else None,
               "seed": rec.get("seed"), "plan": rec.get("plan"), "verified": bool(rec.get("verified")),
               "answer": rec.get("answer"), "said": said, "tier": rec.get("tier"), "reason": rec.get("reason"),
               "match": ok, "wall_s": round(time.perf_counter() - t1, 2)}
        out.append(row)
        c["asked"] += 1
        c["spoken"] += row["verified"]
        c["spoken_by_form"] += row["verified"] and bool(p)
        c["match"] += ok is True
        c["mismatch"] += ok is False
        if not row["verified"]:
            c["refused:" + str(row["reason"]).split(":")[0]] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}: spoken {c['spoken']}, match {c['match']}, mismatch {c['mismatch']}", flush=True)
    summary = {"what": "exp_everyday_webq", "everyday": a.everyday, "emitter": a.emitter, "online": a.online,
               "source": a.source, "tense": a.tense,
               "namesake_relation": os.environ.get("CB_NAMESAKE_RELATION") != "0",
               "source_over_snapshot": os.environ.get("CB_SOURCE_OVER_SNAPSHOT") != "0",
               "namesake_tier": os.environ.get("CB_NAMESAKE_TIER") == "1",
               "questions": "stack 20" if a.stack else (a.questions or "webq_na_eu.csv"), "parsed_only": a.parsed_only,
               "counts": dict(c), "by_form": dict(collections.Counter(r["form"] for r in out if r["verified"])),
               "emitter_calls": getattr(loop.emitter, "calls", None), "wall_s": round(time.perf_counter() - t0, 1)}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": out}, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
