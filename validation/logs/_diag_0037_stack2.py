"""Scratch (untracked): the stack's 20 in order; for the watched questions, every source call, every gated fact, and
the hdc world's decision on the seed."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cubbyllm.worlds import wikikg as wk
from cubbyllm.worlds.hdc_store import HdcWorld, Stores
from cubbyllm.reasoning.ask import AskLoop
from validation.exp_everyday_webq import NoModel
from validation.stack_answers import QUESTIONS

WATCH = ("Italie", "World War")
wk.ensure_data(("triplets",))
wiki = wk.wiki_world()
hdc = HdcWorld()
world = Stores(wiki, hdc)
loop = AskLoop(NoModel(), world=world, source="wikidata-offline", lexicon=True, history_path=None)
src = loop.source
calls = []
orig = src.facts


def facts(entity, relations=None, via=None, **kw):
    got = list(orig(entity, relations=relations, via=via, **kw))
    calls.append((entity, relations, via, len(got), dict(getattr(src, "last", None) or {}), [str(g) for g in got if "capital" in str(g) or "time" in str(g)][:12]))
    return got


src.facts = facts
print("flags", {k: os.environ.get(k) for k in ("CB_SOURCE_OVER_SNAPSHOT", "CB_NAMESAKE_RELATION")})
for q in QUESTIONS:
    calls.clear()
    rec = loop.ask(q)
    if any(w in q for w in WATCH):
        print("###", q, "->", rec.get("reply") or rec.get("answer"), rec.get("reason"), rec.get("refused"))
        print("   seed", rec.get("seed"), "plan", rec.get("plan"), "entities", rec.get("entities"), "aliased", rec.get("aliased"))
        print("   hdc label", hdc.label(rec.get("seed") or ""), "namesakes", list(hdc.namesakes(rec.get("seed") or "")),
              "decision", json.dumps(hdc.decision(rec.get("seed") or ""), default=str)[:400], "withheld", hdc.withheld(rec.get("seed") or ""))
        print("   learned", [(p["fact"], p["status"]) for p in rec.get("learned") or [] if "capital" in p["fact"] or "time" in p["fact"]][:40])
        print("   trace", rec.get("trace"), "over", rec.get("over_snapshot"), rec.get("snapshot_stood"), "candidates", rec.get("candidates"))
        for c in calls:
            print("   source call", json.dumps(c, default=str, ensure_ascii=False)[:1500])
        print("   known end time?", "end time" in getattr(loop.known, "_rels", set()), "capital?", "capital" in getattr(loop.known, "_rels", set()))
    else:
        print("   ", q, "->", rec.get("reply") or rec.get("answer"), rec.get("reason"), "calls", len(calls), flush=True)
