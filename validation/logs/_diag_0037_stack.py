"""Scratch (untracked): the stack's 20 in order, full records for the questions that changed on 2026-10-27.
    python validation/logs/_diag_0037_stack.py   (env: CB_SOURCE_OVER_SNAPSHOT, CB_NAMESAKE_RELATION)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cubbyllm.worlds import wikikg as wk
from cubbyllm.worlds.hdc_store import HdcWorld, Stores
from cubbyllm.reasoning.ask import AskLoop
from validation.exp_everyday_webq import NoModel
from validation.stack_answers import QUESTIONS

WATCH = ("Italie", "World War", "France", "Canada")
wk.ensure_data(("triplets",))
wiki = wk.wiki_world()
world = Stores(wiki, HdcWorld())
loop = AskLoop(NoModel(), world=world, source="wikidata-offline", lexicon=True, history_path=None)
print("flags", {k: os.environ.get(k) for k in ("CB_SOURCE_OVER_SNAPSHOT", "CB_NAMESAKE_RELATION")})
for q in QUESTIONS:
    rec = loop.ask(q)
    if any(w in q for w in WATCH):
        keep = {k: rec.get(k) for k in ("answer", "verified", "reason", "tier", "reply", "trace", "refused", "plan", "seed",
                                        "over_snapshot", "snapshot_stood", "namesakes", "aliased", "entities", "candidates")}
        keep["learned"] = [(p["fact"], p["status"]) for p in rec.get("learned") or []
                           if "capital" in p["fact"] or "end time" in p["fact"] or "time" in p["fact"]][:10]
        print("###", q, json.dumps(keep, ensure_ascii=False, default=str)[:2500], flush=True)
    else:
        print("   ", q, "->", rec.get("reply") or rec.get("answer"), rec.get("reason"), flush=True)
