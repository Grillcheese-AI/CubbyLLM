"""Scratch (untracked): the two stack questions that changed on 2026-10-27, under each flag."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cubbyllm.worlds import wikikg as wk
from cubbyllm.worlds.hdc_store import HdcWorld, Stores
from cubbyllm.reasoning.ask import AskLoop
from validation.exp_everyday_webq import NoModel

QS = ["Quelle est la capitale de l'Italie ?", "What year did World War II end?"]
wk.ensure_data(("triplets",))
wiki_triples = None
for src_first, rule in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
    os.environ["CB_SOURCE_OVER_SNAPSHOT"] = src_first
    os.environ["CB_NAMESAKE_RELATION"] = rule
    wiki = wk.wiki_world()
    world = Stores(wiki, HdcWorld())
    loop = AskLoop(NoModel(), world=world, source="wikidata-offline", lexicon=True, history_path=None)
    print(f"=== source first {src_first}, rule {rule}")
    for q in QS:
        rec = loop.ask(q)
        keep = {k: rec.get(k) for k in ("answer", "verified", "reason", "tier", "reply", "trace", "refused",
                                        "over_snapshot", "snapshot_stood", "namesakes", "aliased", "entities")}
        keep["learned"] = [(p["fact"], p["status"]) for p in rec.get("learned") or []][:8]
        print(q, json.dumps(keep, ensure_ascii=False, default=str)[:1500])
