"""Scratch (untracked): the scoreboard/stack questions that changed with CB_SOURCE_NAMESAKES=1 -- full records."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cubbyllm.worlds import wikikg as wk
from cubbyllm.worlds.hdc_store import HdcWorld, Stores
from cubbyllm.reasoning.ask import AskLoop
from cubbyllm.sources.wikidata import WikidataSource
from cubby_serve.sources.http import wikidata as wikidata_fetch
from validation.exp_everyday_webq import NoModel

QS = ["what is the capital city of germany now?", "what is the currency of spain called?", "what is the currency in italy?",
      "what continent is belize located in?", "where did richard nixon die?", "what kind of government does spain have now?",
      "Who wrote Pride and Prejudice?", "Who painted the Mona Lisa?"]
wk.ensure_data(("triplets",))
wiki = wk.wiki_world()
hdc = HdcWorld()
for flag in ("1", "0"):
    os.environ["CB_SOURCE_NAMESAKES"] = flag
    src = WikidataSource(offline=False, use_choices=True, fetch=wikidata_fetch())
    loop = AskLoop(NoModel(), world=Stores(wiki, hdc), source=src, lexicon=True, history_path=None)
    print(f"=============== flag {flag}")
    for q in QS:
        rec = loop.ask(q)
        keep = {k: rec.get(k) for k in ("answer", "verified", "reason", "reply", "seed", "plan", "trace", "refused", "candidates", "over_snapshot", "snapshot_stood")}
        ns = rec.get("namesakes") or {}
        keep["namesakes"] = {k: ns.get(k) for k in ("rule", "item", "items", "how")}
        keep["answers"] = [(a["item"], a["label"], a["said"], a["reason"]) for a in rec.get("answers") or []]
        keep["learned"] = [(p["fact"], p["status"]) for p in rec.get("learned") or [] if any(w in p["fact"] for w in ("capital", "currency", "continent", "death", "author", "creator"))][:8]
        keep["source_decision"] = src.namesake_decision(rec.get("seed") or "")
        keep["hdc"] = {"namesakes": list(hdc.namesakes(rec.get("seed") or "")), "decision": (hdc.decision(rec.get("seed") or "") or {}).get("fit"), "withheld": hdc.withheld(rec.get("seed") or "")}
        print("###", q, json.dumps(keep, ensure_ascii=False, default=str)[:2200], flush=True)
