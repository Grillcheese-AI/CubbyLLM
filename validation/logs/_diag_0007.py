"""Scratch (order 0007): the reader on facts phase 1 learned -- what it reads, against every stored value."""
import collections, json, os, sys
R = r"C:\Users\grill\Documents\GitHub\CubbyLLM"
for p in (R, R + r"\standin", R + r"\standin\data", R + r"\validation"):
    sys.path.insert(0, p)
from cubbyllm.reasoning.planner import parse_fact, normalize
from loci_reader import LociReader
import torch
torch.set_num_threads(2)
rep = json.load(open(R + r"\validation\logs\bench_loci_learn_2026-10-22.json", encoding="utf-8"))
learned = [json.loads(l)["fact"] for l in open(R + r"\standin\data\out\bench_loci_learned.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for f in learned:
    t = parse_fact(f)
    if t:
        by[normalize(t.subj)].append((t.subj, t.rel, t.obj))
r = LociReader()
rows = [x for x in rep["phase1"]["rows"] if x["path"] == "loci_after_lookup"][:12]
for x in rows:
    facts = by.get(normalize(x["entity"]), [])
    want = [o for s, rel, o in facts if r.shard_relation(rel) == r.shard_relation(x["relation"])]
    got = r.read(x["entity"], x["relation"], facts)
    print(x["relation"], "|", x["entity"], "| read", repr(got.get("answer")), "| lam", [round(v, 2) for v in (got.get("lam") or []) if v is not None][:4],
          "| stored", want[:4], "| facts in scene", len([f for f in facts if r.shard_relation(f[1])]), flush=True)
