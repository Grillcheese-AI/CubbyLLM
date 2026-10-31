"""Scratch (order 0007): replay the loop's reads -- the same facts, the same entity -- and print the scene's tail."""
import collections, json, sys
R = r"C:\Users\grill\Documents\GitHub\CubbyLLM"
for p in (R, R + r"\standin", R + r"\standin\data", R + r"\validation"):
    sys.path.insert(0, p)
from cubbyllm.reasoning.planner import parse_fact, normalize
from loci_reader import LociReader
import torch
torch.set_num_threads(2)
learned = [json.loads(l)["fact"] for l in open(R + r"\standin\data\out\bench_loci_learned.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for f in learned:
    t = parse_fact(f)
    if t:
        by[normalize(t.subj)].append((t.subj, t.rel, t.obj))
r = LociReader()
for ent in ["La Coma i la Pedra", "Sedalia Township", "Max Ilgner", "Harpdog Brown"]:
    facts = by[normalize(ent)]
    rows = [(s, o, r.shard_relation(rel)) for s, rel, o in facts if r.shard_relation(rel)]
    rel = "located in the administrative territorial entity" if "Township" in ent or "Coma" in ent else "country of citizenship"
    ids, pos, roles = r.scene(ent, rel, rows)
    print("---", ent, "| rows", rows)
    print("   scene tail:", repr(r.tk.decode(ids[-40:])))
    got = r.read(ent, rel, facts)
    print("   read", repr(got.get("answer")), "lam", [round(v, 2) for v in got.get("lam") or [] if v is not None], got.get("why"), flush=True)
