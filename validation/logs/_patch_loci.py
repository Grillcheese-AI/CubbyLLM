import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\loci_reader.py"
s = open(p, encoding="utf-8").read()
reps = [
# 2. several values -> a miss
('''        rows = [(s, o, self.shard_relation(r)) for s, r, o in facts if self.shard_relation(r)]
        if not rows:
            return {"answer": None, "miss": True, "why": "nothing written about it"}''',
 '''        rows = [(s, o, self.shard_relation(r)) for s, r, o in facts if self.shard_relation(r)]
        if not rows:
            return {"answer": None, "miss": True, "why": "nothing written about it"}
        if len({o for s, o, r in rows if r == R and normalize(s) == normalize(entity)}) > 1:
            # one address holds one value (the shard's facts): several written there superimpose ('lawyer' +
            # 'politician' read 'lawyerician', 2026-10-22), and choosing among them is the walk's, not the reader's
            return {"answer": None, "miss": True, "why": "several values for it: the walk's to choose"}'''),
('''        from cubbyllm.model.spatial import NO_PLACE
        from make_fact_shard import address
        torch = self.torch''', '''        from cubbyllm.model.spatial import NO_PLACE
        from cubbyllm.reasoning.planner import normalize
        from make_fact_shard import address
        torch = self.torch'''),
# 3. stop when the gate closes
('''            if j == 0 and (lam is None or lam < 0.5):
                return {"answer": None, "miss": True, "why": "the gate stayed shut", "lam": lams}
            if tok in (self.nl, self.eos):
                break''',
 '''            if j == 0 and (lam is None or lam < 0.5):
                return {"answer": None, "miss": True, "why": "the gate stayed shut", "lam": lams}
            if tok in (self.nl, self.eos) or lam is None or lam < 0.5:
                break                                     # the gate shut: what follows is the trunk's guess, not the fact''')
,
# 1. the plan's wording -> a label the shard knows
('''    reader, rel = loop.loci, rels[0]
    R = reader.shard_relation(rel)''',
 '''    reader, rel = loop.loci, rels[0]
    if reader.shard_relation(rel) is None:                # the question's own word ('administrative territorial
        named = [lab for res in getattr(loop, "resolvers", []) for lab in (getattr(res, "relations", None) or (lambda _r: []))(rel)]
        known = list(dict.fromkeys(lab for lab in named if reader.shard_relation(lab)))   # entity'): its label, if one
        if len(known) == 1:
            rel = known[0]
    R = reader.shard_relation(rel)'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
s = s.replace('''is one the shard never addressed, nothing is written about the entity, the copy's read-aware gate stays shut at the
first piece (lam below 0.5), or the first piece is the newline.''', '''is one the shard never addressed, nothing is written about the entity, the store holds several values for it (one
address holds one value), the copy's read-aware gate stays shut at the first piece (lam below 0.5), or the first piece
is the newline. A later piece is taken only while the gate stays open: once it shuts, the next token is the trunk's own
guess, not the fact.''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\n"))
