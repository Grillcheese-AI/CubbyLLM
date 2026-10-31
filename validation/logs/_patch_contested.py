import ast
R = r"C:\Users\grill\Documents\GitHub\CubbyLLM"
p = R + r"\standin\ask.py"
s = open(p, encoding="utf-8").read()
reps = [
('''        undated = self._undated(question, lr) if lr.result.verified and not why and not snap else None
        if why:''',
 '''        undated = self._undated(question, lr) if lr.result.verified and not why and not snap else None
        h = lr.result.trace[-1] if lr.result.verified and lr.result.trace else None
        contested = h is not None and h.triple is not None and (normalize(h.triple.subj), normalize(h.triple.rel)) in self.contested
        if contested:                                     # the bulk pass held two values of it (order 0013): never said
            rec.update(answer=None, verified=False, reason="contested",
                       refused={"answer": lr.result.answer, "why": "the sources disagree on it"})
        elif why:'''),
('''    def load_learned(self, path) -> int:''',
 '''    def load_held(self, path) -> int:
        """The facts a bulk pass held (two values of a relation that has one: order 0013's `_held.jsonl`), as a veto:
        an answer whose last fact is about a held (subject, relation) is refused `contested`, whoever states it --
        a live read that sees one value only does not make the other go away."""
        from cubbyllm.reasoning.planner import parse_fact
        from cubbyllm.reasoning.sleep import load_facts
        n = 0
        for key, r in load_facts(pathlib.Path(path)).items():
            t = parse_fact(key, known={r["rel"]} if r.get("rel") else None) if r.get("rel") else parse_fact(key)
            subj = r.get("subject") or (t.subj if t else None)
            if subj and (r.get("rel") or t):
                self.contested.add((normalize(subj), normalize(r.get("rel") or t.rel))); n += 1
        return n

    def load_learned(self, path) -> int:'''),
('''        self.books = books
        if books is not None''', '''        self.books = books
        self.contested: set[tuple[str, str]] = set()     # (subject, relation) a bulk pass held: `load_held`
        if books is not None'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = R + r"\cubbyllm\reasoning\sleep.py"
s = open(p, encoding="utf-8").read()
a = '''    "time_undated": ('''
assert s.count(a) == 1
s = s.replace(a, '''    "contested": ("latent_audit", "a bulk pass holds two values of the fact's relation (order 0013); never said until one is settled"),
    "time_undated": (''')
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = R + r"\validation\exp_books_bench.py"
s = open(p, encoding="utf-8").read()
a = '''                 if not os.path.basename(f).startswith("_")) if bulk else None                   # (never the held)'''
assert s.count(a) == 1
s = s.replace(a, '''                 if not os.path.basename(f).startswith("_")) if bulk else None                   # (never the held)
    held = loop.load_held(os.path.join(bulk, "_held.jsonl")) if bulk else None     # the held: a veto, never an answer''')
a = '''    return {"bulk_facts_loaded": loaded,'''
assert s.count(a) == 1
s = s.replace(a, '''    return {"bulk_facts_loaded": loaded, "bulk_held_loaded": held,''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
