import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\bench_loci_learn.py"
s = open(p, encoding="utf-8").read()
reps = [
('''    rows = hdc.db.execute(f"SELECT subj, rel, obj FROM facts WHERE rel IN ({','.join('?' * len(rels))}) ORDER BY RANDOM() LIMIT ?",
                          (*rels, n * 30)).fetchall()''',
 '''    rows = hdc.db.execute(f"SELECT subj, rel, obj FROM facts WHERE rel IN ({','.join('?' * len(rels))}) ORDER BY rowid",
                          rels).fetchall()
    rows = rng.sample(rows, min(len(rows), n * 30))              # seeded: both phases ask the same questions'''),
('''                         "answer": rec.get("answer"), "reason": rec.get("reason"), "learned": len(got),''',
 '''                         "answer": rec.get("answer"), "reason": rec.get("reason"), "learned": len(got),
                         "reads": [x.get("answer") for x in rec.get("loci") or []],
                         "stored": (rec.get("refused") or {}).get("stored") if isinstance(rec.get("refused"), dict) else None,'''),
('''                         "verified": bool(rec["verified"]), "answer": rec.get("answer"), "reason": rec.get("reason"),
                         "source_calls": sum(src.asked.values()) - before,''',
 '''                         "verified": bool(rec["verified"]), "answer": rec.get("answer"), "reason": rec.get("reason"),
                         "reads": [x.get("answer") for x in rec.get("loci") or []],
                         "source_calls": sum(src.asked.values()) - before,'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
