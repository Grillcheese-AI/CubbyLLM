import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\worlds_ingest.py"
s = open(p, encoding="utf-8").read()
reps = [
('''def run(root, only=None, frames=False, release=False, log=print, plan=None) -> list[dict]:''',
 '''def run(root, only=None, frames=False, release=False, log=print, plan=None, mentions=None, tracker=TRACKER) -> list[dict]:
    """`plan`: the decisions (default: the local manifest's); `mentions`: the names a passage index matches (default:
    the wiki world's); `tracker`: the rights tracker (a missing file: every work cleared, none marked fiction)."""'''),
('''    ctx = wa.Context(root, reg, Release(only=release) if TRACKER.exists() or release else None, log)''',
 '''    ctx = wa.Context(root, reg, Release(tracker, only=release) if pathlib.Path(tracker).exists() or release else None, log)
    ctx._mentions = mentions'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
