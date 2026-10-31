import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\book_frames.py"
s = open(p, encoding="utf-8").read()
old = '''    p = _BROKEN.sub("", m["p"].strip())
    return None if p.split()[0] in _NOT_PLACE or re.search(r"[A-Za-z]\\d|\\d[A-Za-z]", p) or _HEADING.search(p) else p'''
new = '''    return clean_place(m["p"])


def clean_place(p: str) -> str | None:
    """`_place`'s rules on any place string (the bulk pass applies them to reads stored before a rule came in)."""
    p = _BROKEN.sub("", (p or "").strip())
    return None if not p or p.split()[0] in _NOT_PLACE or re.search(r"[A-Za-z]\\d|\\d[A-Za-z]", p) or _HEADING.search(p) else p'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\books_bulk_read.py"
s = open(p, encoding="utf-8").read()
old = '''    for book, _pid, off, subj, code, value, by, sentence, wh in rows:
        if (norm(subj), book) in unfit:                            # the same-name guard: another person of that name
            continue'''
new = '''    for book, _pid, off, subj, code, value, by, sentence, wh in rows:
        if (norm(subj), book) in unfit:                            # the same-name guard: another person of that name
            continue
        if br.LABEL.get(code, code) in PLACES:                     # today's place rules on reads stored before them
            value = bf.clean_place(value)
            if value is None:
                continue'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''ONE_VALUE = {'''
assert s.count(old) == 1
s = s.replace(old, '''PLACES = {"place of birth", "place of death", "place of burial", "capital", "location", "part of"}
ONE_VALUE = {''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
