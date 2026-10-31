import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\book_frames.py"
s = open(p, encoding="utf-8").read()
old = '''def _place(m) -> str | None:
    """The place a frame read, or None: a frame word first ('The'), or a word with a digit glued in ('Egypt216Lorenzo':
    a page number and a name run together by the PDF's text)."""
    p = m["p"].strip()
    return None if p.split()[0] in _NOT_PLACE or re.search(r"[A-Za-z]\\d|\\d[A-Za-z]", p) else p'''
new = '''_BROKEN = re.compile(r"(?<=[a-z])-(?=[a-z])")                 # a line-end hyphen the PDF's text kept: 'Mem-phis'
_HEADING = re.compile(r"[A-Z]{3,}[a-z]|\\b[A-Z]{4,}\\b")        # a page heading in capitals run into a name


def _place(m) -> str | None:
    """The place a frame read, or None: a frame word first ('The'), a word with a digit glued in ('Egypt216Lorenzo':
    a page number and a name run together by the PDF's text), or a heading in capitals run into it ('BICULTURAL
    KINGDOMAlexandria'). A word the PDF broke at a line end is joined ('Mem-phis', 'The-bes'); a name hyphenated
    on purpose keeps its capital after the hyphen ('Horn-Bad Meinberg') and is left alone (2026-10-22, the bulk pass)."""
    p = _BROKEN.sub("", m["p"].strip())
    return None if p.split()[0] in _NOT_PLACE or re.search(r"[A-Za-z]\\d|\\d[A-Za-z]", p) or _HEADING.search(p) else p'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
