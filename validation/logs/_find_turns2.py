import re
X = r"E:\datasets\gutenberg_extra"
D = r"E:\datasets\domains"
CASES = [
    ("Madame Bovary", X + r"\2413.txt", [r"silent\s+walls", r"great\s+coldness", r"shiver"]),
    ("Great Expectations", X + r"\1400.txt", [r"could\s+not\s+have\s+spoken\s+one\s+word", r"blacksmith.s\s+hammer", r"was\s+wrecked", r"Miss\s+Havisham.s\s+intentions"]),
    ("Romeo and Juliet (modern)", X + r"\1513.txt", [r"black\s+fate", r"fortune.s\s+fool"]),
    ("Monte Cristo", D + r"\fiction\1184.txt", [r"Marriage\s+Feast", r"Château\s+d.If", r"Cemetery\s+of\s+the", r"The\s+Escape"]),
]
HEAD = re.compile(r"(?im)^\s*(chapter|act|scene|part)\b[^\n]{0,60}$")
for name, path, pats in CASES:
    t = open(path, encoding="utf-8", errors="replace").read()
    s = t.find("*** START"); e = t.find("*** END")
    body = t[s if s > 0 else 0: e if e > 0 else len(t)]
    n = len(body)
    print(f"== {name} ({n:,} chars)")
    for p in pats:
        ms = list(re.finditer(p, body))
        if not ms:
            print(f"   NOT FOUND {p}")
            continue
        for m in ms[:3]:
            i = m.start()
            heads = [h.group(0).strip()[:50] for h in HEAD.finditer(body, 0, i)]
            print(f"   {p} at {i/n:.3f} under {heads[-1] if heads else '-'}")
