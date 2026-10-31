import re, sys
BASE = r"E:\datasets\domains"
CASES = [
    ("Pride and Prejudice", r"factual_books\1342.txt", ["absolutely ashamed of herself", "How despicably", "prided myself on my discernment"]),
    ("Romeo and Juliet", r"drama\1112.txt", ["black fate", "fortune's fool", "fortune\u2019s fool"]),
    ("Frankenstein (1818)", r"fiction\41445.txt", ["beauty of the dream vanished", "breathless horror and disgust", "infusing life into an inanimate body"]),
]
HEAD = re.compile(r"^\s*(chapter|act|scene|letter|volume)\b[^\n]{0,40}$", re.I | re.M)
for name, rel, needles in CASES:
    t = open(BASE + "\\" + rel, encoding="utf-8", errors="replace").read()
    n = len(t)
    print(f"== {name}  ({n:,} chars)")
    for nd in needles:
        i = t.find(nd)
        if i < 0:
            print(f"   NOT FOUND: {nd!r}")
            continue
        heads = [m.group(0).strip() for m in HEAD.finditer(t, 0, i)]
        ctx = t[max(0, i - 60): i + 110].replace("\n", " ")
        print(f"   found {nd!r} at {i/n:.3f} of the book, under {heads[-2:] if heads else '-'}")
        print(f"      ...{ctx}...")
