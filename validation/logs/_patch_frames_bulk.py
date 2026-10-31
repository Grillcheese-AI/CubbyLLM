import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\exp_books_bench.py"
s = open(p, encoding="utf-8").read()
anchor = "class Offline:"
new = '''HDC_GOLD = {"capital": "capital", "part of": "part of", "located in the administrative territorial entity": "location",
            "country": "location"}


def frames_bulk(worlds: str, pairs, hdc_path=None, within=None) -> dict:
    """The frames over every paragraph of every book (order 0013's bulk worlds, after the same-name guard), against
    the wiki world's values and the hdc world's capitals, parts and locations: agreement per half and per relation."""
    import sqlite3
    gold = collections.defaultdict(set)
    for (s, rel), vs in pairs.items():
        gold[(norm(s), rel)] |= set(vs)
    if hdc_path and os.path.exists(hdc_path):
        db = sqlite3.connect(f"file:{hdc_path}?mode=ro", uri=True)
        q = f"SELECT skey, rel, obj FROM facts WHERE rel IN ({','.join('?' * len(HDC_GOLD))})"
        for sk, rel, o in db.execute(q, list(HDC_GOLD)):
            gold[(sk, HDC_GOLD[rel])].add(o)
    stats, per, rows = {h: collections.Counter() for h in ("dev", "test")}, collections.defaultdict(collections.Counter), []
    for f in sorted(glob.glob(os.path.join(worlds, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("inverse") or not r.get("rel") or " is the " not in r["fact"]:
                continue
            value, rel, subj = r["fact"].split(" is the ", 1)[0], r["rel"], r["subject"]
            a = agree(rel, value, gold.get((norm(subj), rel), set()), within)
            per[rel]["facts"] += 1
            if a is None:
                continue
            h = half(subj)
            stats[h][a] += 1; per[rel][a] += 1
            rows.append({"subject": subj, "relation": rel, "value": value, "verdict": a, "half": h, "tier": r["tier"]})
    tot = sum(stats.values(), collections.Counter())
    ok = lambda c: round((c["agree"] + c["finer"] + c["coarser"]) / max(1, sum(c[k] for k in ("agree", "finer", "coarser", "contradict"))), 3)   # noqa: E731
    return {"compared": sum(tot[k] for k in ("agree", "finer", "coarser", "contradict")), "all": dict(tot),
            "agreement": {h: ok(c) for h, c in list(stats.items()) + [("all", tot)]},
            "per_relation": {k: dict(v) for k, v in sorted(per.items())}, "rows": rows}


'''
assert s.count(anchor) == 1
s = s.replace(anchor, new + anchor, 1)
a_ = '''    rep["bench"] = bench('''
assert s.count(a_) == 1
s = s.replace(a_, '''    if a.frames_bulk:                                                    # 0012's frames line on >= 200 triples
        rep["frames_bulk"] = frames_bulk(a.frames_bulk, pairs, a.hdc, within)
        print(f"frames over every book: {rep['frames_bulk']['compared']} compared, agreement {rep['frames_bulk']['agreement']}",
              flush=True)
    rep["bench"] = bench(''')
a_ = '''    ap.add_argument("--bulk", default=None'''
assert s.count(a_) == 1
s = s.replace(a_, '''    ap.add_argument("--frames-bulk", default=None, help="order 0013's worlds dir: frames over every book, against the gold")
    ap.add_argument("--bulk", default=None''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\\n"))
