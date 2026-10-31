import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\standin\worlds_ingest.py"
s = open(p, encoding="utf-8").read()
old = '''    fiction, reality, reasons = [], [], {}
    for p in wa.text_files(pathlib.Path(entry.path)):
        mark = ctx.release.fiction(p) if ctx.release else None
        if mark in ("yes", "no"):
            is_f, why = mark == "yes", f"the rights tracker marks it fiction={mark}"
        else:
            is_f, why = wa.looks_like_fiction(p.read_bytes()[:400_000].decode("utf-8", errors="replace"))
        (fiction if is_f else reality).append(p)
        reasons[why.split(",")[0] if mark else ("fiction by its text" if is_f else "reality by its text")] = \\
            reasons.get(why.split(",")[0] if mark else ("fiction by its text" if is_f else "reality by its text"), 0) + 1'''
new = '''    import collections
    fiction, reality, reasons = [], [], collections.Counter()
    for p in wa.text_files(pathlib.Path(entry.path)):
        mark = ctx.release.fiction(p) if ctx.release else None
        if mark in ("yes", "no"):
            is_f, why = mark == "yes", "the rights tracker's fiction mark"
        else:
            is_f, why = wa.looks_like_fiction(p.read_bytes()[:400_000].decode("utf-8", errors="replace"))[0], "its text"
        (fiction if is_f else reality).append(p)
        reasons[f"{'fiction' if is_f else 'reality'} by {why}"] += 1'''
assert s.count(old) == 1, "classify"
s = s.replace(old, new)
old = '''    out = {"classified": {"fiction": len(fiction), "reality": len(reality), "why": reasons}}'''
assert s.count(old) == 1
s = s.replace(old, '''    out = {"classified": {"fiction": len(fiction), "reality": len(reality), "why": dict(reasons)}}''')
old = '''    rows = run(a.root, a.only, a.frames, a.release, lambda m: print(m, flush=True))
    pathlib.Path(a.log).write_text(json.dumps({"datasets": rows, "counts": {s: sum(r["status"] == s for r in rows)
                                                                             for s in ("ingest", "excluded", "queued")}}, indent=1), encoding="utf-8")'''
new = '''    rows = run(a.root, a.only, a.frames, a.release, lambda m: print(m, flush=True))
    pathlib.Path(a.log).write_text(json.dumps(public(rows), indent=1), encoding="utf-8")


def public(rows) -> dict:
    """The report as it may be committed: a dataset by its manifest row and kind, never its name (collections and
    authors stay in the local manifest and registry); the reasons are the manifest's routes."""
    names = [d.dataset for d in wm.plan()]
    out = [{"row": names.index(r["dataset"]) + 2 if r["dataset"] in names else None, "tree": r["tree"], "status": r["status"],
            "reason": r["reason"] if r["status"] != "ingest" else None, "seconds": r.get("seconds"),
            "entries": r["entries"]} for r in rows]
    return {"datasets": out, "counts": {s: sum(r["status"] == s for r in rows) for s in ("ingest", "excluded", "queued")}}'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\n"))
