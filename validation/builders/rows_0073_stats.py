"""Order 0073, step 3: what a build of `long_rows_0073` reports about its rows (counts only, no row text).

`summary(rows)` for one set of rows: the bounds said with a number other than 0 by side and by place (the lower
bounds' share is what the order holds within 45 to 55%), where the tied pairs sit by number of parts, each
connective said first or after its partner, how the checks were worded and how many carry a ``not`` in arm W's
form, what the replies reach (N - 1, N, N + 1 for a numbered part; a tied pair's four combinations) and how many
one-reversal mutants they tell apart. `shapes` counts the held-out rows whose composition without its values is a
train row's. `digest` is the SHA-256 of a set's lines as written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import view_draw as D  # noqa: E402
import view_said as S  # noqa: E402

__wiring__ = "STANDALONE"

PLACES = ("size", "count", "position", "filter")


def line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("".join(line(r) for r in rows).encode("utf-8")).hexdigest()


def tally(rows: list[dict]) -> Counter:
    t = Counter()
    for r in rows:
        D.tally_row(t, [s["w"] for s in r["said"]])
    return t


def _sides(t: Counter, place: str | None) -> dict:
    n = {side: sum(c for (p, s), c in t.items() if s == side and place in (None, p))
         for side in ("lower", "upper", "exactly", "other")}
    share = D.lower_share(t, place)
    return dict(n, **{"lower share": None if share is None else round(share, 4)})


def bounds(rows: list[dict]) -> dict:
    t = tally(rows)
    return dict({"all": _sides(t, None)}, **{p: _sides(t, p) for p in PLACES})


def places(rows: list[dict]) -> dict:
    """By number of parts: the rows, and how many have a tied pair at each place (parts numbered from 1)."""
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["n_parts"], Counter())
        d["rows"] += 1
        for g in r["groups"]:
            if g[0] != "one":
                d[f"pair at {min(g[1:]) + 1}-{max(g[1:]) + 1}"] += 1
    return {str(k): dict(sorted(v.items())) for k, v in sorted(out.items())}


def connectives(rows: list[dict]) -> dict:
    c = Counter()
    for r in rows:
        for s in r["segments"]:
            if s.get("join"):
                c[f"{r['lang']} {s['join'][0]} {'first' if s['join'][1] == 'next' else 'after'}"] += 1
    return dict(sorted(c.items()))


def wording(rows: list[dict]) -> dict:
    how, nots, q = Counter(), Counter(), Counter()
    for r in rows:
        for s in r["said"]:
            how[s["how"]] += 1
            nots[str(S.marks(s["w"]))] += 1
            q[s["m"]["q"]] += 1
    return {"how": dict(sorted(how.items())), "not in arm W's form": dict(sorted(nots.items())),
            "arm M's quantifier": dict(sorted(q.items()))}


def reach(rows: list[dict]) -> dict:
    """``alone``: on a reply where the row's verdict turns on that part or pair alone (`view_tell.extras`, 2)."""
    names = [f"{a}{b}" for a in ("numbered parts", "values reached", "all three values", "tied pairs",
                                 "combinations reached", "all four combinations") for b in ("", ", alone")
             if (a, b) not in (("numbered parts", ", alone"), ("tied pairs", ", alone"))]
    t = Counter({k: 0 for k in names + ["replies", "replies added", "replies built to fail"]})
    for r in rows:
        for what, one, every, got in ([("numbered parts", "values reached", "all three values", g)
                                       for _, g in r["reach"]["numbers"]]
                                      + [("tied pairs", "combinations reached", "all four combinations", g)
                                         for _, g in r["reach"]["groups"]]):
            t[what] += 1
            t[one] += sum(1 for x in got if x)
            t[one + ", alone"] += sum(1 for x in got if x == 2)
            t[every] += all(got)
            t[every + ", alone"] += all(x == 2 for x in got)
        t["replies"] += len(r["replies"])
        t["replies added"] += sum(1 for x in r["replies"] if x.get("extra"))
        t["replies built to fail"] += sum(1 for x in r["replies"] if not x["held"])
    return dict(t)


def mutants(rows: list[dict]) -> dict:
    t = Counter({k: 0 for k in ("parts", "parts told", "parts told alone", "groups", "groups told", "groups told alone")})
    for r in rows:
        t.update(r["mutants"])
    return dict(t)


def shape(row: dict) -> str:
    """A row's composition without its values: each folded part's slots and comparisons, and the groups."""
    def bare(m):
        out = {k: v for k, v in m.items() if k not in ("v", "val", "where")}
        if "where" in m:
            out["where"] = [[c["col"], c["op"]] for c in m["where"]]
        return out
    return json.dumps([[bare(s["m"]) for s in row["said"]], row["groups"]], sort_keys=True)


def shapes(held: list[dict], train: list[dict]) -> dict:
    seen, out = {shape(r) for r in train}, {}
    for r in held:
        d = out.setdefault(f"{r['split']} {r['n_parts']} parts", Counter({"rows": 0, "shape in train": 0}))
        d["rows"] += 1
        d["shape in train"] += shape(r) in seen
    return {k: dict(v) for k, v in sorted(out.items())}


def summary(rows: list[dict]) -> dict:
    return {"rows": len(rows), "by_lang": dict(Counter(r["lang"] for r in rows)),
            "by_parts": {str(k): v for k, v in sorted(Counter(r["n_parts"] for r in rows).items())},
            "segments_by_kind": dict(Counter(s["kind"] for r in rows for s in r["segments"])),
            "groups": dict(Counter(g[0] for r in rows for g in r["groups"])), "tied pairs by place": places(rows),
            "connectives": connectives(rows), "bounds": bounds(rows), "wording": wording(rows),
            "replies reach": reach(rows), "mutants": mutants(rows)}


__all__ = ["bounds", "digest", "line", "shape", "shapes", "summary", "tally"]
