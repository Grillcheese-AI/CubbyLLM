"""r4_statements -- order 0046, R4: a relation's values with their rank and time, fetched by id and dated.

Wikidata5M keeps no ranks or qualifiers, so R4 reads the statements themselves (the order: "fetch the statements by id
(rank, start, end, point in time) through the source's cache, and record the date of the fetch"):
  out     the item's own statements of the property (``wbgetentities`` claims, through the source's cache)
  in      the statements naming the item as their value (one SPARQL query a (property, item): the subject, the rank,
          start, end and point in time); a query the service cannot answer is recorded as unavailable, never as none
  single  the property's single-value constraints (P2302 with Q19474404 "single-value", or Q52060874 "single-best-
          value"), from the property's own statements
  ends    an item's own end (R4's rule 1): the earliest year of its death (P570), dissolution (P576) or end time
          (P582), not deprecated; None if it states none; "unread" if the source did not answer
A value: {"v": item id or literal (None for "no value" / "unknown value"), "rank", "start", "end", "point"}, each time
as {"y": signed year, "m", "d", "prec"} (Wikidata's precision: 9 a year, 10 a month, 11 a day), and "replaced": True when
the statement carries a "replaced by" (P1366) qualifier. Entries written before "replaced" was read (no "r": 2) are
fetched again.
Kept in ``$CUBBY_DATA_ROOT/retrieval/r4_statements.jsonl``, one line per (item, property, direction), with its date.

Wired: STANDALONE (R4's bench).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

__wiring__ = "STANDALONE"

SPARQL = "https://query.wikidata.org/sparql"
TIME_Q = {"P580": "start", "P582": "end", "P585": "point"}
SINGLE = {"Q19474404", "Q52060874"}
RANKS = {"PreferredRank": "preferred", "NormalRank": "normal", "DeprecatedRank": "deprecated"}


def parse_time(t: str, prec: int | None = None) -> dict | None:
    """'+2001-05-00T00:00:00Z' -> {"y": 2001, "m": 5, "d": 0, "prec": ...}; the sign is the era."""
    t = str(t or "")
    if t[:1] not in "+-" or len(t) < 5:
        return None
    sign = -1 if t.startswith("-") else 1
    try:
        y, m, d = t[1:].split("T")[0].split("-")[:3]
        return {"y": sign * int(y), "m": int(m), "d": int(d), "prec": int(prec if prec is not None else 11)}
    except ValueError:
        return None


def values_of(claims: dict, pid: str) -> list[dict]:
    """The property's statements in `claims` (a ``wbgetentities`` claims object), as values."""
    out = []
    for st in claims.get(pid) or ():
        snak = st.get("mainsnak") or {}
        dv = (snak.get("datavalue") or {}).get("value")
        v = None
        if snak.get("snaktype", "value") == "value" and dv is not None:
            v = dv.get("id") if isinstance(dv, dict) and "id" in dv else \
                (dv.get("time") if isinstance(dv, dict) and "time" in dv else
                 (dv.get("amount") if isinstance(dv, dict) and "amount" in dv else
                  (dv.get("text") if isinstance(dv, dict) else dv)))
        val = {"v": v, "rank": st.get("rank", "normal")}
        if (st.get("qualifiers") or {}).get("P1366"):
            val["replaced"] = True
        for q, slot in TIME_Q.items():
            for qs in (st.get("qualifiers") or {}).get(q) or ():
                tv = (qs.get("datavalue") or {}).get("value") or {}
                pt = parse_time(tv.get("time"), tv.get("precision"))
                if pt and slot not in val:
                    val[slot] = pt
        out.append(val)
    return out


def _sparql(query: str, tries: int = 3, timeout: float = 70) -> list[dict] | None:
    from cubbyllm.sources.wikidata import UA
    for k in range(tries):
        try:
            req = urllib.request.Request(SPARQL, data=urllib.parse.urlencode({"query": query}).encode(),
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.2)
            return rows
        except urllib.error.HTTPError as e:
            time.sleep((int(e.headers.get("Retry-After") or 30) + 1) if e.code == 429 else 5 * (k + 1))
        except Exception:                                   # noqa: BLE001 -- recorded as unavailable, never as none
            time.sleep(5 * (k + 1))
    return None


def incoming(pid: str, obj: str) -> list[dict] | None:
    """The statements `?s pid obj`, as values whose "v" is the subject; None if the service could not say."""
    opt = " ".join(f"OPTIONAL {{ ?st pqv:{q} ?{s}v . ?{s}v wikibase:timeValue ?{s} ; wikibase:timePrecision ?{s}p . }}"
                   for q, s in TIME_Q.items())
    rows = _sparql(f"SELECT ?st ?s ?rank ?start ?startp ?end ?endp ?point ?pointp ?rb WHERE {{ ?s p:{pid} ?st . "
                   f"?st ps:{pid} wd:{obj} ; wikibase:rank ?rank . {opt} OPTIONAL {{ ?st pq:P1366 ?rb . }} }} LIMIT 5000")
    if rows is None:
        return None
    by: dict[str, dict] = {}
    for b in rows:
        st = b["st"]["value"]
        if st in by:
            continue
        val = {"v": b["s"]["value"].rsplit("/", 1)[1], "rank": RANKS.get(b["rank"]["value"].rsplit("#", 1)[1], "normal")}
        if "rb" in b:
            val["replaced"] = True
        for s in TIME_Q.values():
            if s in b:
                t = b[s]["value"]
                val[s] = parse_time(t if t[:1] in "+-" else "+" + t, int(b[s + "p"]["value"]) if s + "p" in b else None)
        by[st] = val
    return list(by.values())


class Statements:
    """One fetch per (item, property, direction), ever; dated."""

    def __init__(self, src, path: str | None = None) -> None:
        from retrieval_sets.common import data_root
        self.src, self.path = src, path or os.path.join(data_root(), "r4_statements.jsonl")
        self.c: dict[str, dict] = {}
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    self.c[d["key"]] = d

    def _put(self, key: str, values: list | None, **extra) -> dict:
        d = {"key": key, "values": values, "fetched": _dt.date.today().isoformat(), **extra}
        self.c[key] = d
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return d

    def out(self, qid: str, pid: str) -> dict:
        key = f"out|{qid}|{pid}"
        if self.c.get(key, {}).get("r") != 2:
            from retrieval_sets.common import entities
            claims = (entities(self.src, [qid], props="claims", langs="en").get(qid) or {}).get("claims") or {}
            self._put(key, values_of(claims, pid), r=2)
        return self.c[key]

    def inc(self, qid: str, pid: str) -> dict:
        key = f"in|{qid}|{pid}"
        if self.c.get(key, {}).get("r") != 2:
            self._put(key, incoming(pid, qid), r=2)
        return self.c[key]

    def single(self, pid: str) -> bool:
        key = f"single|{pid}"
        if key not in self.c:
            from retrieval_sets.common import entities
            claims = (entities(self.src, [pid], props="claims", langs="en").get(pid) or {}).get("claims") or {}
            self._put(key, None, single=any(v["v"] in SINGLE for v in values_of(claims, "P2302")))
        return bool(self.c[key].get("single"))

    def ends(self, qids) -> dict:
        """{item: the earliest year its own end is stated (P570, P576, P582), or None}, fetched 50 to a call."""
        from retrieval_sets.common import entities
        todo = [q for q in dict.fromkeys(qids) if f"end|{q}" not in self.c]
        got = entities(self.src, todo, props="claims", langs="en") if todo else {}
        unread = {q for q in todo if q not in got}       # not answered: never read as "no end" (rule 1 cannot place)
        for q in todo:
            if q in unread:
                continue
            claims = (got.get(q) or {}).get("claims") or {}
            ys = [v["v"] for p in ("P570", "P576", "P582") for v in values_of(claims, p)
                  if v.get("rank") != "deprecated" and isinstance(v.get("v"), str)]
            yrs = [t["y"] for t in (parse_time(x) for x in ys) if t]
            self._put(f"end|{q}", None, end=min(yrs) if yrs else None)
        return {q: "unread" if q in unread else self.c[f"end|{q}"].get("end") for q in qids}


__all__ = ["Statements", "incoming", "parse_time", "values_of"]
