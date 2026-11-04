"""r2_b1q_prep -- order 0046, B1q's inputs and its arms in R2's harness (``r2_choosing``), from Cowork's check of
2026-11-04:

  cues           rule (a)'s cue runs per row (``r2_b1q.cue_runs``), looked up as names in R1b's segment (exact)
  relation given Mintaka's rows: R3's gold rule (the properties linking a question item to an answer item in today's
                 Wikidata), used as the row's relation when exactly one property links them. WebQSP_EL's rows carry
                 no Wikidata property (WebQSP's SPARQL is Freebase's) and stay without one.
  gold_bears     ShadowLink: whether the gold item bears the mention in R1b's names (the line is read where it does)
  shared_today   fame, bare rows: whether two or more of the name's items still bear it today (rule (c))
The arms: ``b1q`` on every row (on the qualifier alone where a row has no relation); ``b1_rel`` and ``b1q_rel`` on
Mintaka's rows with a relation given.

Wired: STANDALONE (R2's bench).
"""
from __future__ import annotations

import json

__wiring__ = "STANDALONE"

_ITEMS: dict[str, list[str]] = {}


def prepare(rows: list[dict], profiles: dict, kpath: str) -> dict:
    """Every row's cues, relation given, gold_bears and shared_today; the branch segment's extra facts appended."""
    from retrieval_bench import r2_arms as A
    from retrieval_bench import r2_b1q as Q
    from retrieval_bench.r3_gold import ClaimIds, links
    from retrieval_sets.common import source
    from retrieval_sets.lexicon import check
    from retrieval_sets.names_segment import exact_items
    lx, src = check(), source()
    mk = [r for r in rows if r["set"] == "mintaka"]
    claims = ClaimIds(src).get(sorted({q for r in mk for q in r.get("row_items", []) + r.get("answer_qids", [])})) if mk else {}
    for r in rows:
        r["named"] = A.bearing(r["mention"], r["cands"], profiles)
        if r["set"] == "mintaka":
            ls = links(r.get("row_items") or [], r.get("answer_qids") or [], claims)
            r["pids_given"] = ls if len(ls) == 1 else []
        names = [r["mention"]] + [v for q in r.get("items") or () for v in ((profiles.get(q) or {}).get("labels") or {}).values() if v]
        r["cues"] = Q.cue_runs(r["question"], names, r["pids"] or r.get("pids_given") or [], lx)
    names = sorted({c for r in rows for c in r["cues"]} | {r["mention"] for r in rows if r["set"] in ("shadowlink", "fame")})
    found = exact_items(names)
    _ITEMS.clear()
    _ITEMS.update({k: v for k, v in found.items() if v})
    for r in rows:
        if r["set"] == "shadowlink":
            r["gold_bears"] = r["gold"] in set(found.get(r["mention"]) or ())
        if r["set"] == "fame" and r["kind"] == "bare":
            r["shared_today"] = len(set(r.get("items") or ()) & set(found.get(r["mention"]) or ())) >= 2
    small = sorted({v for c in {c for r in rows for c in r["cues"]} for v in (_ITEMS.get(c) or ()) if len(_ITEMS[c]) <= Q.IN_MAX})
    claims_in = ClaimIds(src).get(small) if small else {}
    n_q = Q.write_segment(kpath, rows, profiles, _ITEMS, claims_in)
    n_rel = 0
    with open(kpath, "a", encoding="utf-8") as f:                 # the given relation's values, for B1 on those rows
        for r in rows:
            for pid in r.get("pids_given") or ():
                for q in r["named"]:
                    for v in dict.fromkeys((profiles.get(q, {}).get("values") or {}).get(pid) or ()):
                        f.write(json.dumps({"key": f"{q} {pid}", "text": v, "source": f"wikidata:{q}"}, ensure_ascii=False) + "\n")
                        n_rel += 1
    return {"qualifier_facts": n_q, "relation_given_facts": n_rel, "cue_names": len(_ITEMS),
            "relation_given_rows": sum(1 for r in rows if r.get("pids_given"))}


def b_arms(session, kpath: str, r: dict, profiles: dict) -> dict:
    from retrieval_bench import r2_vm as V
    from retrieval_bench.r2_b1q import b1q
    named, out = r.get("named") or [], {}
    none = {"outcome": "refuse", "why": "no candidate bears the name"}
    if r["pids"]:
        out["b1q"] = b1q(session, kpath, named, r["pids"][0], r["cues"], _ITEMS) if named else none
        return out
    out["b1q"] = b1q(session, kpath, named, None, r["cues"], _ITEMS) if named else none
    if r.get("pids_given"):
        pid = r["pids_given"][0]
        out["b1_rel"] = V.b1(session, kpath, named, pid) if named else none
        out["b1q_rel"] = b1q(session, kpath, named, pid, r["cues"], _ITEMS) if named else none
    return out


__all__ = ["b_arms", "prepare"]


def b1_qfhrr(rows: list[dict], profiles: dict, taus: list[float]) -> dict:
    """Cowork's comparison arm, your (b): B1 first, then qFHRR among the branches that verify (on grilly2), picking past
    a rank-normalized margin; otherwise B1's outcome stands. The words are rule (a)'s cues. Arms ``b1_qfhrr@<tau>``."""
    from cubbyllm.worlds.fact_pool import V4_TABLE
    from retrieval_bench import r2_arms as A
    from retrieval_bench import r2_qfhrr as Q
    from retrieval_bench.r2_choosing import judge
    todo, docs = [], {}
    for r in rows:
        items = r.get("b1_items") or []
        words = sorted({w for c in r.get("cues") or () for w in c.split()})
        if len(items) >= 2 and words:
            keys = [f"{q}|{r['lang']}" for q in items]
            for q, k in zip(items, keys):
                docs.setdefault(k, " ".join(A.doc(profiles[q], r["lang"])))
            todo.append({"id": r["id"], "cues": " ".join(words), "items": keys})
    rep = Q.score({"table": str(V4_TABLE), "docs": docs, "rows": todo}) if todo else {"scores": {}}
    scored = {t["id"]: t for t in todo}
    for r in rows:
        if "b1" not in r.get("arms", {}):
            continue
        for t in taus:
            o = dict(r["arms"]["b1"])
            if r["id"] in scored:
                d = Q.decide([k.split("|")[0] for k in scored[r["id"]]["items"]], rep["scores"][r["id"]], t)
                if d["outcome"] == "pick":
                    picked = {"outcome": "pick", "item": d["item"]}
                    o = {**picked, "judge": judge(r, picked), "by": "qFHRR among the branches that verify"}
            r["arms"][f"b1_qfhrr@{t}"] = o
    return {k: v for k, v in rep.items() if k != "scores"}


def pick_margin(summary: dict, taus: list[float]) -> float | None:
    """The comparison arm's margin, set on seed 0: the smallest with 0 bare-name single picks and 0 qualified wrong."""
    for t in sorted(taus):
        a = f"b1_qfhrr@{t}"
        if (summary.get("fame/bare", {}).get(a) or {}).get("single", 0) == 0 and \
                (summary.get("fame/qualified", {}).get(a) or {}).get("wrong", 0) == 0:
            return t
    return None
