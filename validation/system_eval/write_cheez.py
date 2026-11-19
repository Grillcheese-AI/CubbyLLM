"""Order 0076, acceptance lines 1 and 2: arm C's run (with ``CB_CERT_FACTS=1``) written as CheezML records, one per
question, each checked.

- **Line 1:** every question gets a certificate or a refusal, except those the writer cannot state (counted by
  reason: v0.2's rule); each written record conforms to CheezML's shapes at the pin (`vendor/cheezml`).
- **Line 2:** each record is read back field by field against the run's record: the answer value, the program's
  hash, each fact version's item, property, statement id, rank and revision, the snapshot, the as-of time, the
  checks and their outcomes, the time. Any difference is the kill: the writer stops until the cause is fixed.

The records are written off-repo (``--records``, JSON lines: real records are never published); the report holds
counts only. The run's facts: the store snapshot from its meta, the as-of time from the store's manifest (the dump's
date: the store's own present, for a question that names no year).

  python validation/system_eval/write_cheez.py --run <run dir> --records <out.jsonl> --out <report.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT]

from cubbyllm.records import cheez_writer as W  # noqa: E402
from cubbyllm.records.cheez_facts import RANK  # noqa: E402
from cubbyllm.records.refusals import CHECK, CODE, head  # noqa: E402

__wiring__ = "STANDALONE"

HOST, VM = "urn:uuid:4c0b0000-0000-4000-8000-00000000c0de", "urn:uuid:4c0b0000-0000-4000-8000-0000000000f1"


def by_type(g: dict) -> dict:
    out: dict = {}
    for n in g["@graph"]:
        for t in n["@type"] if isinstance(n["@type"], list) else [n["@type"]]:
            out.setdefault(t, []).append(n)
    return out


def differences(g: dict, rec: dict, run: dict) -> list:
    """Each field of the record against the run's record; [] when every field is the run's."""
    t, cert, bad = by_type(g), rec["cert"], []
    checks = [c for c in cert.get("checks") or [] if c["check"] in CHECK]
    runs = t.get("cheez:CheckRun", [])
    if sorted((n["dcterms:type"], n["cheez:outcome"]) for n in runs) != sorted(
            (CHECK[c["check"]], "cheez:Satisfied" if c["satisfied"] else "cheez:Violated") for c in checks):
        bad.append("checks")
    if rec.get("spoken"):
        ans, = t["cubby:Answer"]
        prog, = t["cube:Program"]
        pr, = t["cube:ProgramRun"]
        snap = [n for n in t["cheez:Artifact"] if n.get("dcterms:type") == "cheez:StoreSnapshot"]
        bad += [k for k, ok in (("answer", ans["prov:value"] == rec["answer"]),
                                ("program", [prog["cheez:hash"]] == sorted(set(cert["programs"]))),
                                ("snapshot", [s["cheez:hash"] for s in snap] == [run["snapshot"]]),
                                ("as of", pr["cheez:asOf"] == run["as_of"]),
                                ("time", t["cheez:Certificate"][0]["prov:generatedAtTime"] == cert["at"])) if not ok]
        src = {n["@id"]: n for n in t.get("prov:Entity", []) if "schema:version" in n}
        got = sorted((f["rdf:subject"], f["rdf:predicate"], f.get("schema:identifier"), f["wikibase:rank"],
                      src[f["prov:hadPrimarySource"]]["schema:version"]) for f in t["cheez:FactVersion"])
        want = sorted((f"wd:{s['item']}", f"wdt:{s['pid']}", s["guid"], RANK[s["rank"]], str(s["revision"]))
                      for f in cert["facts"] for w in f["worlds"] if w["world"] == run["store"] and w["items"]
                      for s in w["statements"])
        if got != want:
            bad.append("fact versions")
    else:
        ref, = t["cubby:Refusal"]
        if ref["cubby:refusalReason"] != CODE.get(head(rec.get("reason"))) or ref["prov:generatedAtTime"] != cert["at"]:
            bad.append("refusal")
    return bad


def main() -> None:
    import rdflib
    import pyshacl
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from cubbyllm.local_paths import value
    dump = json.load(open(os.path.join(value("CUBBY_WD_STORE"), "manifest.json"), encoding="utf-8"))["dump_date"]
    src = os.path.join(ROOT, "vendor", "cheezml", "src")
    ctx = json.load(open(os.path.join(src, "context.jsonld"), encoding="utf-8"))["@context"]
    shapes = rdflib.Graph().parse(os.path.join(src, "shapes.ttl"), format="turtle")
    t, why, kinds, failed = Counter(), Counter(), Counter(), []
    with open(a.records, "w", encoding="utf-8") as out:
        for f in sorted(glob.glob(os.path.join(a.run, "c_gather_*.jsonl"))):
            meta = json.load(open(f[:-6] + ".meta.json", encoding="utf-8"))
            run = {"run": os.path.basename(a.run.rstrip("/\\")), "snapshot": meta["wd_store"], "as_of": dump,
                   "store": "wdstore", "host": HOST, "vm": VM}
            bench = meta["bench"]
            for ln in open(f, encoding="utf-8"):
                rec = json.loads(ln)
                side = "spoken" if rec["spoken"] else "refused"
                t[f"{bench} {side}: questions"] += 1
                g, w = W.write(rec, run)
                if g is None:
                    why[f"{side}: {w}"] += 1
                    t[f"{bench} {side}: no record"] += 1
                    continue
                data = rdflib.Graph().parse(data=json.dumps({"@context": ctx, **g}), format="json-ld")
                ok = pyshacl.validate(data, shacl_graph=shapes, inference="none", advanced=True)[0]
                diff = differences(g, rec, run)
                t[f"{bench} {side}: records"] += 1
                t[f"{bench} {side}: records that conform"] += ok
                t[f"{bench} {side}: records that differ from the run (kill)"] += bool(diff)
                kinds[next(iter(n["cubby:refusalReason"] for n in g["@graph"] if "cubby:refusalReason" in n), "certificate")] += 1
                if diff or not ok:
                    failed.append({"id": rec["id"], "conforms": ok, "differs in": diff})
                out.write(json.dumps({"id": rec["id"], "record": {"@context": "https://w3id.org/cheezml/context.jsonld", **g}},
                                     ensure_ascii=False) + "\n")
    report = {"order": "0076", "what": "acceptance 1 and 2: arm C's run written as CheezML records, checked",
              "run": os.path.basename(a.run.rstrip("/\\")), "cheezml": "vendor/cheezml (the pin)",
              "counts": dict(sorted(t.items())), "records by kind": dict(kinds.most_common()),
              "no record, by reason": dict(why.most_common()), "failed (ids only)": failed[:50]}
    open(a.out, "w", encoding="utf-8").write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "no record, by reason")}, indent=1))


if __name__ == "__main__":
    main()
