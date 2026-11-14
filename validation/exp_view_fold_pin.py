"""Order 0073, step 1 (second half): `view_spec.fold` and ``absent: pass`` pinned on every stored part and reply.

For each part of each long row and pair row (an atom, or ``{"not": atom}``) and each reply of its row:
- **the fold keeps the truth:** `evaluate(fold(part))` is `evaluate(part)`;
- **one folded spelling:** the part's other said forms (the ``not`` moved onto the atom, onto each condition with the
  comparison turned, around the part twice, onto the other quantifier) have the part's truth and fold to the same
  atom;
- **the VM agrees:** where folding changed the part, the folded atom compiled (`view_compile.program`) and run in the
  VM gives the evaluator's verdict. This is where ``absent: pass`` and the folded quantifiers are run.

``not all`` of two conditions has no folded form (an or-filter): counted, and only its said forms' truth is checked.
A disagreement is counted and its row id listed; the build does not go on until every list is empty. No row text is
written.

  python validation/exp_view_fold_pin.py [--out <json>] [--limit N] [--no-vm]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

SETS = (("long rows", "long_rows_0067/rows.jsonl"), ("pair rows", "long_rows_0067/pair_rows.jsonl"))


def said_forms(part: dict) -> list[dict]:
    """Other ways to write `part` with the same truth: where its ``not`` sits, and how a condition is spelled."""
    neg = "q" not in part
    a = part["not"] if neg else part
    out = [{"not": {"not": part}}]
    if "where" in a:                                       # each condition as "not" its turned comparison
        w = dict(a, where=[dict(c, op=VS.INV[c["op"]], **{"not": True}) for c in a["where"]])
        out.append({"not": w} if neg else w)
    if a["q"] == "at":                                     # the comparison turned and negated; around it, as said
        w = dict(a, op=VS.INV[a["op"]], **{"not": True})
        out.append({"not": w} if neg else w)
        if neg:
            out.append(dict(a, absent="pass", **{"not": True}))
    else:
        if neg:
            out.append(dict(a, **{"not": True}))           # the atom's own quantifier or comparison negated
        if a["q"] in ("any", "none"):                      # the other quantifier, negated
            other = dict(a, q="none" if a["q"] == "any" else "any")
            out.append(other if neg else {"not": other})
    return out


def check(path: Path, vm, limit: int) -> dict:
    raw = path.read_bytes()
    t, bad = Counter(), {"the fold changes the truth": [], "a said form differs": [], "the VM differs": []}
    for n, ln in enumerate(raw.decode("utf-8").splitlines()):
        if limit and n >= limit:
            break
        row = json.loads(ln)
        views = [RV.views(rep["text"]) for rep in row["replies"]]
        t["rows"] += 1
        for part in row["parts"]:
            t["parts"] += 1
            truth = [VS.evaluate(part, v) for v in views]
            try:
                folded = VS.fold(part)
            except ValueError:
                folded = None
                t["parts with no folded form (not all of two conditions)"] += 1
            for form in said_forms(part):
                t["said forms"] += 1
                same = [VS.evaluate(form, v) for v in views] == truth
                if folded is not None:
                    same = same and VS.fold(form) == folded
                if not same:
                    bad["a said form differs"].append(row["id"])
            if folded is None:
                continue
            t["parts folded"] += 1
            t["replies"] += len(views)
            if [VS.evaluate(folded, v) for v in views] != truth:
                bad["the fold changes the truth"].append(row["id"])
            if folded == part:
                continue
            t["parts the fold changed"] += 1
            t["parts with absent: pass"] += int(folded.get("absent") == "pass")
            if vm is None:
                continue
            for rep, want in zip(row["replies"], truth):
                try:
                    src = VC.program(folded, rep["text"])
                except RV.TooLong:
                    t["replies too long for the VM"] += 1
                    continue
                res = vm.run(src, fn="solve").get("result")
                t["VM runs"] += 1
                if (str(res).strip() in ("1", "1.0")) != want:
                    bad["the VM differs"].append(row["id"])
    return {"counts": dict(t), "disagreements": {k: sorted(set(v)) for k, v in bad.items()},
            "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-vm", action="store_true")
    a = ap.parse_args()
    vm, identity = None, None
    if not a.no_vm:
        from cubbyllm.bridges import cubelang_client as cc
        from cubbyllm.bridges import vm_identity
        vm, identity = cc.CubelangSession(), vm_identity.identity()
    root = Path(local_paths.value("CUBBY_DATASETS") or ".")
    report = {"order": "0073", "what": "step 1: fold and absent: pass on every stored part and reply (the evaluator, "
              "the said forms, the VM)", "vm": identity, "sets": {}}
    for name, rel in SETS:
        r = check(root / rel, vm, a.limit)
        report["sets"][name] = dict(r, file=rel)
        c = Counter(r["counts"])
        print(f"{name}: {c['rows']} rows, {c['parts']} parts, folded {c['parts folded']} (changed {c['parts the fold changed']}, "
              f"absent: pass {c['parts with absent: pass']}), no folded form "
              f"{c['parts with no folded form (not all of two conditions)']}; said forms {c['said forms']}; VM runs "
              f"{c['VM runs']}; disagreements " + ", ".join(f"{k}: {len(v)}" for k, v in r["disagreements"].items()),
              flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
