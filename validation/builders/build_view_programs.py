"""Order 0067, the second reader's step 2 (Nick, 2026-11-13: "go ahead with step 2"): programs over the reply's
views, built in reverse and run before they are kept.

For each program: a pool of replies built for it (`view_replies.py`); a spec drawn from the pool's views
(`view_programs.sample`: its values and thresholds come from the replies, held-out test kinds never drawn); compiled to
CubeLang (`view_compile.py`); then run in the VM on K replies of the pool, half that the spec passes and half that it
fails (by the Python evaluator), each reply's views short enough for the VM (`reply_views.MAX_ROWS`). (Until cubelang
P6-9, `676df1d`, replies whose program carried two strings sharing the VM's 2-byte name key were left out.) Kept only when
the VM's verdict is the evaluator's on every reply and some replies pass and some fail. A disagreement stops the build:
it is a compiler or VM defect, never a row.

  python validation/builders/build_view_programs.py [--n 2000] [--seed 0] [--k 8] [--out $CUBBY_DATASETS/view_programs_0067]
Writes rows.jsonl ({"id", "spec", "body", "views", "replies": [{"text", "held"}]}) and manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import view_compile as VC  # noqa: E402
import view_programs as VP  # noqa: E402
import view_replies as VR  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.bridges import cubelang_client as cc  # noqa: E402
from cubbyllm.bridges import vm_identity  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402

__wiring__ = "STANDALONE"


def shape(spec: dict) -> str:
    """The spec's form without its values, for the manifest ("count word[first_lower] | and | n sentence")."""
    def f(s):
        if "q" in s:
            cols = ",".join(c["col"] for c in s.get("where", [])) or s.get("col", "")
            return f"{s['q']} {s['view']}" + (f"[{cols}]" if cols else "")
        if "not" in s:
            return f"not ({f(s['not'])})"
        if "if" in s:
            return f"if ({f(s['if'])}) then ({f(s['then'])})"
        key = next(k for k in ("and", "or", "unless") if k in s)
        return f" {key} ".join(f"({f(x)})" for x in s[key])
    return f(spec)


def fits(text: str, views: list[str]) -> bool:
    vs = RV.views(text)
    return all(len(vs[v]) <= RV.MAX_ROWS for v in views)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or Path(local_paths.value("CUBBY_DATASETS") or ".") / "view_programs_0067")
    rng, vm = random.Random(a.seed), cc.CubelangSession()
    rows, seen, why = [], set(), Counter()
    while len(rows) < a.n:
        why["drawn"] += 1
        replies = VR.pool(rng, a.pool)
        views = [RV.views(t) for t in replies]
        spec = VP.sample(rng, views)
        if spec is None:
            why["no spec drawn from the pool"] += 1
            continue
        assert VP.held_out(spec) is None
        used = VC.views_of(spec)
        cand = [(t, VP.evaluate(spec, v)) for t, v in zip(replies, views) if fits(t, used)]   # the VM can read the reply
        yes, no = [c for c in cand if c[1]], [c for c in cand if not c[1]]
        if len(yes) < 2 or len(no) < 2:
            why["too few short replies on one side"] += 1
            continue
        chosen = rng.sample(yes, min(len(yes), a.k // 2)) + rng.sample(no, min(len(no), a.k - min(len(yes), a.k // 2)))
        body = VC.body(spec)
        if body in seen:
            why["the same program again"] += 1
            continue
        verdicts = []
        for text, want in chosen:
            res = vm.run(VC.program(spec, text), fn="solve").get("result")
            if res is None:
                raise SystemExit(f"the VM returned nothing for a reply that fits: {json.dumps(spec)}\n{text[:300]}")
            got = str(res).strip() in ("1", "1.0")
            if got != want:                              # kept whole for the write-up (and P6-1's cross-check)
                (out / "_disagreement.cube").parent.mkdir(parents=True, exist_ok=True)
                (out / "_disagreement.cube").write_text(VC.program(spec, text), encoding="utf-8")
                (out / "_disagreement.json").write_text(json.dumps({"spec": spec, "reply": text, "evaluator": want,
                                                                     "vm": got}, ensure_ascii=False, indent=1), encoding="utf-8")
                raise SystemExit(f"the VM disagrees with the evaluator ({got} != {want}): {json.dumps(spec)}; "
                                 f"kept in {out}/_disagreement.*")
            verdicts.append(got)
        if all(verdicts) or not any(verdicts):
            why["one-sided in the VM"] += 1
            continue
        seen.add(body)
        rows.append({"id": f"p{len(rows):06d}", "spec": spec, "shape": shape(spec), "body": body, "views": used,
                     "replies": [{"text": t, "held": h} for (t, _), h in zip(chosen, verdicts)]})
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} kept of {why['drawn']} drawn", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    atoms = Counter(f"{x['q']} {x['view']}" for r in rows for x in VP.atoms(r["spec"]))
    joins = Counter(next((k for k in ("and", "or", "not", "if", "unless") if k in r["spec"]), "atom") for r in rows)
    man = {"order": "0067", "what": "the second reader's step 2: programs over the reply's views, run before kept",
           "seed": a.seed, "programs": len(rows), "replies_per_program": a.k, "pool": a.pool, "drop_reasons": dict(why),
           "vm_runs_agreeing_with_the_evaluator": sum(len(r["replies"]) for r in rows), "vm_disagreements": 0,
           "atoms": dict(atoms.most_common()), "joins": dict(joins), "shapes": len({r["shape"] for r in rows}),
           "held_out_forms": "never drawn (view_programs.held_out)", "max_rows_per_view": RV.MAX_ROWS,
           "vm": vm_identity.identity(), "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("atoms",)}, indent=1))


if __name__ == "__main__":
    main()
