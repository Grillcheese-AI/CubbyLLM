"""Order 0073, step 2: what the wording says, pinned on every stored part, in English and in French, for each way a
part is said (`view_said.say`: the imperative, the condition of an "if" or "unless", the goal-first wording).

For each sentence that has a description (arm W's form; arm M's is its fold):
- **a ``not`` exactly where a verb is negated:** the form carries as many ``not`` as the sentence has negation words
  ("not"; "pas"), quoted values aside;
- **the truth is the stored part's:** arm M's form has the stored part's truth on every reply of its row;
- **one sentence, one target:** over the whole set, a sentence never has two forms (in either arm);
- **the VM agrees:** arm M's form, compiled and run on the row's replies, gives the stored part's truth (the
  imperative in the row's own language, wherever the form is not the stored part itself).

Counts only; a row that fails a line is listed by id, and the build does not go on until every list is empty.

  python validation/exp_view_said_pin.py [--out <json>] [--limit N] [--no-vm]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE / "builders")]
import view_said as S  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_compile as VC  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

SETS = (("long rows", "long_rows_0067/rows.jsonl"), ("pair rows", "long_rows_0067/pair_rows.jsonl"))


def key(form: dict) -> str:
    return json.dumps(form, sort_keys=True, ensure_ascii=False)


def check(path: Path, vm, limit: int, targets: dict) -> dict:
    raw = path.read_bytes()
    t = Counter()
    bad = {"a not without its word": [], "the said form changes the truth": [], "the VM differs": []}
    for n, ln in enumerate(raw.decode("utf-8").splitlines()):
        if limit and n >= limit:
            break
        row = json.loads(ln)
        views = [RV.views(rep["text"]) for rep in row["replies"]]
        t["rows"] += 1
        for part in row["parts"]:
            t["parts"] += 1
            truth = [VS.evaluate(part, v) for v in views]
            for lang in ("en", "fr"):
                for how in ("imp", "inverted") + (("cond",) if "q" in part else ()):
                    text, w, m = S.say(part, lang, random.Random(0), how)
                    if text is None:
                        continue
                    t[f"sentences, {how}, {lang}"] += 1
                    if w is None:
                        t[f"sentences with no description, {how}, {lang}"] += 1
                        continue
                    t["forms"] += 1
                    t["forms with a not"] += int(S.marks(w) > 0)
                    t[f"forms said as {w['q']}" + (" with absent: pass" if w.get("absent") else "")] += 1
                    if S.marks(w) != S.negations(text, lang) or S.marks(m) or VS.fold(m) != m:
                        bad["a not without its word"].append(row["id"])
                    if [VS.evaluate(m, v) for v in views] != truth or [VS.evaluate(w, v) for v in views] != truth:
                        bad["the said form changes the truth"].append(row["id"])
                    targets[(lang, text)].add((key(w), key(m)))
                    if vm is None or how != "imp" or lang != row["lang"] or m == part:
                        continue
                    for rep, want in zip(row["replies"], truth):
                        try:
                            src = VC.program(m, rep["text"])
                        except RV.TooLong:
                            t["replies too long for the VM"] += 1
                            continue
                        t["VM runs"] += 1
                        if (str(vm.run(src, fn="solve").get("result")).strip() in ("1", "1.0")) != want:
                            bad["the VM differs"].append(row["id"])
    return {"counts": dict(sorted(t.items())), "disagreements": {k: sorted(set(v)) for k, v in bad.items()},
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
    targets: dict = defaultdict(set)
    report = {"order": "0073", "what": "step 2: each wording's said form (arm W) and its fold (arm M) on every "
              "stored part, English and French", "vm": identity, "sets": {}}
    for name, rel in SETS:
        r = check(root / rel, vm, a.limit, targets)
        report["sets"][name] = dict(r, file=rel)
        c = Counter(r["counts"])
        print(f"{name}: {c['rows']} rows, {c['parts']} parts, forms {c['forms']} (with a not {c['forms with a not']}), "
              f"VM runs {c['VM runs']}; disagreements " + ", ".join(f"{k}: {len(v)}" for k, v in r["disagreements"].items()),
              flush=True)
    two = [k for k, v in targets.items() if len(v) > 1]
    report["sentences"] = {"distinct": len(targets), "with two targets": len(two),
                           "distinct, en": sum(1 for k in targets if k[0] == "en"),
                           "distinct, fr": sum(1 for k in targets if k[0] == "fr")}
    print(f"distinct sentences {len(targets)}, with two targets {len(two)}", flush=True)
    for lang, text in two[:5]:                             # wording templates, not row text: shown to be fixed
        print("  two targets:", lang, text, sorted(targets[(lang, text)])[:2], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
