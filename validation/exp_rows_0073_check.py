"""Order 0073, step 3: `long_rows_0073` read back from its files and checked, before anything is trained on it.

The builder ran every reply in the VM when it kept a row (`build_rows_0073.py`). This reads the files as written and
counts, with the evaluator only, what must be 0 for the set to be used:
- an instruction of 0067's rows, an instruction or id twice, a train or dev instruction sharing 5 words with a
  benchmark, a part of a kind kept for the test, a held-out row with a train row's folded parts and groups;
- segments that do not tile the instruction, parts not numbered in the text's order, a join tag that is not on a
  tied pair's check beside its partner;
- a said form that is not the wording's, an arm M form that is not the fold of arm W's or carries a ``not``, an arm
  W form whose count of ``not`` is not the count of negation words in the check's text;
- a program that is not the join of the folded forms, a reply whose stored verdict is not the evaluator's on the
  folded forms or on the parts as drawn, a train row with replies added;
- twins that differ in more than the flipped part's sentence;
- a file or a split whose SHA-256 is not the manifest's.

  python validation/exp_rows_0073_check.py [--set $CUBBY_DATASETS/long_rows_0073] [--out validation/logs/<name>.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "validation" / "builders")]
import bench_overlap as BO  # noqa: E402
import build_rows_0073 as B  # noqa: E402
import rows_0073_stats as ST  # noqa: E402
import view_draw as D  # noqa: E402
import view_said as S  # noqa: E402

from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import reply_views as RV  # noqa: E402
from cubbyllm.speech import view_join as VJ  # noqa: E402
from cubbyllm.speech import view_spec as VS  # noqa: E402

__wiring__ = "STANDALONE"

COUNTED = ("rows", "checks", "replies", "pairs")
BY_SPLIT = "the lines of rows.jsonl by split"


def _read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(path, encoding="utf-8")]


def _row(r: dict, out: Counter) -> None:
    text, at, tiled = r["instruction"], 0, True
    for s in r["segments"]:
        tiled &= s["start"] >= at and not text[at:s["start"]].strip() and bool(text[s["start"]:s["end"]].strip())
        at = s["end"]
    out["segments that do not tile the instruction"] += not (tiled and not text[at:].strip())
    checks = [s for s in r["segments"] if s["kind"] == "check"]
    out["parts not numbered in the text's order"] += [s["part"] for s in checks] != list(range(r["n_parts"]))
    m, groups = [s["m"] for s in r["said"]], [tuple(g) for g in r["groups"]]
    for seg, part, said in zip(checks, r["parts"], r["said"]):
        out["checks"] += 1
        out["a said form that is not the wording's"] += seg["how"] != said["how"] or said["w"] != S.said(part, r["lang"], seg["how"])
        out["an arm M form that is not the fold of arm W's"] += said["m"] != VS.fold(said["w"]) or S.marks(said["m"]) != 0
        out["an arm W form whose nots are not the words'"] += S.marks(said["w"]) != S.negations(
            text[seg["start"]:seg["end"]], r["lang"])
        out["a part of a kind kept for the test"] += D.held(part)
    tags = {s["part"]: s["join"] for s in checks if s.get("join")}
    tied = [g for g in groups if g[0] != "one"]
    wrong = len(tags) != len(tied)
    for kind, a, b in tied:
        i = next((x for x in (a, b) if x in tags), None)
        wrong |= i is None or tags[i][0] != kind or i + (1 if tags[i][1] == "next" else -1) != (b if i == a else a)
    out["a join tag not on a tied pair's check beside its partner"] += wrong
    out["a program that is not the join of the folded forms"] += r["join"] != VJ.join(m, groups)
    gold, drawn = VJ.spec(m, groups), VJ.spec(r["parts"], groups)
    for x in r["replies"]:
        views = RV.views(x["text"])
        out["replies"] += 1
        out["a verdict that is not the evaluator's on the folded forms"] += VS.evaluate(gold, views) != x["held"]
        out["a verdict that is not the evaluator's on the parts as drawn"] += VS.evaluate(drawn, views) != x["held"]
    out["a train row with replies added"] += r["split"] == "train" and any(x.get("extra") for x in r["replies"])


def _twins(a: dict, b: dict) -> bool:
    other = [i for i, (p, q) in enumerate(zip(a["parts"], b["parts"])) if p != q]
    said = [[(s["kind"], s.get("part"), r["instruction"][s["start"]:s["end"]]) for s in r["segments"]] for r in (a, b)]
    return (a["pair"] == b["pair"] and a["groups"] == b["groups"] and len(other) == 1
            and ["one", other[0]] in a["groups"] and B.flip(a["parts"][other[0]]) == b["parts"][other[0]]
            and len(said[0]) == len(said[1]) and [x == y for x, y in zip(*said)].count(False) == 1)


def check(ds: Path, old: Path | None = None) -> dict:
    """{what: how many}; everything but `COUNTED` must be 0."""
    rows, pairs = _read(ds / "rows.jsonl"), _read(ds / "pair_rows.jsonl")
    man, every = json.loads((ds / "manifest.json").read_text(encoding="utf-8")), rows + pairs
    out = Counter({"rows": len(every), "pairs": len(pairs) // 2})
    for r in every:
        _row(r, out)
    texts = {json.loads(ln)["instruction"] for name in ("rows.jsonl", "pair_rows.jsonl")
             if old and (old / name).exists() for ln in open(old / name, encoding="utf-8")}
    bench = BO.benchmark_grams(ifbench=True)
    train = {B.key(r) for r in every if r["split"] == "train"}
    out["an instruction of 0067's rows"] = sum(r["instruction"] in texts for r in every)
    out["an instruction twice"] = len(every) - len({r["instruction"] for r in every})
    out["an id twice"] = len(every) - len({r["id"] for r in every})
    out["a train or dev instruction sharing 5 words with a benchmark"] = sum(
        bool(BO.overlaps(r["instruction"], bench)) for r in every if r["split"] in ("train", "dev"))
    out["a held-out row with a train row's folded parts and groups"] = sum(
        B.key(r) in train for r in rows if r["split"] != "train")
    out["twins that differ in more than the flipped part's sentence"] = sum(
        not _twins(a, b) for a, b in zip(pairs[::2], pairs[1::2]))
    out["a SHA-256 that is not the manifest's"] = sum(
        man["sha256"][name] != hashlib.sha256((ds / name).read_bytes()).hexdigest() for name in ("rows.jsonl", "pair_rows.jsonl")
    ) + sum(h != ST.digest([r for r in rows if r["split"] == s]) for s, h in man["sha256"][BY_SPLIT].items())
    return {"instructions of 0067 looked up": len(texts), "counted": {k: out[k] for k in COUNTED},
            "must be 0": {k: int(v) for k, v in sorted(out.items()) if k not in COUNTED}, "sha256": man["sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = Path(local_paths.value("CUBBY_DATASETS") or ".")
    got = dict({"order": "0073", "what": "long_rows_0073 read back and checked with the evaluator (no VM)"},
               **check(Path(a.set or ds / "long_rows_0073"), ds / "long_rows_0067"))
    print(json.dumps(got, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(got["must be 0"].values()):
        raise SystemExit("the set fails its check and is not to be used")


if __name__ == "__main__":
    main()
