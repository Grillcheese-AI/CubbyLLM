"""Order 0067, the second reader's read: the counts `exp_view_reader.py --score` does not print, from the same stored
answers and the same VM. Per read and split, on the ids given:

- answers unfinished (the grammar's `complete` is false), by number of parts;
- the split's check lines against the part functions the program wrote (where a long answer stops);
- replies built to fail, how many the VM could check, and the false passes among those (the rate a user meets);
- replies built to pass that the program refuses (false fails); rows with every reply checked and agreeing;
- replies, checked, agree and false passes by number of parts.

Counts only: no id and no row text is written. The totals repeat the scorer's (`replies`, `checked`, `agree`,
`false passes`), which is how a log of this script is checked against the scorer's.

  python validation/exp_view_counted_tables.py --out <json>
      --read <label> <answers.jsonl> <typed|typed8|lines> <counted ids json | dev | test | test_long> [--read ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import exp_view_reader as E  # noqa: E402

__wiring__ = "STANDALONE"

SPLITS = ("dev", "test", "test_long")
SEP = "\n---\n"


def lines_and_parts(answer: str) -> tuple[int, int]:
    """(check lines in the split, part functions in the program)."""
    head, _, program = answer.partition(SEP)
    return (sum(1 for ln in head.split("\n") if ln.startswith("check")),
            len(re.findall(r"function part\d+\(", program)))


def shape(checks: int, written: int, done: bool, gold: int) -> list[str]:
    """The labels one answer adds to a split's shape table."""
    out = ["check lines = gold parts" if checks == gold else
           "check lines > gold parts" if checks > gold else "check lines < gold parts"]
    if done:
        out.append("finished: as many parts as check lines" if checks == written
                   else "finished: parts and check lines differ")
        out.append("finished: 5 or more parts written" if written >= 5 else "finished: 4 parts or fewer written")
    else:
        out.append("unfinished: more check lines than parts written" if checks > written else
                   "unfinished: as many check lines as parts written" if checks == written else
                   "unfinished: fewer check lines than parts written")
        out.append(f"unfinished: {written if written < 5 else '5 or more'} parts written")
    return out


def tally(t: Counter, verdict, held: bool) -> bool:
    """One reply added to `t`; True when the reply was checked and the verdict is the generator's."""
    t["replies"] += 1
    t["replies built to fail"] += int(not held)
    if verdict is None:
        return False
    t["checked"] += 1
    t["agree"] += int(verdict == held)
    if held:
        t["checked, built to pass"] += 1
        t["false fails"] += int(not verdict)
    else:
        t["checked, built to fail"] += 1
        t["false passes"] += int(verdict)
    return verdict == held


def read(vm, rows: dict, path: str, grammar: str, ids: str) -> dict:
    raw = open(path, "rb").read()
    answers = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    want = ({s: set(v["ids"]) for s, v in json.load(open(ids, encoding="utf-8")).items()} if ids.endswith(".json")
            else {ids: None})                              # a whole split
    G = E.grammar_class(grammar)
    out = {"answers": Path(path).name, "answers_sha256": hashlib.sha256(raw).hexdigest(), "grammar": grammar,
           "splits": {}}
    for split in SPLITS:
        if split not in want:
            continue
        t, by_parts, shapes = Counter(), defaultdict(Counter), Counter()
        for a in answers:
            if a["split"] != split or (want[split] is not None and a["id"] not in want[split]):
                continue
            r = rows[a["id"]]
            k, ans = r["n_parts"], a["answer"]
            done = bool(G(r["instruction"]).complete(ans))
            t["rows"] += 1
            by_parts[k]["rows"] += 1
            t["answers unfinished"] += int(not done)
            by_parts[k]["answers unfinished"] += int(not done)
            shapes.update(shape(*lines_and_parts(ans), done, k))
            got = E.VR.parse(ans, r["instruction"])
            if "error" in got:
                t["answers not parsed"] += 1
                continue
            right = True
            for rep in r["replies"]:
                v, _ = E.verdict(vm, got["program"], rep["text"])
                right = tally(t, v, rep["held"]) and right
                tally(by_parts[k], v, rep["held"])
            t["rows with every reply checked and agreeing"] += int(right)
            by_parts[k]["rows with every reply checked and agreeing"] += int(right)
        out["splits"][split] = {"counts": dict(t), "by_parts": {str(k): dict(v) for k, v in sorted(by_parts.items())},
                                "shape": dict(sorted(shapes.items()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", nargs=4, action="append", required=True,
                    metavar=("LABEL", "ANSWERS", "GRAMMAR", "IDS"))
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows_p = Path(a.rows or Path(E.local_paths.value("CUBBY_DATASETS") or ".") / "long_rows_0067" / "rows.jsonl")
    rows = {r["id"]: r for r in map(json.loads, open(rows_p, encoding="utf-8"))}
    vm = E.cc.CubelangSession()
    report = {"order": "0067", "what": "the second reader's stored answers: unfinished answers, replies built to "
              "fail, the split against the program, by number of parts (the scorer's totals repeated)",
              "vm": E.vm_identity.identity(), "rows_sha256": hashlib.sha256(open(rows_p, "rb").read()).hexdigest(),
              "reads": {}}
    for label, path, grammar, ids in a.read:
        report["reads"][label] = read(vm, rows, path, grammar, ids)
        for split, v in report["reads"][label]["splits"].items():
            c = Counter(v["counts"])
            print(f"{label} {split}: rows {c['rows']}, unfinished {c['answers unfinished']}; replies {c['replies']}, "
                  f"checked {c['checked']}, agree {c['agree']}, false passes {c['false passes']} "
                  f"(of {c['checked, built to fail']} checked that were built to fail); false fails "
                  f"{c['false fails']} of {c['checked, built to pass']}; rows all right "
                  f"{c['rows with every reply checked and agreeing']}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
