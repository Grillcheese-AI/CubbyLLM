"""Order 0073, point 7: which variant each arm keeps, decided on dev and written down before test is read.

Each arm's dev is read twice, free and under the own-text rule (a description may use only the numbers, strings and
views its line's own text gives), and both reads are scored (`exp_view_split_score.py --splits dev`). The rule, as
the order states it: kept is the variant with fewer false passes over checked replies; on a tie, the higher
agreement; on a tie again, the rule on. The two variants do not check the same number of replies, so both rates
are compared as exact fractions; a variant with no checked reply loses to one that has some.

  python validation/exp_view_split_point7.py --m-free <report> --m-own <report> --w-free <report> --w-own <report>
      --out validation/logs/view_split_point7.json

The file is committed before `view_split_read_test.ipynb` will read test or test_long (the notebook looks for it).
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

__wiring__ = "STANDALONE"

COUNTS = ("rows", "rows whole", "rows checked", "replies", "replies checked", "agree", "false passes",
          "checked replies built to fail", "gold checks", "parts right")
RULE = ("fewer false passes over checked replies (exact fractions); on a tie, the higher agreement; on a tie again, "
        "the own-text rule on; a variant with no checked reply loses to one that has some")


def rates(c: dict):
    """(false passes, agreement) over checked replies, exact; None when no reply was checked."""
    n = c.get("replies checked", 0)
    return (Fraction(c.get("false passes", 0), n), Fraction(c.get("agree", 0), n)) if n else None


def choose(free: dict, own: dict) -> dict:
    """{"own_text": the variant kept, "why"} from the two variants' dev counts."""
    f, o = rates(free), rates(own)
    if f is None or o is None:
        if f is None and o is None:
            return {"own_text": True, "why": "neither variant checked a reply: the rule on"}
        return {"own_text": f is None, "why": "the other variant checked no reply"}
    if f[0] != o[0]:
        return {"own_text": o[0] < f[0], "why": "fewer false passes over checked replies"}
    if f[1] != o[1]:
        return {"own_text": o[1] > f[1], "why": "the same false passes; the higher agreement"}
    return {"own_text": True, "why": "the same false passes and the same agreement: the rule on"}


def dev_of(path: str, arm: str, own_text: bool) -> dict:
    """A scored report's dev counts, checked to be that arm's whole dev in that variant."""
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    name = f"arm {arm}, {'own text' if own_text else 'free'}"
    if (rep.get("order"), rep.get("arm"), bool(rep.get("own_text"))) != ("0073", arm, own_text) or "dev" not in rep.get("splits", {}):
        raise ValueError(f"{path} is not the dev report of {name}")
    c = rep["splits"]["dev"]["counts"]
    if c.get("rows not read", 0) or set(rep["splits"]) != {"dev"}:
        raise ValueError(f"{path}: {name} must be dev alone, read whole ({c.get('rows not read', 0)} rows not read)")
    return {"report": Path(path).name, "answers_sha256": rep["answers_sha256"], "rows_sha256": rep["rows_sha256"],
            "counts": {k: c.get(k, 0) for k in COUNTS}}


def decide(reports: dict) -> dict:
    """{"m": {"free": path, "own": path}, "w": {...}} -> the file's content."""
    kept, rows = {}, set()
    for arm in ("m", "w"):
        free, own = dev_of(reports[arm]["free"], arm, False), dev_of(reports[arm]["own"], arm, True)
        rows |= {free.pop("rows_sha256"), own.pop("rows_sha256")}
        kept[arm] = dict(choose(free["counts"], own["counts"]), dev={"free": free, "own": own})
    if len(rows) != 1:
        raise ValueError("the four reports were not scored on the same rows")
    return {"order": "0073", "what": "point 7: the variant each arm keeps, decided on dev before test is read",
            "rule": RULE, "rows_sha256": rows.pop(), "kept": kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    for arm in ("m", "w"):
        for variant in ("free", "own"):
            ap.add_argument(f"--{arm}-{variant}", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    got = decide({arm: {v: getattr(a, f"{arm}_{v}") for v in ("free", "own")} for arm in ("m", "w")})
    Path(a.out).write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    for arm, k in got["kept"].items():
        print(f"arm {arm}: {'own text' if k['own_text'] else 'free'} ({k['why']})")
    print("decided:", a.out)


if __name__ == "__main__":
    main()
