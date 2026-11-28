"""talk_format2_0079 -- order 0079's perspective addendum, item 1: every talk row of the final mix in the second shape
(``User:`` / ``Cubby:`` to the end), rendered by the function serving uses (`cubbyllm.emit.talk_format.render`).

    python validation/sft/talk_format2_0079.py --rows <talk rows.jsonl> [more ...] [--out <rows in the second shape>]
        [--manifest validation/logs/talk_format2_0079_<date>.json]

Each row's prompt (today's shape) is read back into the host's context (`talk_format.parse_v1`) and rendered again;
the answer is unchanged. A row whose lines cannot be placed for sure (a label inside a question or a wrapped turn)
is left out and counted by family, never guessed. Without ``--out`` it only counts. The rows stay off the repo;
the manifest holds counts only.

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubbyllm.emit import talk_format  # noqa: E402

__wiring__ = "STANDALONE"


def convert(row: dict) -> dict | None:
    """The row in the second shape, or None when its prompt cannot be read back for sure."""
    ctx = talk_format.parse_v1(row["prompt"])
    if ctx is None:
        return None
    return {**row, "prompt": talk_format.render("", ctx), "format": 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    a = ap.parse_args(argv)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    files = {}
    for path in a.rows:
        kept, left = collections.Counter(), collections.Counter()
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            new = convert(row)
            (kept if new is not None else left)[row.get("family") or "?"] += 1
            if new is not None and out is not None:
                out.write(json.dumps(new, ensure_ascii=False) + "\n")
        files[pathlib.Path(path).name] = {"rows": sum(kept.values()) + sum(left.values()),
                                          "converted": sum(kept.values()), "left_out": sum(left.values()),
                                          "left_out_by_family": dict(left.most_common())}
        print(pathlib.Path(path).name, json.dumps(files[pathlib.Path(path).name]), flush=True)
    if out is not None:
        out.close()
    if a.manifest:
        m = {"what": "order 0079 item 1: talk rows in the second shape (User:/Cubby: to the end), by talk_format.render",
             "files": files, "wrote": bool(a.out)}
        pathlib.Path(a.manifest).write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
