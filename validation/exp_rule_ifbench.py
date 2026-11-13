"""Order 0067, the reader's fifth round: IFBench's test set as fresh held-out kinds (58 kinds no row was built from).

Each IFBench prompt (a task with its constraints written in) goes to the reader whole, as a user would send it. Read
here, with no checker: how the reader answers each prompt (a rule written, "asked", "none"), by IFBench kind. The plans
are kept (``--keep``) for the agreement read, which runs IFBench's own checker code and waits for Nick's yes (this
machine's permission check held that code: it was downloaded).

The data: IFBench's test parquet (AllenAI, ODC-BY-1.0), off-repo; its path comes from ``--data`` or
``$CUBBY_DATASETS/ifbench/ifbench_test.parquet``.

  python validation/exp_rule_ifbench.py --export <450M export> --tokenizer <tokenizer.json> --adapter <rules adapter>
      [--data <parquet>] [--out <json>] [--keep <plans.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cubbyllm import local_paths  # noqa: E402
from cubbyllm.speech import rule_reader as rr  # noqa: E402

__wiring__ = "STANDALONE"


def load(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pylist()
    return [{"key": r["key"], "prompt": r["prompt"], "kinds": list(r["instruction_id_list"])} for r in t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    data = Path(a.data or Path(local_paths.value("CUBBY_DATASETS") or ".") / "ifbench" / "ifbench_test.parquet")
    items = load(data)
    from cubbyllm.emit.seats_450m import Cubby450mTrunk
    trunk = Cubby450mTrunk(a.export, a.tokenizer, {"programs": a.adapter})
    tk, eos = trunk.tokenizer, trunk.eos

    def gen(ids, allowed, max_new):
        ids = ids[-trunk.max_ctx:]
        return trunk.generate("programs", [ids], max_new, prefix_allowed_tokens_fn=lambda b, s: allowed(s.tolist()[len(ids):]))[0]

    t0, how, per, kept, written = time.time(), Counter(), defaultdict(Counter), {}, []
    for it in items:
        got = rr.read(it["prompt"], gen, lambda t: tk.encode(t).ids, eos)
        how[got["how"]] += 1
        for k in it["kinds"]:
            per[k].update({"prompts": 1, got["how"]: 1})
        kept[str(it["key"])] = {"kinds": it["kinds"], "how": got["how"], "plan": got["plan"], "surface": got["surface"]}
        if got["how"] == "written" and len(written) < 40:
            written.append({"kinds": it["kinds"], "key": str(it["key"]), "rule": got["surface"]})   # ids only: no prompt text
    n = len(items)
    out = {"order": "0067", "what": "IFBench's test prompts, read whole by the reader: how often an unseen kind is "
                                    "answered 'asked' instead of a rule; agreement waits for IFBench's checker",
           "adapter": Path(a.adapter).name, "prompts": n, "kinds": len(per),
           "how": dict(how), "shares": {h: round(c / max(1, n), 4) for h, c in how.items()},
           "per_kind": {k: dict(v) for k, v in sorted(per.items())}, "some_written": written,
           "seconds": round(time.time() - t0, 1),
           "agreement": "not read: IFBench's checker code waits for Nick's yes"}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.keep:
        Path(a.keep).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("prompts", "kinds", "how", "shares", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
