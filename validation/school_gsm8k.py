"""Order 0078, S2: the school on GSM8K, its mistakes named by `gsm8k.cube` (cubelang `examples/gsm8k.cube`, run in the
VM through its CLI). Used by `train_school_torch.py --gsm8k ...`.

- **The pool:** S1's step rows (`s1_0078_step.jsonl`) that carry a `gsm8k_id`, in train, and not one of dev's 329
  (`$CUBBY_DATASETS/gsm8k/dev_ids.json`): the school never practises on what its gate reads.
- **The gold's text** by that id (`$CUBBY_DATASETS/gsm8k/train.jsonl`, off-repo).
- **Naming a wrong step** (the school's unit is one step, k): `gsm8k.cube`'s `classify_error` on the gold cut to its
  first k+1 `<<...>>` steps, when the gold's steps are the plan's (as many, each replaying to the plan's value);
  otherwise on the plan's own values written as steps ("<<v=v>>", the plan's values being the program's). Which one
  was used is counted (`source`). A step the VM could not name stays "other", as the school's own default does.

The wrong try's first wrong step is the step itself (one step is one try in --step mode); its kind is the cube's.

Wiring: STANDALONE (`validation/train_school_torch.py`).
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

__wiring__ = "STANDALONE"

KIND = {1: "stopped early", 2: "reversed", 3: "unit conversion", 4: "one number off", 5: "other"}
ANN = re.compile(r"<<[^=>]*=[^>]*>>")


def plain(v: float) -> str:
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


class CubeNamer:
    def __init__(self, exe: str, cube: str, gsm8k_dir: str) -> None:
        self.exe, self.cube = exe, cube
        d = Path(gsm8k_dir)
        self.golds = {r["id"]: r for r in (json.loads(ln) for ln in open(d / "train.jsonl", encoding="utf-8"))}
        self.dev = set(json.loads((d / "dev_ids.json").read_text(encoding="utf-8")))
        self.source: Counter = Counter()
        self._cache: dict = {}

    def keep(self, row: dict) -> bool:
        """A row the school may practise on: GSM8K, train, not dev."""
        gid = row.get("gsm8k_id")
        return bool(gid) and gid in self.golds and gid not in self.dev and row.get("split") == "train"

    def call(self, fn: str, *args) -> float | None:
        key = (fn, *args)
        if key not in self._cache:
            cmd = [self.exe, "run", self.cube, "--fn", fn, "--json"]
            for a in args:
                cmd += ["--arg", str(a)]
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            try:
                got = json.loads(p.stdout.strip().splitlines()[-1])
                self._cache[key] = got.get("result") if got.get("ok") else None
            except Exception:                            # noqa: BLE001
                self._cache[key] = None
        return self._cache[key]

    def gold_text(self, row: dict) -> str:
        """The gold cut to its first k+1 steps, when its steps are the plan's; else the plan's values as steps."""
        k, plan = row["stats"]["k"], row.get("plan_values") or []
        answer = self.golds[row["gsm8k_id"]]["answer"]
        anns = ANN.findall(answer)
        if len(anns) == len(plan) and k < len(anns):
            cut = answer[:answer.index(anns[k]) + len(anns[k])]
            vals = [self.call("solve", answer[:answer.index(a) + len(a)]) for a in anns[:k + 1]]
            if all(v is not None and abs(v - p) <= 1e-6 * max(1.0, abs(p)) for v, p in zip(vals, plan)):
                self.source["the gold's own steps"] += 1
                return cut
        self.source["the plan's values as steps"] += 1
        return "".join(f"<<{plain(v)}={plain(v)}>>" for v in plan[:k + 1])

    def for_row(self, row: dict):
        """A `classify` for `school.mistake_rows` (classify_error's arguments; the cube's kind)."""
        text, question = self.gold_text(row), self.golds[row["gsm8k_id"]]["question"]

        def classify(wrong, gold, stated=None, right_values=None, wrong_values=None):
            values = ";".join(plain(v) for v in (wrong_values or [wrong]))
            code = self.call("classify_error", text, question, values)
            return KIND.get(int(code), "other") if code else "other"
        return classify


__all__ = ["CubeNamer", "KIND"]
