"""Order 0082's mix builder (validation/sft/build_0082_mix.py) on a small fixture: S0's file is copied unchanged, the
program-first plan and chain records are added at their own repeat (H-E18's mix's), no program-first arithmetic
record goes in, a record sharing a prompt / a seed with the held-out eval is left out, and one the VM disagrees with
is left out."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "validation" / "sft"))
import build_0082_mix as B  # noqa: E402

PROG = 'bind frame, SEED, "$E1";'


def rec(i, task, split="train", prompt=None, repeat=1, ref="ok"):
    return {"id": i, "task": task, "split": split, "subtype": "x", "gold": None, "repeat": repeat, "reference": ref,
            "question": prompt or f"q {i}", "prompt": prompt or f"Question:\nq {i}\nProgram:\n", "program": PROG,
            "spans": [{"id": "$E1", "text": "a", "start": 0, "end": 1, "kind": "E"}]}


def raw(i, task, seed, rels=("instance",)):
    return {"id": i, "task": task, "split": "train", "pf": {"seed": seed, "relations": list(rels)}}


def write(p, rs):
    p.write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
    return str(p)


def seed_in_train_bucket(stem):
    return next(f"{stem} {n}" for n in range(100) if B.bucket(f"{stem} {n}") != 0)


def test_the_mix(tmp_path):
    s0 = [rec("plan-1", "plan", repeat=2), rec("chain-1", "chain", repeat=2), rec("arithmetic-1", "arithmetic"),
          rec("plan-v", "plan", "val")]
    held_prompt = "Question:\nthe other writer's wording\nProgram:\n"
    seeds = {k: seed_in_train_bucket(k) for k in ("p", "c", "shared", "bad", "planted")}
    pf = [s0[0], s0[1],                                                    # v12e's own records: not added again
          rec("plan-pf-1", "plan"), rec("chain-pf-1", "chain", repeat=1), rec("arithmetic-pf-1", "arithmetic"),
          rec("plan-pf-2", "plan", prompt=held_prompt),                    # planted: the held-out eval's prompt
          rec("chain-pf-2", "chain"),                                      # its seed is a held-out seed
          rec("plan-pf-3", "plan", ref="bad")]                             # the VM disagrees
    pf_raw = [raw("plan-pf-1", "plan", seeds["p"]), raw("chain-pf-1", "chain", seeds["c"]),
              raw("plan-pf-2", "plan", seeds["planted"]), raw("chain-pf-2", "chain", seeds["shared"]),
              raw("plan-pf-3", "plan", seeds["bad"]), raw("plan-pf-9", "plan", seeds["bad"])]  # 9: lost in conversion
    held = [rec("plan-pf-h", "plan", "val", prompt=held_prompt)]
    held_raw = [raw("plan-pf-h", "plan", seeds["shared"])]

    def vm(program):
        return (True, "bad-value") if program == "bad" else (True, "a")
    out = tmp_path / "mix.jsonl"
    m = B.build(write(tmp_path / "s0.jsonl", s0), write(tmp_path / "pf.jsonl", pf), write(tmp_path / "pfr.jsonl", pf_raw),
                write(tmp_path / "h.jsonl", held), write(tmp_path / "hr.jsonl", held_raw), str(out), vm=vm)
    got = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert got[:4] == s0                                                   # S0's file, unchanged, first
    assert [r["id"] for r in got[4:]] == ["plan-pf-1", "chain-pf-1"] and all(r["repeat"] == 1 for r in got[4:])
    assert m["arithmetic added"] == 0 and not any(r["id"].startswith("arithmetic-pf") for r in got)
    assert m["left out by the checks"] == {"plan: prompt shared with the held-out eval": 1,
                                           "chain: seed shared with the held-out eval (or its bucket)": 1,
                                           "plan: the VM disagrees": 1}
    assert m["s0 plan and chain records identical in the program-first file"] == {"same": 2, "differ": 0}
    assert m["dropped by the slot conversion"] == {"plan": 1, "chain": 0}
    assert m["added train rows (each record's own repeat)"] == {"plan": 1, "chain": 1}
    assert m["program-first share within the family (H-E18's mix's)"] == {"plan": 0.3333, "chain": 0.3333}
