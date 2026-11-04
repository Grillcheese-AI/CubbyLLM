"""r3_450m -- order 0046, R3's model arm: the 450M in the programs seat reads the question and writes the program; the
relation is what its program binds as ``HOP1``, and that wording becomes properties through the serving property table
(``r3_arms.pids_of``), as the host turns any program's relation into what an item's claims are keyed by.

  the seat    ``Cubby450mEmitter`` over a ``Cubby450mTrunk`` on grilly2 (the local card), as ``cubby_serve`` mounts it;
              greedy, at most 120 new tokens. The base is ``base450m_cont``'s export: the programs adapter was trained
              on it (H-E15, live 2026-10-17); on ``base450m``'s export the same adapter writes no program.
  the prompt  the question with each gold item's words as a slot, ``[E1: ...]``, as the host's ``SlotTable.annotate``
              writes it (the host places names; the model never copies one). A row whose names were not located goes
              in as written.

Two halves: ``run`` (this interpreter) writes the job and starts ``main`` under grilly2's interpreter (``uv run``), which
appends one program per row to a jsonl kept off-repo, so a stopped run resumes where it stopped.

    uv run --project <grilly2> python validation/retrieval_bench/r3_450m.py JOB OUT.jsonl

Wired: STANDALONE (R3's bench).
"""
from __future__ import annotations

import json
import os
import re
import sys

__wiring__ = "STANDALONE"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HOP1 = re.compile(r'bind\s+frame\s*,\s*HOP1\s*,\s*"([^"]*)"\s*;')
HOPS = re.compile(r'bind\s+frame\s*,\s*HOP\d+\s*,')
MAX_NEW = 120


def annotate(question: str, mentions: list[dict]) -> str:
    """Each located name as ``[En: words]``, numbered by first position, one number per item."""
    spans = sorted((m["span"][0], m["span"][1], m["qid"]) for m in mentions if m.get("span"))
    ids: dict[str, int] = {}
    out, at = [], 0
    for s, e, q in spans:
        if s < at:
            continue
        ids.setdefault(q, len(ids) + 1)
        out += [question[at:s], f"[E{ids[q]}: {question[s:e].strip()}]"]
        at = e
    return "".join(out + [question[at:]])


def read(program: str) -> dict:
    """{"hop1": wording | None, "hops": n} from a program."""
    m = HOP1.search(program or "")
    return {"hop1": m.group(1).strip() if m else None, "hops": len(HOPS.findall(program or ""))}


def seat(adapter: str | None = None) -> dict:
    """The export and the programs adapter: `adapter` (order 0048's ``450m_wd``), else ``$CUBBY_450M_EMITTER``, else the
    serving one."""
    from cubbyllm.local_paths import value
    models = value("CUBBY_MODELS") or ""
    return {"export": value("CUBBY_450M_EXPORT") or os.path.join(models, "base450m_cont", "grilly"),
            "adapter": adapter or value("CUBBY_450M_EMITTER") or os.path.join(models, "emitter", "cubby450m_v12e_w_step_cont")}


def run(rows: list[dict], out_path: str, adapter: str | None = None) -> dict:
    """Every row's program (resumed from ``out_path``); returns {row id: program} and the seat's report."""
    import subprocess
    import tempfile
    from retrieval_bench.r2_qfhrr import grilly2_dir
    jb = {**seat(adapter), "max_new": MAX_NEW, "rows": [{"id": r["id"], "prompt": annotate(r["question"], r["mentions"])} for r in rows]}
    jp = os.path.join(tempfile.mkdtemp(), "job.json")
    json.dump(jb, open(jp, "w", encoding="utf-8"))
    subprocess.run(["uv", "run", "--project", grilly2_dir(), "python", os.path.abspath(__file__), jp, out_path], check=True, cwd=ROOT)
    got, info = {}, {}
    for line in open(out_path, encoding="utf-8"):
        d = json.loads(line)
        if "seat" in d:
            info = d
        else:
            got[d["id"]] = d
    return {"programs": got, "seat": info}


def main() -> None:
    import time
    sys.path.insert(0, ROOT)
    from cubbyllm.emit.seats_450m import Cubby450mEmitter, Cubby450mTrunk
    jp, op = sys.argv[1], sys.argv[2]
    jb = json.load(open(jp, encoding="utf-8"))
    done = set()
    if os.path.exists(op):
        done = {json.loads(x).get("id") for x in open(op, encoding="utf-8") if x.strip()}
    todo = [r for r in jb["rows"] if r["id"] not in done]
    t0 = time.time()
    trunk = Cubby450mTrunk(jb["export"], os.path.join(jb["export"], "tokenizer.json"), {"programs": jb["adapter"]})
    em = Cubby450mEmitter(trunk)
    with open(op, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seat": em.name, "base": os.path.basename(os.path.dirname(os.path.normpath(jb["export"]))),
                            "interpreter": "grilly2", "max_new": jb["max_new"], "greedy": True}) + "\n")
        for i, r in enumerate(todo):
            t = time.perf_counter()
            prog = em.emit(r["prompt"], max_new_tokens=jb["max_new"])
            f.write(json.dumps({"id": r["id"], "program": prog, "wall_s": round(time.perf_counter() - t, 3)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} programs, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {len(todo)} programs in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
