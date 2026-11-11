"""validation/system_eval/run_c.py -- order 0054's arm C (and C-VM, from C's records), at home on CPU.

The stack as ``cubby_serve`` mounts it for ``--ask wikidata --world hdc``: the wiki world, the hdc world behind it, the
Wikidata source live through its cache (a polite rate, the server's own fetch), the lexicon, the everyday table, the
tense check, the rule library (the server's default ``sleep/skills.jsonl`` when present), the VM certifying. No web
search (no key in a measurement), no books index unless ``CUBBY_BOOKS_INDEX`` is set (recorded). The programs seat:
  --seat record   the first pass: every prompt the seat is handed is written to ``prompts.jsonl`` (for Colab), and
                  the seat proposes nothing; the records are the stack with no model in the seat (``c_none``)
  --seat replay   the second pass: Colab's program for each prompt (``programs.jsonl``); the records are C (``c``)
  --seat gather   between Colab rounds, into its own folder: the replay where Colab wrote a program, else the prompt
                  is written down (``prompts.jsonl`` there) and nothing proposed; its records (``c_gather``) are not C
  --seat grilly2  the parity read: the serving seat itself (the 450M on grilly2, the local card; ``--export``,
                  ``--adapter``), on a few questions in short chunks; its records (``c_grilly2``) are compared with C's
  --vm permissive a check of C-VM only (``cvm_check``): the VM's similarity raised to 1 wherever it recovered a value,
                  so the first proposal that yields a value is spoken; compared with C-VM read from C's records
Each record (one jsonl line a question, resumable: a question already written is skipped) keeps the verdict, the
proposer, the refusal's kind, where its facts came from, the VM's calls and time, and C-VM's value
(``CB_RECORD_UNCERTIFIED=1``, set here). CLUTRR is ``clutrr_c.py``.
With ``CB_WD_STORE=1`` (``CB_WD_KINDS``, ``CB_ANSWER_CLASS`` as set) the full Wikidata store is mounted ahead of the
hdc world, as ``cubby_serve/cli.py`` mounts it (order 0055 step 4, the "after"); its snapshot and the flags go in the meta.

    python validation/system_eval/run_c.py --bench simpleqa --seat record --out <dir> [--n 50]

Wired: STANDALONE (order 0054's harness).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL), os.path.join(os.path.dirname(VAL), "standin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

KINDS = {"no_plan": "no_plan", "emit_error": "no_plan", "unparseable": "no_plan",
         "plan_does_not_cover_question": "no_plan",
         "unknown_relation": "relation_missing", "ambiguous_relation": "relation_missing",
         "entity_unresolved": "name_unresolved", "ambiguous_entity": "name_unresolved",
         "retrieval_exhausted": "fact_missing", "no_path": "fact_missing", "no_events_for_date": "fact_missing",
         "ambiguous_hop": "walk_failed", "relation_paths_overflow": "walk_failed", "no_rule": "walk_failed",
         "rule_split": "walk_failed",
         "vm_verify_failed": "not_certified", "answer_type_mismatch": "not_certified", "time_mismatch": "not_certified",
         "time_undated": "not_certified", "contested": "not_certified", "latent_only": "not_certified",
         "attributed_only": "not_certified"}


def kind(reason) -> str | None:
    """The refusal's kind (order 0054's list, plus ``name_unresolved``); ``other:<reason>`` for the rest."""
    if not reason:
        return None
    head = str(reason).split(":")[0].strip()
    return KINDS.get(head, f"other:{head}")


def proposer(rec: dict, seat_called: bool, program: str | None) -> str | None:
    """Who proposed the plan the record walked: the everyday table, the 450M programs seat, the rule library (a
    relation ask) or the host's grammar (the question's shape); None when nothing was proposed."""
    from system_eval.seats import NO_PLAN
    if rec.get("restated"):
        return "everyday"
    if rec.get("scene_shape"):
        return "scene"
    if rec.get("kind") == "relation":
        return "rules"
    if seat_called and program and program != NO_PLAN and (rec.get("plan") or rec.get("verified")):
        return "450m"
    if rec.get("plan") or rec.get("verified") or rec.get("how"):
        return "host"
    return None


def facts_source(rec: dict, run_learned: set) -> str | None:
    """Where the spoken answer's facts came from: ``wikidata`` (the source gave one of them, in this run) or
    ``store`` (the offline worlds); None for an answer with no trace."""
    tr = rec.get("trace") or []
    if not tr:
        return None
    return "wikidata" if any(f in run_learned for f in tr) else "store"


def timed(run_fn, timing: dict, permissive: bool = False):
    """``run_fn`` counted and timed into ``timing``; ``permissive``: wherever the VM recovered a value, its similarity
    is raised to 1 (the C-VM check: the first proposal that yields a value is spoken)."""
    def run(source, fn):
        t = time.perf_counter()
        try:
            r = run_fn(source, fn)
        finally:
            timing["vm_calls"] = timing.get("vm_calls", 0) + 1
            timing["vm_s"] = timing.get("vm_s", 0.0) + time.perf_counter() - t
        if permissive and isinstance(r, dict) and r.get("result") is not None:
            r = {**r, "ok": True, "similarity": max(1.0, r.get("similarity") or 0.0)}
        return r
    return run


def build(seat, online: bool, vm: str):
    from cubbyllm.worlds import wikikg as wk
    from cubbyllm.worlds.hdc_store import HdcWorld, Stores
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.slots import SlotEmitter
    from cubbyllm.local_paths import data_dir
    from cubby_serve.build import slot_index
    wk.ensure_data(("triplets",))
    wiki = wk.wiki_world()
    em = SlotEmitter(seat, index=None)
    em.index = slot_index(wiki)                          # the names the host knows, as the server's index
    if online:
        from cubbyllm.sources.wikidata import WikidataSource
        from cubby_serve.sources.http import wikidata as wikidata_fetch
        source = WikidataSource(offline=False, use_choices=True, fetch=wikidata_fetch())
    else:
        source = "wikidata-offline"                      # the cache only (no network; the cache is not written)
    skills = str(data_dir("sleep", "skills.jsonl"))
    others, wd = [], None
    if os.environ.get("CB_WD_STORE") == "1":             # order 0055 step 4: the full store ahead of Wikidata5M, as
        from cubby_serve.wd_world import build as build_wd   # cubby_serve/cli.py mounts it (the "after")
        wd = build_wd(lambda m: print(m, flush=True))
        others.append(wd)
    loop = AskLoop(em, world=Stores(wiki, *others, HdcWorld()), source=source, lexicon=True, history_path=None,
                   skills=skills if os.path.exists(skills) else None, books=os.environ.get("CUBBY_BOOKS_INDEX") or None)
    timing = {"vm_calls": 0, "vm_s": 0.0}
    loop._run_fn = timed(loop._run_fn, timing, vm == "permissive")
    meta = {"wd_store": wd.state if wd else None, "wd_dump": wd.provenance_name if wd else None,
            "flags": {k: os.environ.get(k) for k in ("CB_WD_STORE", "CB_WD_KINDS", "CB_ANSWER_CLASS", "CB_QWORD_KIND",
                                                          "CB_ORIGINAL_FIRST", "CB_ALIAS_OPEN", "CB_VALUE_DATE",
                                                          "CB_ORIGINAL_ASIDE")},
            "skills": len(loop.skills), "books": bool(os.environ.get("CUBBY_BOOKS_INDEX")), "online": online,
            "everyday": loop._everyday is not None, "tense": loop.tense, "vm": vm}
    return loop, timing, meta


def one(loop, seat, timing, row: dict, run_learned: set) -> dict:
    from system_eval.grade import verdict
    c0, v0, s0 = seat.calls, timing["vm_calls"], timing["vm_s"]
    t = time.perf_counter()
    try:
        rec = loop.ask(row["question"])
    except Exception as e:                               # noqa: BLE001 -- a crash is a row (refused, its reason kept)
        rec = {"answer": None, "verified": False, "reason": f"error: {str(e)[:160]}"}
    called = seat.calls > c0
    prog = getattr(seat, "last_program", None) if called else None
    for l in rec.get("learned") or []:
        if l.get("status") == "accepted" and l.get("fact"):
            run_learned.add(l["fact"])
    spoken = bool(rec.get("verified"))
    cvm = rec.get("answer") if spoken else rec.get("uncertified")
    return {"id": row["id"], "bench": row["bench"], "question": row["question"], "golds": row["golds"][:8],
            "seen": row["seen"], "spoken": spoken, "answer": rec.get("answer"),
            "said": rec.get("reply") or rec.get("answer"), "tier": rec.get("tier"),
            "verdict": verdict(spoken, rec.get("answer"), row["golds"], row["bench"]),
            "reason": rec.get("reason"), "refusal": None if spoken else kind(rec.get("reason") or "no_plan"),
            "proposer": proposer(rec, called, prog), "prompt": seat.last if called else None,
            "plan": rec.get("plan"), "seed": rec.get("seed"), "trace": rec.get("trace") or [],
            "facts_from": facts_source(rec, run_learned) if spoken else None,
            "cvm_answer": cvm, "cvm_verdict": verdict(cvm is not None, cvm, row["golds"], row["bench"]),
            "vm_calls": timing["vm_calls"] - v0, "vm_s": round(timing["vm_s"] - s0, 3),
            "wall_s": round(time.perf_counter() - t, 2),
            # order 0066's checks, as the record keeps them: the question word whose kind was unknown, the word a
            # refusal answered, the value's own date beside the dump's (None: no check said anything)
            "checks": {k: rec[k] for k in ("kind_unknown", "qword", "dates") if rec.get(k)} or None}


def main(argv=None) -> None:
    from system_eval.bench import BENCHES, load
    from system_eval.seats import Gather, Recorder, Replay, SeatFailure
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=[b for b in BENCHES if b != "clutrr"] + ["opposites"])
    ap.add_argument("--seat", required=True, choices=("record", "replay", "gather", "grilly2"))
    ap.add_argument("--export", default=os.environ.get("CUBBY_450M_EXPORT"), help="grilly2: the base export")
    ap.add_argument("--adapter", default=os.environ.get("CUBBY_450M_EMITTER"), help="grilly2: the programs adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--programs", default=None, help="replay: Colab's programs.jsonl (default <out>/programs.jsonl)")
    ap.add_argument("--vm", choices=("certify", "permissive"), default="certify")
    ap.add_argument("--ids", default=None, help="a file of ids (one a line): only these (the C-VM check)")
    ap.add_argument("--offline", action="store_true", help="the Wikidata source from its cache only (a preflight)")
    ap.add_argument("--n", type=int, default=0, help="the first n questions (the preflight)")
    a = ap.parse_args(argv)
    os.environ["CB_RECORD_UNCERTIFIED"] = "1"
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.bench)
    if a.ids:
        keep = {x.strip() for x in open(a.ids, encoding="utf-8") if x.strip()}
        rows = [r for r in rows if r["id"] in keep]
    if a.n:
        rows = rows[:a.n]
    arm = {"record": "c_none", "replay": "c", "gather": "c_gather", "grilly2": "c_grilly2"}[a.seat] + ("_permissive" if a.vm == "permissive" else "")
    path = os.path.join(a.out, f"{arm}_{a.bench}.jsonl")
    done = {json.loads(x)["id"] for x in open(path, encoding="utf-8") if x.strip()} if os.path.exists(path) else set()
    if a.seat == "grilly2":
        from cubbyllm.emit.seats_450m import Cubby450mEmitter
        seat = Cubby450mEmitter(a.export, a.adapter, os.path.join(a.export, "tokenizer.json"))
        seat.calls, seat.last, inner_emit = 0, None, seat.emit

        def counted(prompt, **kw):                       # the replay's bookkeeping, on the live seat
            seat.calls += 1
            seat.last = prompt
            try:
                seat.last_program = inner_emit(prompt, **kw)
            except Exception as e:                       # the model never ran (2026-11-06: a missing module made
                raise SeatFailure(repr(e)) from e        #   125 refusals): the run stops, as a missing program does
            return seat.last_program
        seat.emit = counted
    else:
        seat = Recorder() if a.seat == "record" else             (Gather if a.seat == "gather" else Replay).load(a.programs or os.path.join(a.out, "programs.jsonl"))
    t0 = time.perf_counter()
    loop, timing, meta = build(seat, not a.offline, a.vm)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.bench}: {len(rows)} questions, {len(done)} done, {len(todo)} to go; loaded in "
          f"{time.perf_counter() - t0:.0f}s; {json.dumps(meta)}", flush=True)
    run_learned: set = set()
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            f.write(json.dumps(one(loop, seat, timing, r, run_learned), ensure_ascii=False) + "\n")
            f.flush()
            if a.seat in ("record", "gather") and (i + 1) % 25 == 0:
                seat.save(os.path.join(a.out, "prompts.jsonl"))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.seat in ("record", "gather"):
        seat.save(os.path.join(a.out, "prompts.jsonl"))
    json.dump({**meta, "bench": a.bench, "arm": arm, "rows": len(rows), "wall_s": round(time.perf_counter() - t0, 1)},
              open(os.path.join(a.out, f"{arm}_{a.bench}.meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
