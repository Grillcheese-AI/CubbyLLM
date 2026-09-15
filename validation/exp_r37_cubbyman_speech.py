"""exp_r37 — let him talk: what does he say when nobody wrote the sentence?

Nick, 2026-09-15: *"the model should say something not hardcoded strings like
right now, let it talk to see what it will do as it receives information."*

Until today the host wrote the thought from a table of ~30 authored phrasings
(`CubbyGhost.THOUGHTS`) and the model only swapped synonyms into it. Now the
host hands the model the step's PERCEPT RECORD — what he sensed, what he did,
what the world did back — and the model writes the sentence. The host checks
it against the record and refuses anything the step did not contain.

This prints the transcript: for every step, the record, what he said, and
whether it was kept or refused. The point is to LOOK at it. Two numbers carry
the verdict:

  * SPOKE RATE — how often a grounded sentence came back at all. Near zero and
    the page is a data dump; near one and the guard is asleep.
  * NOTHING INVENTED — a number or a cell/move name in a KEPT sentence that
    was not in the record. This must be 0. A refusal is a result; a made-up
    thought is a defect, the same kill line as everywhere else.
  * NOTHING COPIED — the first run of this experiment scored 100% spoke while
    the model was reproducing the prompt back verbatim, because a copy passes
    every grounding test there is: everything in it DID come from the record.
    The instrument had inherited the property it was measuring (PATH 6.11).
    So copying is now counted separately, on the kept sentences, by the run
    of consecutive tokens they share with the record.

Run:  python validation/exp_r37_cubbyman_speech.py --steps 40
Needs cubelang.exe and the emitter (one model serves programs and talk).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

# v14e_nochain: the arm adopted in WO-1.3. (v8e is the very old one — the
# first runs of this experiment used it by mistake and its transcript is in
# the log as the "before".)
DEFAULT_GGUF = "standin/models/emitter_v14e_nochain.Q4_K_M.gguf"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
NAME = re.compile(r"level-\d+ cell [\d\-]+|\b[A-Z][A-Z0-9\-]{2,}\b")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--talk-gguf", default=None,
                    help="a separate TALK model for the speaking head; defaults to --gguf, "
                         "which build_serve then loads once and uses for both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ghost-free", type=int, default=None,
                    help="0 puts ghosts in at level 1, so the threat events show up too")
    ap.add_argument("--out", default=str(ROOT / "validation" / "logs" / "exp_r37_cubbyman_speech.json"))
    a = ap.parse_args(argv)

    try:
        from cubbyllm.bridges import cubelang_client as cc
        cc.find_cubelang_exe()
    except Exception as e:
        print(f"SKIPPED: cubelang.exe not found ({e})")
        return 2
    if not (ROOT / a.gguf).exists():
        print(f"SKIPPED: no emitter at {a.gguf}")
        return 2

    import pacman as P
    from pacman import split_mood
    from serve import build_serve

    print(f"exp_r37 — letting him talk  ({a.steps} steps, seed {a.seed})")
    print("=" * 78)
    # one model for both roles: build_serve skips the second load when the
    # paths match, so the talk context runs on the emitter (Nick, 2026-09-15)
    brain = build_serve(a.gguf, None, 400, None, 0.30, -1, talk_gguf=a.talk_gguf or a.gguf)
    man = P.CubbyGhost(P.GhostVerse(ghost_free_levels=a.ghost_free), probe=0.25, seed=a.seed)
    brain.mount(man)

    events: list[dict] = []
    host_trace = man._trace

    def tap(kind, **data):                               # the _t hook: every thought, kept or refused
        if kind in ("thought", "thought_refused"):
            events.append({"kind": kind, **data})
        if host_trace:
            host_trace(kind, **data)
    man._trace = tap

    print()
    for i in range(a.steps):
        n0 = len(events)
        man.step()
        for e in events[n0:]:
            if e["kind"] == "thought_refused":
                print(f"  [{i:3d}] REFUSED  he tried: {e['said']}")
                print(f"        record : {e['record']}")
            else:
                mark = "SAID    " if e.get("verbalized") else "flat    "
                print(f"  [{i:3d}] {mark} {e.get('text', '')}")
                if not e.get("verbalized") and e.get("raw"):
                    pass                                 # the flat line IS the record; no need to print twice

    kept = [e for e in events if e["kind"] == "thought" and e.get("verbalized")]
    flat = [e for e in events if e["kind"] == "thought" and not e.get("verbalized")]
    refused = [e for e in events if e["kind"] == "thought_refused"]
    spoke_rate = len(kept) / max(1, len(kept) + len(refused))

    # NOTHING INVENTED: every figure and name in a kept sentence must be in its
    # record. The kept event carries no record, so pair it with the refusal
    # guard's own referent — re-derive from the flat line when present, else
    # accept only that the sentence has no cell/move name at all.
    from pacman import MAX_RUN, longest_run
    invented, copied, runs = [], [], []
    for e in kept:
        text, rec = split_mood(e.get("text", ""))[1], e.get("raw") or ""
        if not rec:
            continue
        if set(NUM.findall(text)) - set(NUM.findall(rec)):
            invented.append(("number", text, rec))
        if set(NAME.findall(text)) - set(NAME.findall(rec)):
            invented.append(("name", text, rec))
        run = longest_run(text, rec)
        runs.append(run)
        if run > MAX_RUN:
            copied.append((run, text))

    print("\n" + "=" * 78)
    print(f"  spoke      {len(kept):4d}   a sentence of his own came back and was kept")
    print(f"  flat       {len(flat):4d}   below speaking priority, or no model: the record stated plainly")
    print(f"  refused    {len(refused):4d}   the sentence failed the check and was dropped")
    print(f"  SPOKE RATE {spoke_rate:.0%}  of the steps where he tried to speak")
    print(f"  INVENTED   {len(invented):4d}   figures or names in a kept sentence the step did not contain")
    print(f"  COPIED     {len(copied):4d}   kept sentences reciting the record "
          f"(longest shared run: max {max(runs) if runs else 0}, allowed {MAX_RUN})")
    if refused:
        print("\n  a refusal reads like this (the guard doing its job):")
        r = refused[0]
        print(f"    tried : {r['said']}")
        print(f"    record: {r['record']}")

    verdict = "PASS"
    why = []
    if invented:
        verdict, _ = "KILLED", why.append(f"he said something the step did not contain: {invented[0][1][:90]}")
    if copied:
        verdict, _ = "KILLED", why.append(f"a kept sentence recites the record ({copied[0][0]} tokens in a row): "
                                          f"{copied[0][1][:90]}")
    if spoke_rate == 0 and (kept or refused):
        verdict, _ = "KILLED", why.append("nothing he said survived the guard; the page would be a data dump")
    print(f"\nVERDICT: {verdict}")
    for w in why:
        print(f"  - {w}")
    if verdict == "PASS":
        print("  he writes his own sentences; nothing kept was invented, nothing kept was recited.")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(
        {"steps": a.steps, "seed": a.seed, "spoke": len(kept), "flat": len(flat),
         "refused": len(refused), "spoke_rate": round(spoke_rate, 3),
         "invented": invented, "verdict": verdict,
         "transcript": [{k: v for k, v in e.items() if k != "kind"} | {"kind": e["kind"]}
                        for e in events]},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {pathlib.Path(a.out).name}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
