"""grow_mem_readout -- H-E20's gates 0-3 from the two arms' metrics and probes (order 0045).

The notebook ``grow_mem_continue.ipynb`` applied these by hand in its readout cell; the pod runner applies them at
the end of a run. One place, so both read the same numbers the same way. The numbers are the pre-registered ones
(H-E20; `docs/orders/0045-rented-card-grow-and-memory.md`): nothing here moves them.

- gate 0, equal compute: the same tokens a step (within 2 x S), the same card model, training hours within 0.5 h;
- gate 1, growth: the grown arm's held-out mix <= the control's + 0.02 (kill: above by more than 0.05);
- gate 2, memory: on facts_text_far the grown final's exact >= both floors + 0.10 (the control's final, the 450M as
  it is), its `_copy` not above its first reading + 0.05, and the needle's beyond-window cells > 0.25 where the
  control sits at chance (within 0.10 of it);
- gate 3, the decay data: each arm's last reading on instruct, agent and program below its first; the checked sources
  of the grown arm within 0.02 of the control's (kill: both arms above their first reading on every checked source).

    python validation/grow_mem_readout.py --grown <metrics.jsonl> --control <metrics.jsonl> [--cont <metrics.jsonl>]
        [--probe-grown <hypertoken json>] [--probe-control ...] [--probe-base ...] [--needle-grown ...] [--needle-control ...]
        [--seq 2048] [--out readout.json]

Wired: WIRED (``validation/pod/pod_phases.py``, the notebook's readout cell).
"""
from __future__ import annotations

import argparse
import json
import os

__wiring__ = "WIRED"

NEW = ("instruct", "agent", "program")
KEY = ("fineweb_edu", "books", "nemotron_qa", "nemotron_code")
WINDOW = 512
MIX_TOL, MIX_KILL, KEY_TOL, MEM_POINTS, COPY_TOL, NEEDLE_LINE, CHANCE_TOL, HOURS_TOL = 0.02, 0.05, 0.02, 0.10, 0.05, 0.25, 0.10, 0.5


def rows_of(path) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def vals(rows) -> list[dict]:
    return [r for r in rows if "val" in r]


def step_size(rows):
    """Tokens per optimizer step as measured (tokens seen / steps), from the training rows."""
    tr = [r for r in rows if "tokens" in r and r.get("step")]
    return round(tr[-1]["tokens"] / tr[-1]["step"]) if tr else None


def hours_of(rows):
    return max((r["hours"] for r in rows if "hours" in r), default=None)


def start_reading(cont_rows) -> dict:
    """The continuation's last validation before its decay: what both arms start from."""
    cont = vals(cont_rows)
    decay = min((r["step"] for r in cont_rows if r.get("phase") == "decay"), default=10 ** 9)
    return ([r for r in cont if r["step"] <= decay] or cont or [{"val": {}}])[-1]["val"]


def _exact(probe: dict | None, shard: str = "facts_text_far"):
    if not probe:
        return None
    return ((probe.get("shards") or {}).get(shard) or {}).get("exact")


def _beyond(needle: dict | None, key: str = "trained") -> list[float]:
    """The needle's beyond-window cells: a depth at which the needle sits more than WINDOW tokens before the end."""
    if not needle:
        return []
    out = []
    for ln, row in (needle.get(key) or {}).items():
        for dep, v in zip(needle.get("depths") or [], row):
            if (1 - float(dep)) * int(ln) > WINDOW:
                out.append(float(v))
    return out


def gates(grown, control, cont=None, seq: int = 2048, probes: dict | None = None, cards: dict | None = None) -> dict:
    """The readout: {"sizes", "hours", "cards", "table", "checks", "kills", "start"}; a check is True, False or None
    (not measurable yet). `grown` / `control` / `cont`: metrics rows; `probes`: {"grown", "control", "base"} the
    hypertoken reports, {"needle_grown", "needle_control"} the needle reports."""
    probes = probes or {}
    arms = {"grown": vals(grown), "control": vals(control)}
    have = {a: v for a, v in arms.items() if v}
    sizes = {"grown": step_size(grown), "control": step_size(control)}
    hours = {"grown": hours_of(grown), "control": hours_of(control)}
    start = start_reading(cont or [])
    out = {"sizes": sizes, "hours": hours, "cards": cards or {}, "start": start, "table": {}, "checks": {}, "kills": {}}
    keys = sorted({k for v in have.values() for k in v[-1]["val"] if not k.startswith("_")})
    for k in keys + ["_mix", "_copy"]:
        out["table"][k] = {a: v[-1]["val"].get(k) for a, v in have.items()}
        out["table"][k]["start"] = start.get(k)
    checks, kills = out["checks"], out["kills"]
    if len(have) == 2:
        g, c = have["grown"][-1]["val"], have["control"][-1]["val"]
        same_step = bool(sizes["grown"] and sizes["control"] and abs(sizes["grown"] - sizes["control"]) <= 2 * seq)
        same_hours = bool(hours["grown"] and hours["control"] and abs(hours["grown"] - hours["control"]) <= HOURS_TOL)
        same_card = (cards or {}).get("grown") == (cards or {}).get("control") if cards else True
        checks["0. equal compute: the same tokens a step, the same card, hours within 0.5 h"] = same_step and same_hours and bool(same_card)
        checks["1. growth: grown mix <= control mix + 0.02"] = g["_mix"] <= c["_mix"] + MIX_TOL
        kills["1. growth killed: grown mix above the control's by more than 0.05"] = g["_mix"] > c["_mix"] + MIX_KILL
        checks["3b. checked sources: grown within 0.02 of the control"] = all(g.get(k, 9) <= c.get(k, 0) + KEY_TOL for k in KEY if k in c)
    else:
        for k in ("0. equal compute", "1. growth", "3b. checked sources"):
            checks[k] = None
    for a, v in have.items():
        first, last = v[0]["val"], v[-1]["val"]
        checks[f"3a. {a}: instruct, agent, program each below their first reading"] = all(
            last.get(k, 9) < first.get(k, 0) for k in NEW if k in first)
    if len(have) == 2:
        kills["3. decay data killed: both arms above their first reading on every checked source"] = all(
            v[-1]["val"].get(k, 0) > v[0]["val"].get(k, 9) for v in have.values() for k in KEY if k in v[0]["val"])
    eg, ec, eb = _exact(probes.get("grown")), _exact(probes.get("control")), _exact(probes.get("base"))
    if eg is not None and ec is not None and eb is not None:
        checks["2a. memory: grown exact on facts_text_far >= both floors + 0.10"] = eg >= ec + MEM_POINTS and eg >= eb + MEM_POINTS
    else:
        checks["2a. memory: grown exact on facts_text_far >= both floors + 0.10"] = None
    gv = have.get("grown")
    checks["2b. memory: grown _copy not above its first reading + 0.05"] = (
        gv[-1]["val"]["_copy"] <= gv[0]["val"]["_copy"] + COPY_TOL if gv and "_copy" in gv[0]["val"] else None)
    bg, bc = _beyond(probes.get("needle_grown")), _beyond(probes.get("needle_control"))
    chance = None
    if probes.get("needle_control"):
        chance = 1.0 / (int(probes["needle_control"].get("distract", 7)) + 1)
    if bg and bc and chance is not None:
        control_at_chance = all(v <= chance + CHANCE_TOL for v in bc)
        checks["2c. memory: needle beyond-window cells > 0.25 where the control sits at chance"] = (
            control_at_chance and all(v > NEEDLE_LINE for v in bg))
    else:
        checks["2c. memory: needle beyond-window cells > 0.25 where the control sits at chance"] = None
    out["probes"] = {"exact": {"grown": eg, "control": ec, "base": eb}, "needle_beyond": {"grown": bg, "control": bc, "chance": chance}}
    return out


def markdown(r: dict) -> str:
    lines = ["# H-E20 readout", "",
             f"tokens a step: {r['sizes']}; hours: {{{', '.join(f'{a}: {round(h, 2) if h else None}' for a, h in r['hours'].items())}}}"
             + (f"; cards: {r['cards']}" if r.get("cards") else ""), "",
             "| source | grown | control | start |", "|---|---|---|---|"]
    for k, row in r["table"].items():
        tag = " (new shard)" if k in NEW else " (checked)" if k in KEY else ""
        f = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
        lines.append(f"| {k}{tag} | {f(row.get('grown'))} | {f(row.get('control'))} | {f(row.get('start'))} |")
    lines += ["", "## Gates", ""]
    for k, ok in r["checks"].items():
        lines.append(f"- {'PASS' if ok else 'fail' if ok is False else 'not yet'}  {k}")
    for k, ok in r.get("kills", {}).items():
        lines.append(f"- {'KILLED' if ok else 'not killed'}  {k}")
    p = r.get("probes") or {}
    if p:
        lines += ["", f"facts_text_far exact: {p.get('exact')}", f"needle beyond-window cells: {p.get('needle_beyond')}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grown", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--cont", default="")
    for k in ("probe-grown", "probe-control", "probe-base", "needle-grown", "needle-control"):
        ap.add_argument(f"--{k}", default="")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    load = lambda p: json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None  # noqa: E731
    probes = {"grown": load(a.probe_grown), "control": load(a.probe_control), "base": load(a.probe_base),
              "needle_grown": load(a.needle_grown), "needle_control": load(a.needle_control)}
    r = gates(rows_of(a.grown), rows_of(a.control), rows_of(a.cont), a.seq, probes)
    print(markdown(r))
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
        open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8").write(markdown(r))


if __name__ == "__main__":
    main()
