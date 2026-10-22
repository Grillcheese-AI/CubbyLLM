"""plugins_routing -- order 0028's routing line: each plugin's 10 ``examples`` wake it (>= 9) and its 10 ``not`` lines
do not (<= 1), with ``tau_plugin`` tuned on them -- the lines written before any tuning.

Two readings:
- **as the order has it**: the router indexes each plugin's description and examples, so an example scores against
  itself (the in-sample line);
- **leave one out**: each example scored with its own line taken out of the index -- how a new phrasing of the same
  need would fare (the honest number for generalisation).
For every ``tau`` on a grid: per plugin, examples that wake it, ``not`` lines that wake it, and the other plugins' lines
that wake it (the confusion table). The chosen ``tau`` is the one meeting the line for every plugin in the in-sample
reading with the most leave-one-out examples woken; ties go to the higher ``tau`` (fewer wrong wakes).

    python validation/plugins_routing.py [--dirs plugins]
-> validation/logs/plugins_routing_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "standin")]

GRID = [round(0.20 + 0.025 * i, 3) for i in range(25)]          # 0.20 .. 0.80


def table(index, tau: float, loo: bool, enc=None) -> dict:
    from plugins_route import Router
    shared = None if loo else Router(index, enc=enc, tau=tau)
    out = {}
    for name in index.names():
        p = index[name]
        row = {"examples": 0, "not": 0, "others_woken_by_examples": {}, "not_woke": []}
        for ex in p.examples:
            r = Router(index, enc=enc, tau=tau, leave_out=" ".join(ex.split())) if loo else shared
            woke = [n for n, _s in r.wake(ex)]
            row["examples"] += name in woke
            for n in woke:
                if n != name:
                    row["others_woken_by_examples"][n] = row["others_woken_by_examples"].get(n, 0) + 1
        r = shared or Router(index, enc=enc, tau=tau)
        for line in p.not_:
            woke = [n for n, _s in r.wake(line)]
            if name in woke:
                row["not"] += 1
                row["not_woke"].append(line)
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dirs", default=os.path.join(os.path.dirname(HERE), "plugins"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--encoder", action="store_true", help="the serve stack's fastword table (V4), as the server routes")
    a = ap.parse_args()
    enc = None
    if a.encoder:
        from exp_m3_cot_pipeline import V4_TABLE
        from exp_m3_domain_routing import _load_semantic_words
        enc = _load_semantic_words().FastWordEncoder.from_npz(str(V4_TABLE))
    from plugins_index import discover
    idx = discover(a.dirs)
    rep = {"plugins": idx.names(), "lines": {n: [len(idx[n].examples), len(idx[n].not_)] for n in idx.names()},
           "encoder": "fastword v4 (the serve stack's)" if enc is not None else "IDF word overlap (no encoder)", "grid": {}}
    best = None
    for tau in GRID:
        ins, loo = table(idx, tau, False, enc), table(idx, tau, True, enc)
        meets = all(ins[n]["examples"] >= 9 and ins[n]["not"] <= 1 for n in idx.names())
        loo_hits = sum(loo[n]["examples"] for n in idx.names())
        rep["grid"][str(tau)] = {"meets": meets, "in_sample": ins, "leave_one_out": loo}
        if meets and (best is None or loo_hits >= best[1]):
            best = (tau, loo_hits)
    rep["start_tau"] = 0.35
    rep["chosen_tau"] = best[0] if best else None
    rep["chosen"] = rep["grid"][str(best[0])] if best else None
    out = os.path.join(HERE, "logs", f"plugins_routing{'_enc' if enc is not None else ''}_{a.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for tau in ("0.35", str(best[0]) if best else None):
        if tau is None:
            continue
        g = rep["grid"][tau]
        print(f"tau {tau}: meets {g['meets']}")
        for n in idx.names():
            i, l = g["in_sample"][n], g["leave_one_out"][n]
            print(f"  {n:18s} examples {i['examples']}/10 (leave-one-out {l['examples']}/10)  not {i['not']}/10 "
                  f"{i['not_woke']}  others {i['others_woken_by_examples']}")


if __name__ == "__main__":
    main()
