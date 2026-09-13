"""exp_r15_replay -- the three policies over a recorded exp_r15 stream, instantly (no GPU, no VM).

The stream (every proposer's outcome and cost on every question) was recorded once by
exp_r15_striatum.py; policy questions -- cost weight, learning rate, exploration -- replay
on it in seconds. Same tallies as the live run.

  python validation/exp_r15_replay.py [--stream exp_r15_striatum_stream.json] [--cost-weight 0.5 0.0 1.0]
"""
from __future__ import annotations

import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "validation" / "logs"
for p in (ROOT, ROOT / "validation"):
    sys.path.insert(0, str(p))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_r15_striatum import replay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(LOGS / "exp_r15_striatum_stream.json"))
    ap.add_argument("--cost-weight", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from cubbyllm.reasoning.striatum import Striatum
    d = json.loads(pathlib.Path(a.stream).read_text(encoding="utf-8"))
    stream = [(r["q"], r["gold"], {p: tuple(o) for p, o in r["outcomes"].items()}) for r in d["stream"]]
    proposers = ("grammar", "hippocampus", "emitter", "frontier")
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)
    log(f"stream: {len(stream)} questions ({d.get('n_items', '?')} matched chains x 4 + the SimpleQA record)")
    log(f"{'policy':22s}{'certified':>10}{'WRONG':>7}{'proposals':>11}{'emitter':>9}{'frontier':>9}{'hippo':>7}{'grammar':>9}{'seconds':>9}")
    rows = {}
    for policy in ("fixed", "oracle"):
        t = replay(stream, policy, None, proposers); rows[policy] = t
        log(f"{policy:22s}{t['certified']:10d}{t['WRONG']:7d}{t['proposals']:11d}{t['tried:emitter']:9d}{t['tried:frontier']:9d}{t['tried:hippocampus']:7d}{t['tried:grammar']:9d}{t['seconds']:9.0f}")
    best = None
    for cw in a.cost_weight:
        st = Striatum(alpha=a.alpha, seed=a.seed)
        t = replay(stream, "striatum", st, proposers, cost_weight=cw); rows[f"striatum cw={cw}"] = t
        log(f"{'striatum cw=' + str(cw):22s}{t['certified']:10d}{t['WRONG']:7d}{t['proposals']:11d}{t['tried:emitter']:9d}{t['tried:frontier']:9d}{t['tried:hippocampus']:7d}{t['tried:grammar']:9d}{t['seconds']:9.0f}")
        if best is None or t["seconds"] < best[1]["seconds"]:
            best = (cw, t, st)
    fx, ora = rows["fixed"], rows["oracle"]
    cw, t, st = best
    log(f"\nbest cost weight {cw}: {fx['seconds'] - t['seconds']:.0f}s faster than fixed ({100 * (1 - t['seconds'] / max(1, fx['seconds'])):.0f}%), "
        f"oracle floor {fx['seconds'] - ora['seconds']:.0f}s; emitter calls {fx['tried:emitter']} -> {t['tried:emitter']} (oracle {ora['tried:emitter']}); "
        f"certified {t['certified']} = fixed {fx['certified']}; WRONG {t['WRONG']}")
    log("\nwhat the striatum expects at the best weight (proposer | shape -> expected reward, n), top 12 by n:")
    for k, v in sorted(st.expected.items(), key=lambda kv: -st.n[kv[0]])[:12]:
        log(f"  {k:48s} {v:+.2f}  (n={st.n[k]})")
    (LOGS / f"exp_r15_replay{a.tag}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / f"exp_r15_replay{a.tag}.json").write_text(json.dumps({k: dict(v) for k, v in rows.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
