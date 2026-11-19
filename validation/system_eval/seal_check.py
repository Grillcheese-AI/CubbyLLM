"""Order 0076, acceptance lines 3 and 7: every written record sealed (`cubby_serve/records/envelope.py`) and read
back by the verifier (`validation/verify_cheez.py`): "signed" and "same" against the store at the run's snapshot, and
the verifier's time per record (median and 95th percentile, at home).

The keys are made in memory for this check and dropped when it ends: they are test keys, not Nick's, and nothing
sealed here is kept. Counts and times only in the report.

  python validation/system_eval/seal_check.py --records <records.jsonl> --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "validation")]

__wiring__ = "STANDALONE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from joserfc.jwk import OctKey, OKPKey
    import verify_cheez as V
    from cubby_serve.board.store_client import from_env
    from cubby_serve.records import envelope as E
    keys = E.Keys(OKPKey.generate_key("Ed25519"), "check-sign", OctKey.generate_key(256), "check-conv",
                  OctKey.generate_key(256), "check-prov")
    pub = OKPKey.import_key(keys.sign.as_dict(private=False))
    client = from_env()
    snap = (client.ask({"op": "snapshot"}) or {}).get("snapshot")
    said, times, t_seal = Counter(), {"signature": [], "read": []}, []
    for ln in open(a.records, encoding="utf-8"):
        rec = json.loads(ln)["record"]
        t0 = time.perf_counter()
        sealed = E.seal(rec, keys)
        t1 = time.perf_counter()
        s = V.signed(sealed, pub)
        t2 = time.perf_counter()
        r = V.read(sealed, pub, keys.conversation, keys.provenance, ask=client.ask, snapshot=snap)
        t3 = time.perf_counter()
        kind = "certificate" if any("cheez:Certificate" in (n.get("@type") or []) for n in rec["@graph"]) else "refusal"
        said[f"{kind}: {s}, {r['says']}"] += 1
        t_seal.append(t1 - t0)
        times["signature"].append(t2 - t1)
        times["read"].append(t3 - t2)
    client.close()

    def stats(xs):
        xs = sorted(xs)
        return {"median ms": round(1000 * statistics.median(xs), 3), "95th percentile ms": round(1000 * xs[math.ceil(0.95 * len(xs)) - 1], 3)}
    report = {"order": "0076", "what": "lines 3 and 7: every record sealed, then checked by the verifier at home",
              "records": sum(said.values()), "store snapshot": snap, "said": dict(said),
              "verifier time per record": {"anyone (signature)": stats(times["signature"]),
                                           "reader (decrypt, recompute, store)": stats(times["read"])},
              "sealing time per record": stats(t_seal), "keys": "made in memory for this check, never written"}
    open(a.out, "w", encoding="utf-8").write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
