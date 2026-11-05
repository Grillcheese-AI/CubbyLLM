"""validation/selfimprove/loop.py -- order 0049's T1 nights, on the GPU (a Colab notebook Nick runs).

Each request: its key (``key_of``: the coarse signature read from its record -- the task, the relation words left out,
the language, the shape, a stepped problem's step count), the router's adapter, one greedy program, the VM's mark, the
ledger. Every --night requests, a night:
  probe     each candidate (>= min_failures in the window, not probed within ``recheck`` nights): up to probe_train of
            its failed problems practised by the base adapter (the VM marking each attempt; the mistake memory), a
            probe adapter trained one pass over their right programs (at least 10 steps), and its gain read as the
            gate reads: the greedy right rate on up to probe_held of the key's other problems, probe against base
            (Cowork, 2026-11-05; the first T1's held loss on the base's own samples could only rise). Too few right
            programs (a family whose golds carry no rule): no gain, not flagged. Nothing is learned from a reference
            program, only from what the VM marks right.
  practice  8 sampled attempts a problem; where those find fewer than ``bootstrap_min`` right programs among a key's
            problems, the open ones get up to ``max_tries`` in all (32). The tries are reported per key
  adapter   a flagged gap: its failed problems in the window practised, a LoRA from the base adapter
            trained on the right set (T1's adapter: right programs only, arm A), the gate, the router
  gate      the gap's held-out problems rise; every other key's held-out problems (the emitter families) within 1 point.
            E8 reworded and R3's rows need the full stack: they are read at home on the saved adapters
Writes <out>/requests.jsonl, <out>/nights.jsonl, <out>/adapters/<gap>/, <out>/t1.json.

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cubbyllm.selfimprove import signature  # noqa: E402
from cubbyllm.selfimprove.gaps import GapFinder, Rule  # noqa: E402
from cubbyllm.selfimprove.ledger import Ledger  # noqa: E402
from cubbyllm.selfimprove.mistakes import MistakeMemory  # noqa: E402
from cubbyllm.selfimprove.router import BASE, Gate, Router  # noqa: E402

__wiring__ = "STANDALONE"

CAPS = {"arithmetic": 96, "wd": 160, "plan": 200, "chain": 480, "role_binding": 640, "kernel": 640}


def key_of(rec: dict, program: str | None = None) -> str:
    """The ledger's and the router's key, read from the request's record (or from an attempt, for the routing read)."""
    steps = int((rec.get("stats") or {}).get("K") or 0) if rec["task"] == "arithmetic" else 0
    return signature.of(program if program is not None else rec["reference"], rec.get("lang") or "en",
                        task=rec["task"], steps=steps).key(coarse=True)


def cap(rec: dict) -> int:
    return CAPS.get(rec["task"], 640)


def safe(key: str) -> str:
    return re.sub(r"[^\w.-]+", "_", key).strip("_")[:80]


def ok(m: dict) -> bool:
    return m["verdict"] == "right" and m["gold_right"] is not False


class Loop:
    def __init__(self, trunk, marker, stream: list[dict], held: list[dict], out: str, rule: Rule, a) -> None:
        self.t, self.mark, self.out, self.a = trunk, marker, out, a
        self.led, self.gf = Ledger(window=a.window), GapFinder(rule)
        self.router, self.gate = Router(latent_after=a.latent_after), Gate(battery_drop=0.01)
        self.mm = MistakeMemory(a.max_tries)
        self.by_id = {r["id"]: r for r in stream}
        self.held = {}
        for r in held:
            self.held.setdefault(key_of(r), []).append(r)
        self.weights, self.base_rates, self.result = {}, {}, {"adapters": {}, "nights": [], "tries": {}}
        self.base_ok = {}                                  # request id -> the base adapter's greedy mark (probe reads)
        os.makedirs(os.path.join(out, "adapters"), exist_ok=True)
        self.req_log = open(os.path.join(out, "requests.jsonl"), "a", encoding="utf-8")
        self.night_log = open(os.path.join(out, "nights.jsonl"), "a", encoding="utf-8")

    def event(self, **kw) -> None:
        print(json.dumps(kw, ensure_ascii=False)[:400], flush=True)
        self.night_log.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.night_log.flush()

    # -- reading ------------------------------------------------------------------------------------------------------
    def read(self, rows: list[dict], weights) -> list[dict]:
        """One greedy program a row under ``weights``, marked."""
        self.t.use(weights)
        progs = self.t.programs([r["prompt"] for r in rows], [cap(r) for r in rows])
        return [dict(self.mark(r, p[0]), program=p[0]) for r, p in zip(rows, progs)]

    def rate(self, rows: list[dict], weights) -> float | None:
        return None if not rows else sum(ok(m) for m in self.read(rows, weights)) / len(rows)

    def base_rate(self, key: str) -> float | None:
        if key not in self.base_rates:
            self.base_rates[key] = self.rate(self.held.get(key, []), self.t.base)
        return self.base_rates[key]

    # -- the stream ---------------------------------------------------------------------------------------------------
    def serve(self, reqs: list[dict]) -> None:
        names = [self.router.route(key_of(r), r["id"]) for r in reqs]
        for name in sorted(set(names)):
            rows = [r for r, n in zip(reqs, names) if n == name]
            for r, m in zip(rows, self.read(rows, self.t.base if name == BASE else self.weights[name])):
                k = key_of(r)
                self.led.add(k, m["verdict"], m["gold_right"], request=r["id"])
                self.req_log.write(json.dumps({"id": r["id"], "night": r["night"], "family": r["family"], "key": k,
                                               "attempt_key": key_of(r, m["program"]), "adapter": name,
                                               "verdict": m["verdict"], "gold_right": m["gold_right"],
                                               "why": m["why"]}, ensure_ascii=False) + "\n")
        self.req_log.flush()

    def practice(self, problems: list[dict], key: str, weights) -> None:
        """``tries`` sampled attempts a problem, in order, until one is right (the mistake memory keeps the rest); where
        they find fewer than ``bootstrap_min`` right programs among these problems, the open ones go on to the memory's
        cap (``max_tries``)."""
        self._tries(problems, key, weights, min(self.a.tries, self.mm.tries))
        right = self._right(problems, key)
        if right < self.a.bootstrap_min and self.mm.tries > self.a.tries:
            self._tries(problems, key, weights, self.mm.tries)
            after = self._right(problems, key)
            t = self.result["tries"].setdefault(key, [])
            t.append({"problems": len(problems), "right_after_tries": right, "right_after_max": after})
            self.event(what="bootstrap", key=key, problems=len(problems), right_after_tries=right,
                       right_after_max=after)

    def _right(self, problems: list[dict], key: str) -> int:
        got = dict(self.mm.right_set(key))
        return sum(r["id"] in got for r in problems)

    def _tries(self, problems: list[dict], key: str, weights, upto: int) -> None:
        """Each open problem's attempts brought up to ``upto`` in all (a problem practised before keeps its count)."""
        need = {r["id"]: upto - (self.mm.problems[r["id"]].attempts if r["id"] in self.mm.problems else 0)
                for r in problems if self.mm.open(r["id"])}
        todo = [r for r in problems if need.get(r["id"], 0) > 0]
        if not todo:
            return
        k = max(need[r["id"]] for r in todo)
        self.t.use(weights)
        progs = self.t.programs([r["prompt"] for r in todo], [cap(r) for r in todo], k=k,
                                temperature=self.a.temperature, rows=max(64, k * 2))
        for r, ps in zip(todo, progs):
            ps = ps[:need[r["id"]]]
            for p in ps:
                if not self.mm.open(r["id"]):
                    break
                m = self.mark(r, p)
                self.mm.attempt(r["id"], key, p, ok(m), m["why"] or m["verdict"])

    def failed(self, key: str, n: int) -> list[dict]:
        ids = list(dict.fromkeys(self.led.requests(key, failed_only=True)))
        return [self.by_id[i] for i in ids[-n:]]

    def rights(self, rows: list[dict]) -> list[tuple[str, str]]:
        got = dict(self.mm.right_set())
        return [(r["prompt"], got[r["id"]]) for r in rows if r["id"] in got]

    def probe(self, key: str, n_train: int, n_held: int) -> float | None:
        """The gain in the greedy right rate (a share: 0.05 is 5 points) on the key's held problems, or None."""
        probs = self.failed(key, n_train)
        used = {r["id"] for r in probs}
        held = [self.by_id[i] for i in list(dict.fromkeys(self.led.requests(key)))[::-1] if i not in used][:n_held]
        if len(probs) < 20 or len(held) < 20:
            return None
        self.practice(probs, key, self.t.base)
        tr = self.rights(probs)
        gain, base, new = None, None, None
        if len(tr) >= self.a.probe_min_right:
            todo = [r for r in held if r["id"] not in self.base_ok]
            for r, m in zip(todo, self.read(todo, self.t.base)):
                self.base_ok[r["id"]] = ok(m)
            base = sum(self.base_ok[r["id"]] for r in held) / len(held)
            self.t.use(self.t.base)
            self.t.train(tr, steps=max(self.a.probe_steps, math.ceil(len(tr) / self.a.batch)), batch=self.a.batch,
                         lr=self.a.lr)
            new = self.rate(held, self.t.state())
            gain = new - base
            self.t.use(self.t.base)
        self.event(what="probe", key=key, problems=len(probs), right_train=len(tr), held=len(held),
                   base=None if base is None else round(base, 4), probe=None if new is None else round(new, 4),
                   gain=None if gain is None else round(gain, 4))
        return gain

    def adapter(self, key: str, night: int) -> None:
        t0 = time.time()
        probs = self.failed(key, self.a.practice_max)
        self.practice(probs, key, self.t.base)
        right = [(self.by_id[pid]["prompt"], prog) for pid, prog in self.mm.right_set(key) if pid in self.by_id]
        rec = {"night": night, "practised": len(probs), "right": len(right), "memory": self.mm.stats(key)}
        if len(right) < self.a.min_right:
            self.result["adapters"][key] = {**rec, "promoted": False, "why": "too few right programs"}
            return self.event(what="adapter", key=key, **self.result["adapters"][key])
        steps = max(self.a.min_steps, min(self.a.max_steps, math.ceil(self.a.epochs * len(right) / self.a.batch)))
        self.t.use(self.t.base)
        train = self.t.train(right, steps=steps, batch=self.a.batch, lr=self.a.lr)
        w = self.t.state()
        minutes = (time.time() - t0) / 60
        gap_base, gap_new = self.base_rate(key), self.rate(self.held.get(key, []), w)
        battery_keys = [k for k in self.held if k != key and k not in self.gf.gaps]
        b_base = {k: self.base_rate(k) for k in battery_keys}
        b_new = {k: self.rate(self.held[k], w) for k in battery_keys}
        promoted, why = self.gate.judge(gap_base or 0.0, gap_new or 0.0, b_base, b_new)
        name = safe(key)
        self.t.save(w, os.path.join(self.out, "adapters", name), {"gap": key, "night": night, "train": train})
        if promoted:
            self.router.promote(key, name)
            self.weights[name] = w
        self.result["adapters"][key] = {**rec, "train": train, "minutes": round(minutes, 2), "gap_base": gap_base,
                                        "gap_new": gap_new, "lift_points": None if gap_new is None or gap_base is None
                                        else round(100 * (gap_new - gap_base), 1), "battery_base": b_base,
                                        "battery_new": b_new, "promoted": promoted, "why": why}
        self.event(what="adapter", key=key, **self.result["adapters"][key])

    def night(self, n: int) -> None:
        new = self.gf.night(self.led, n, self.probe)
        for k in new:
            self.adapter(k, n)
        self.result["nights"].append({"night": n, "flagged": new, "failures": dict(self.led.failures()),
                                      "loaded": self.router.loaded()})
        self.event(what="night", night=n, flagged=new, loaded=self.router.loaded())

    def run(self, stream: list[dict]) -> dict:
        nights = sorted({r["night"] for r in stream})
        for n in nights:
            self.serve([r for r in stream if r["night"] == n])
            self.night(n)
        self.result["gaps"] = self.gf.gaps
        self.result["routing"] = self.routing()
        json.dump(self.result, open(os.path.join(self.out, "t1.json"), "w", encoding="utf-8"), indent=1)
        return self.result

    def routing(self) -> dict:
        """The router's line: requests outside a gap sent to its adapter. Keys are read from the request, so that is 0
        by construction; ``attempt_key`` says how often the key read from the base's own attempt would differ."""
        rows = [json.loads(x) for x in open(os.path.join(self.out, "requests.jsonl"), encoding="utf-8")]
        gaps = set(self.gf.gaps)
        out_gap = [r for r in rows if r["key"] not in gaps]
        return {"requests": len(rows), "outside_gaps": len(out_gap),
                "outside_sent_to_an_adapter": sum(r["adapter"] != BASE for r in out_gap),
                "attempt_key_differs": sum(r["attempt_key"] != r["key"] for r in rows),
                "attempt_key_into_a_gap": sum(r["attempt_key"] in gaps and r["key"] not in gaps for r in rows)}


def base_read(trunk, marker, held: list[dict], val: list[dict]) -> dict:
    """The base adapter per arithmetic level: the stream's held problems (train records kept out of the stream) and the
    val split. The planted level is one above the highest level right at least half the time on the held problems."""
    lp = Loop.__new__(Loop)
    lp.t, lp.mark = trunk, marker
    out = {}
    for name, rows in (("held", held), ("val", val)):
        lv = {}
        for r in rows:
            if r["task"] == "arithmetic":
                lv.setdefault(int(r["stats"]["K"]), []).append(r)
        out[name] = {k: {"n": len(v), "right": round(lp.rate(v, trunk.base), 4)} for k, v in sorted(lv.items())}
    good = [k for k, v in out["held"].items() if v["right"] >= 0.5]
    out["base_level"] = max(good) if good else None
    out["planted"] = None if out["base_level"] is None else out["base_level"] + 1
    return out


__all__ = ["CAPS", "Loop", "base_read", "cap", "key_of"]
