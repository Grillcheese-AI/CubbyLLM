"""validation/selfimprove/fastlearn_day.py -- order 0053 A: the days replayed on CPU, from the cache and its marks.

Arms, on the same cached day, in the same order, 3 seeds (the projection and the sampling):
  N  none: the trunk's own scores (held: the top one; day: sampled from them)
  S  the spiking learner, three-factor (dopamine from its own striatum)
  P  plain STDP: the chosen candidate's synapses strengthen at every choice, whatever the verdict
  D  dense, three-factor: the same projection, every unit's value passed on (no threshold)
Days: day 1, wd/en's stream, then wd/en's held read; day 2, wd/fr's stream with the same learner, then wd/fr's held
read and wd/en's held read again (retention); the bootstrap, wd/fr's stream from a fresh learner, then its held read.
Held reads are greedy and never seen before. During a day a choice is sampled (temperature 1.0) and learned from.
The settings (gain, eta, tau, the norm) are chosen before any held read, on
wd/en's day stream: the grid's best day right rate for S, averaged over the seeds, kept for every arm and for wd/fr.
Writes <dir>/fastlearn_day.json.

    CB_FASTLEARN=1 python validation/selfimprove/fastlearn_day.py --dir <the cache's folder>

Wired: STANDALONE (order 0053's harness).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

ARMS = ("N", "S", "P", "D")
GRID = {"gain": (0.5, 1.0, 2.0, 4.0), "eta": (0.005, 0.02, 0.08), "tau": (0.25, 1.0), "norm": (1.0, 4.0)}
_OFFSETS: dict = {}                                     # (seed, dense, neurons, T) -> the calibrated offset


def load(d: str):
    cache = [json.loads(x) for x in open(os.path.join(d, "fastlearn_cache.jsonl"), encoding="utf-8")]
    marks = [json.loads(x) for x in open(os.path.join(d, "fastlearn_marks.jsonl"), encoding="utf-8")]
    states = np.load(os.path.join(d, "fastlearn_states.npy"))
    probs = []
    for i, (c, m) in enumerate(zip(cache, marks)):
        assert c["id"] == m["id"], "the marks are not the cache's"
        probs.append({**c, "h": states[i], "verdicts": m["verdicts"], "props": m["props"]})
    return probs


def part(probs, family: str, held: bool) -> list[dict]:
    return sorted([p for p in probs if p["family"] == family and p["held"] == held], key=lambda p: p["order"])


class Arm:
    def __init__(self, name: str, settings, d: int, props: list[str], seed: int, calib) -> None:
        from cubbyllm.selfimprove.fastlearn import Encoder, FastLearner
        self.name, self.rng = name, np.random.default_rng(seed + 1000)
        self.fl = None
        if name != "N":
            enc = Encoder(d, settings, seed=seed, dense=(name == "D"))
            k = (seed, name == "D", settings.neurons, settings.T)
            if k not in _OFFSETS:                        # the encoder does not learn: one calibration a seed
                _OFFSETS[k] = (enc.calibrate(calib), enc.offset)
            self.rest, enc.offset = _OFFSETS[k]
            self.fl = FastLearner(enc, props, settings, rule="stdp" if name == "P" else "three", seed=seed)

    def pick(self, p: dict, greedy: bool):
        scores = np.asarray(p["scores"], dtype=np.float64)
        if self.fl is None:
            if greedy:
                return int(np.argmax(scores)), None, None
            z = np.exp(scores - scores.max())
            return int(self.rng.choice(len(z), p=z / z.sum())), None, None
        act = self.fl.enc(p["h"])
        i, prop = self.fl.choose(scores, act, p["props"], greedy)
        return i, prop, act

    def day(self, probs: list[dict], key: str) -> dict:
        right, curve, found = [], [], set()
        for p in probs:
            if not p.get("slot"):
                right.append(False)
                continue
            i, prop, act = self.pick(p, greedy=False)
            v = p["verdicts"][i]
            right.append(v == "right")
            if v == "right":
                found.add(p["id"])
            if self.fl is not None:
                self.fl.learn(act, prop, v, key)
        for s in range(0, len(right), 50):
            curve.append(round(sum(right[s:s + 50]) / len(right[s:s + 50]), 4))
        return {"right": round(sum(right) / max(1, len(right)), 4), "curve": curve, "found": len(found)}

    def held(self, probs: list[dict]) -> dict:
        out = {"right": 0, "wrong": 0, "refused": 0, "caught": 0, "no_slot": 0}
        for p in probs:
            if not p.get("slot"):
                out["no_slot"] += 1
                continue
            out[p["verdicts"][self.pick(p, greedy=True)[0]]] += 1
        n = len(probs)
        return {**out, "n": n, "right_rate": round(out["right"] / max(1, n), 4), "wrong_rate": round(out["wrong"] / max(1, n), 4)}


def run_arm(name, settings, probs, props, seed, calib) -> dict:
    en_s, en_h, fr_s, fr_h = (part(probs, f, h) for f in ("wd/en", "wd/fr") for h in (False, True))
    a = Arm(name, settings, calib.shape[1], props, seed, calib)
    r = {"rest": getattr(a, "rest", None), "day1": a.day(en_s, "wd|en"), "en_held_1": a.held(en_h)}
    r.update(day2=a.day(fr_s, "wd|fr"), fr_held=a.held(fr_h), en_held_2=a.held(en_h))
    b = Arm(name, settings, calib.shape[1], props, seed, calib)
    r.update(bootstrap_day=b.day(fr_s, "wd|fr"), bootstrap_fr_held=b.held(fr_h))
    return r


def main(argv=None) -> None:
    from cubbyllm.selfimprove.fastlearn import Settings, on
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--neurons", type=int, default=8192)
    ap.add_argument("--T", type=int, default=8)
    a = ap.parse_args(argv)
    if not on():
        raise SystemExit("CB_FASTLEARN=1: the learner is off by default")
    t0 = time.time()
    probs = load(a.dir)
    props = sorted({q for p in probs for ps in p["props"] for q in ps})
    en_s = part(probs, "wd/en", False)
    calib = np.stack([p["h"] for p in en_s if p.get("slot")])                      # day 1's states only
    sweep = []
    for gain, eta, tau, norm in itertools.product(GRID["gain"], GRID["eta"], GRID["tau"], GRID["norm"]):
        st = Settings(neurons=a.neurons, T=a.T, gain=gain, eta=eta, tau=tau, norm=norm)
        days = [Arm("S", st, calib.shape[1], props, s, calib).day(en_s, "wd|en")["right"] for s in range(a.seeds)]
        sweep.append({"gain": gain, "eta": eta, "tau": tau, "norm": norm, "day1_right": round(float(np.mean(days)), 4)})
        print(json.dumps(sweep[-1]), flush=True)
    best = max(sweep, key=lambda r: r["day1_right"])
    settings = Settings(neurons=a.neurons, T=a.T, gain=best["gain"], eta=best["eta"], tau=best["tau"], norm=best["norm"])
    res = {"settings": {**best, "neurons": a.neurons, "T": a.T, "rest": settings.rest},
           "sweep": sweep, "arms": {}}
    for arm in ARMS:
        res["arms"][arm] = [run_arm(arm, settings, probs, props, s, calib) for s in range(a.seeds)]
        sh = [r["en_held_1"]["right_rate"] for r in res["arms"][arm]]
        print(f"{arm}: en held {np.mean(sh):.3f} +- {np.std(sh):.3f}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(a.dir, "fastlearn_day.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("settings", "minutes")}, indent=1))


if __name__ == "__main__":
    main()
