"""H-E19 beam: pick each step FROM THE MENU (every legal op on the held values, or a stop) by the adapter's own
score, and branch to the top-3 only where the model is unsure (margin between its first and second choice < tau).

Why: the full-menu ranking puts the right step in the top-3 86% of the time, the margin is calibrated (quartile 1
top-1 0.41, quartile 4 0.98), and the loop's 0.229 is the product of per-step accuracies -- errors compound with no
recovery. Free generation also writes worse than the menu ranks (teacher-forced writing 0.549 at later steps vs menu
top-1 0.58-0.62). So:
  --mode calibrate   score every step row of a DEV set (the emitter's own 329 GSM8K val problems, never trained on,
                     disjoint from the 236 pf loop questions) and write tau = the 75th percentile of the margins of
                     the WRONG top-1 picks (branching catches ~3/4 of the dev mistakes). Pre-declared; never tuned on
                     the 236.
  --mode loop        the 236 pf questions: at each step every live path scores its menu (value groups, best spelling;
                     the safe pruning rules drop negative and zero results), expands top-1, or top-3 when the margin
                     < tau; the B best partial paths (summed log-prob) survive; a stop finishes a path. Reported with
                     tau = -inf (width 1: menu argmax) and the calibrated tau, by three whole-path rankers (sum
                     log-prob, mean log-prob per step, the path's weakest margin) plus the oracle (any finished path
                     right), all split seen / unseen templates.
Values are computed with `simulate` (the VM's semantics; the gold replay agrees 2988/3000). Scoring is exact: the
prompt is prefilled once, the menu's common token prefix once, and every candidate's remainder is decoded in one
batch against the copied state.

    python validation/exp_he19_beam.py --mode calibrate --export ... --adapter ... --tokenizer ...
    python validation/exp_he19_beam.py --mode loop --tau <from calibrate> --export ... --adapter ... --tokenizer ...
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from cubbyllm.reasoning.arith_world import ArithmeticWorld  # noqa: E402
from cubbyllm.reasoning.slots import Span, canon, extract, num_value  # noqa: E402
from cubbyllm.reasoning.step_loop import (PROMPT_HEAD, PROMPT_TAIL, derivation, recut, render_step,  # noqa: E402
                                          render_stop, simulate)

OPS = ("add", "sub", "mul", "div")
LOG: list[str] = []
VERBOSE = bool(os.environ.get("HE19_VERBOSE"))
D = os.path.join(ROOT, "standin", "data", "out")


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ---- scoring -------------------------------------------------------------------------------------------------
class Scorer:
    """Sum log-prob of each candidate emission (+ newline + </s>) after a prompt, exact, with shared prefixes."""

    def __init__(self, export: str, adapter: str, tokenizer: str, chunk: int, to_line: bool = False):
        from emitter import Cubby450mEmitter
        em = Cubby450mEmitter(export, adapter, tokenizer)
        em._load()
        self.model, self.tk, self.eos, self.chunk, self.to_line = em._model, em._tk, em._eos, chunk, to_line
        self.trie, self.t_enc = True, 0.0

    def encode(self, text: str) -> list[int]:
        """The candidate's tokens as training encoded them; with `to_line`, a STEP is cut after the token that ends
        its first line (`create sK ... # step K: <derivation>`), which already names the step -- the ops lines after
        it only copy it. Same tokens as the full encoding up to the cut, so the score is a prefix of the exact one."""
        ids = self.tk.encode(text.rstrip() + "\n").ids
        if self.to_line and text.lstrip().startswith("create"):
            for j in range(len(ids)):
                if "\n" in self.tk.decode(ids[:j + 1]):
                    return ids[:j + 1]
        return ids + [self.eos]

    def _step(self, st, tokens: np.ndarray):
        import grilly
        return self.model(grilly.from_numpy(np.ascontiguousarray(tokens)), past_key_values=st).logits

    def score(self, prompt: str, texts: list[str]) -> np.ndarray:
        return self.score_trie(prompt, texts) if self.trie else self.score_flat(prompt, texts)

    def score_trie(self, prompt: str, texts: list[str]) -> np.ndarray:
        """The same sums, decoded over the candidates' token TRIE: every shared prefix once, a level per call, rows =
        the trie nodes that still have children (chunked; a chunk recurses depth-first). Only the log-probs of the
        tokens that exist below each node leave the GPU (a padded gather)."""
        import grilly
        from grilly.nn import functional as GF
        t0 = time.perf_counter()
        p = self.tk.encode(prompt).ids
        cs = [self.encode(t) for t in texts]
        self.t_enc = time.perf_counter() - t0
        out = np.full(len(cs), -np.inf)
        root: dict = {"kids": {}, "ends": []}
        for i, c in enumerate(cs):
            node = root
            for tok in c:
                node = node["kids"].setdefault(tok, {"kids": {}, "ends": []})
            node["ends"].append(i)

        def expand(state, nodes, lps, ls):
            """`nodes[r]` has consumed its prefix in row r of `state`; `ls` holds row r's next-token log-softmax."""
            toks = [list(n["kids"]) for n in nodes]
            width = max(len(t) for t in toks)
            idx = np.zeros((len(nodes), width), dtype=np.int64)
            for r, t in enumerate(toks):
                idx[r, :len(t)] = t
            g = ls.gather(-1, grilly.from_numpy(idx)).numpy()
            nxt = []                                      # (parent row, token, child node, child log-prob)
            for r, n in enumerate(nodes):
                for j, tok in enumerate(toks[r]):
                    child, lp = n["kids"][tok], lps[r] + float(g[r, j])
                    for e in child["ends"]:
                        out[e] = lp
                    if child["kids"]:
                        nxt.append((r, tok, child, lp))
            if not nxt:
                return
            if [x[0] for x in nxt] == list(range(len(nodes))):
                # every row continues with exactly one token (a chain): advance THIS state in place, no copy --
                # the parent rows are never needed again
                logits = self._step(state, np.array([[x[1]] for x in nxt], dtype=np.int64))
                ls2 = GF.log_softmax(logits[:, -1], -1)
                del logits
                expand(state, [x[2] for x in nxt], [x[3] for x in nxt], ls2)
                return
            for s in range(0, len(nxt), self.chunk):
                part = nxt[s:s + self.chunk]
                st = copy.copy(state)
                st.live = state.live.clone()
                st.select_rows(grilly.from_numpy(np.array([x[0] for x in part], dtype=np.int64)))
                logits = self._step(st, np.array([[x[1]] for x in part], dtype=np.int64))
                ls2 = GF.log_softmax(logits[:, -1], -1)
                del logits
                expand(st, [x[2] for x in part], [x[3] for x in part], ls2)
                del st, ls2

        with grilly.no_grad():
            base = self.model.make_cache(1, 0)
            self.model(grilly.tensor([p[:-1]]), past_key_values=base, logits_to_keep=1)
            logits = self._step(base, np.array([[p[-1]]], dtype=np.int64))
            expand(base, [root], [0.0], GF.log_softmax(logits[:, -1], -1))
            del base, logits
        return out

    def score_flat(self, prompt: str, texts: list[str]) -> np.ndarray:
        import grilly
        from grilly.nn import functional as GF
        t0 = time.perf_counter()
        p = self.tk.encode(prompt).ids
        cs = [self.encode(t) for t in texts]
        self.t_enc = time.perf_counter() - t0
        out = np.zeros(len(cs))
        groups = defaultdict(list)                       # candidates grouped by their first token
        for i, c in enumerate(cs):
            groups[c[0]].append(i)
        with grilly.no_grad():
            base = self.model.make_cache(1, 0)
            self.model(grilly.tensor([p[:-1]]), past_key_values=base, logits_to_keep=1)
            for _, members in groups.items():
                seqs = [cs[i] for i in members]
                m = 0                                     # longest common token prefix, leaving >= 1 token each
                while all(len(s) > m + 1 for s in seqs) and all(s[m] == seqs[0][m] for s in seqs):
                    m += 1
                st = copy.copy(base)
                st.live = base.live.clone()
                prev, lp_pre = p[-1], 0.0
                for t in range(m):                        # the shared prefix: one row, one token per call
                    logits = self._step(st, np.array([[prev]], dtype=np.int64))
                    lp_pre += float(GF.log_softmax(logits[:, -1], -1).numpy()[0, seqs[0][t]])
                    prev = seqs[0][t]
                for s in range(0, len(members), self.chunk):
                    part = [seqs[j][m:] for j in range(s, min(len(members), s + self.chunk))]
                    L = max(len(c) for c in part)
                    x = np.zeros((len(part), L), dtype=np.int64)
                    y = np.zeros((len(part), L), dtype=np.int64)
                    for i, c in enumerate(part):
                        x[i, :len(c)] = [prev] + c[:-1]
                        y[i, :len(c)] = c
                    sc = copy.copy(st)
                    sc.live = st.live.clone()
                    sc.select_rows(grilly.from_numpy(np.zeros(len(part), dtype=np.int64)))
                    lp = np.zeros((len(part), L))
                    for t in range(L):
                        logits = self._step(sc, x[:, t:t + 1])
                        g = GF.log_softmax(logits[:, -1], -1).gather(-1, grilly.from_numpy(np.ascontiguousarray(y[:, t:t + 1])))
                        lp[:, t] = g.numpy()[:, 0]
                    for i, c in enumerate(part):
                        out[members[s + i]] = lp_pre + float(lp[i, :len(c)].astype(np.float64).sum())
                    del sc
                del st
            del base
        return out


# ---- the menu ------------------------------------------------------------------------------------------------
def held_values(spans) -> dict:
    vals = {}
    for s in spans:
        kind = s.kind if hasattr(s, "kind") else s.get("kind")
        sid = s.id if hasattr(s, "id") else s.get("id")
        if kind in ("N", "S") and sid not in vals:
            raw = s.filled if hasattr(s, "filled") else (s.get("value") if s.get("value") is not None else s.get("text"))
            v = num_value(raw)
            if v is not None:
                vals[sid] = Fraction(str(v))
    return vals


def menu(vals: dict, k: int, prune: bool = True) -> list[dict]:
    """Every next step on the held values (value groups keep all spellings) plus a stop per $S."""
    out = []
    for op in OPS:
        for a in vals:
            for b in vals:
                ops = [("assign", a), (op, b)]
                v = simulate(ops, vals)
                if v is None:
                    continue
                if prune and (v < 0 or v == 0):           # the two rules that pass on train (H-E19 pruning read)
                    continue
                out.append({"kind": "step", "ops": ops, "value": v, "key": ("v", v), "text": render_step(k, ops)})
    for sid in [s for s in vals if s.startswith("$S")]:
        out.append({"kind": "stop", "sid": sid, "key": ("stop", sid), "text": render_stop(sid)})
    return out


def groups_of(cands: list[dict], scores: np.ndarray) -> list[dict]:
    best = {}
    for c, s in zip(cands, scores):
        g = best.get(c["key"])
        if g is None or s > g["score"]:
            best[c["key"]] = {**c, "score": float(s)}
    return sorted(best.values(), key=lambda g: -g["score"])


# ---- calibrate -----------------------------------------------------------------------------------------------
def calibrate(a, sc: Scorer) -> None:
    recs = [json.loads(l) for l in open(os.path.join(D, "emitter_sft_v12e_w_slots.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if r.get("task") == "arithmetic" and r.get("split") == "val"]
    if a.limit:
        recs = recs[:a.limit]
    rows = [row for r in recs for row in recut(r)]
    log(f"calibrate | dev = the emitter's own GSM8K val split: {len(recs)} problems -> {len(rows)} step rows")
    res, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        k = row["stats"]["k"]
        vals = held_values(row["spans"])
        cands = menu(vals, k, prune=a.prune)
        g = groups_of(cands, sc.score(row["prompt"], [c["text"] for c in cands]))
        is_stop = row["program"].lstrip().startswith("return")
        gold_v = num_value(row["gold"])
        right = (lambda x: x["kind"] == "stop" and x["sid"] == row["program"].split()[1].rstrip(";")) if is_stop else \
            (lambda x: x["kind"] == "step" and abs(float(x["value"]) - gold_v) < 1e-6 * max(1, abs(gold_v)))
        rank = next((j for j, x in enumerate(g) if right(x)), None)
        res.append({"id": row["id"], "k": k, "stop": is_stop, "rank": rank,
                    "margin": g[0]["score"] - g[1]["score"] if len(g) > 1 else 99.0})
        if i % 50 == 0 or i == len(rows):
            ok = [x for x in res if x["rank"] is not None]
            log(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s) top-1 {sum(x['rank'] == 0 for x in ok) / max(1, len(ok)):.3f} "
                f"top-3 {sum(x['rank'] < 3 for x in ok) / max(1, len(ok)):.3f}")
    ok = [x for x in res if x["rank"] is not None]
    wrong = sorted(x["margin"] for x in ok if x["rank"] != 0)
    tau = float(np.percentile(wrong, 75)) if wrong else 0.0
    right_m = [x["margin"] for x in ok if x["rank"] == 0]
    log(f"\n  dev rows on the menu {len(ok)}/{len(res)} | top-1 {sum(x['rank'] == 0 for x in ok) / len(ok):.3f}  "
        f"top-3 {sum(x['rank'] < 3 for x in ok) / len(ok):.3f}")
    log(f"  tau = 75th percentile of the wrong picks' margins = {tau:.3f} | branching would fire on "
        f"{sum(m < tau for m in right_m) / max(1, len(right_m)):.2f} of the right picks and 0.75 of the wrong ones")
    out = os.path.join(HERE, "logs", f"exp_he19_beam_calibrate_{a.tag}")
    json.dump({"args": vars(a), "tau": tau, "rows": res}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"  -> {os.path.relpath(out, ROOT)}.{{log,json}}")


# ---- parity: trie == flat (exact), plan-line scoring vs whole-step scoring (top-1 agreement) -------------------
def parity(a, sc: Scorer) -> None:
    recs = [json.loads(l) for l in open(os.path.join(D, "emitter_sft_v12e_w_slots.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if r.get("task") == "arithmetic" and r.get("split") == "val"][:a.limit or 5]
    rows = [row for r in recs for row in recut(r)]
    worst, agree, n, tt, tf = 0.0, 0, 0, 0.0, 0.0
    for row in rows:
        cands = menu(held_values(row["spans"]), row["stats"]["k"])
        texts = [c["text"] for c in cands]
        sc.to_line = False
        sc.trie = False; t1 = time.perf_counter(); flat = sc.score(row["prompt"], texts); tf += time.perf_counter() - t1
        sc.trie = True; t1 = time.perf_counter(); trie = sc.score(row["prompt"], texts); tt += time.perf_counter() - t1
        worst = max(worst, float(np.max(np.abs(flat - trie))))
        sc.to_line = True
        line = sc.score(row["prompt"], texts)
        agree += groups_of(cands, flat)[0]["key"] == groups_of(cands, line)[0]["key"]
        n += 1
    sc.to_line = bool(a.to_line)
    log(f"parity | {n} dev step rows | trie vs flat: max |diff| {worst:.2e} | time flat {tf:.0f}s, trie {tt:.0f}s | "
        f"plan-line top-1 = whole-step top-1 on {agree}/{n}")


# ---- the loop ------------------------------------------------------------------------------------------------
def train_templates() -> set:
    pat, out = re.compile(r"program (GSM\d+)"), set()
    for name in ("emitter_sft_v12e_w_slots.jsonl", "emitter_sft_v12e_w_tg30_step.jsonl"):
        with open(os.path.join(D, name), encoding="utf-8") as f:
            for line in f:
                if '"split": "train"' in line:
                    out.update(pat.findall(line))
    return out


def solve(q: str, sc: Scorer, tau: float, beam: int, max_steps: int, world) -> dict:
    table = extract(q, constants=world.constants(q))
    paths = [{"table": table, "steps": [], "lps": [], "margins": []}]
    finished, n_scored = [], 0
    for _ in range(max_steps + 1):
        children = []
        for p in paths:
            k = len(p["steps"])
            vals = held_values(p["table"].spans)
            cands = menu(vals, k)
            if not cands:
                continue
            prompt = PROMPT_HEAD + p["table"].annotate().strip() + PROMPT_TAIL
            t1 = time.perf_counter()
            g = groups_of(cands, sc.score(prompt, [c["text"] for c in cands]))
            n_scored += 1
            if VERBOSE:
                print(f"    depth {k} menu {len(cands)} ({len(g)} groups) {time.perf_counter() - t1:.1f}s "
                      f"[encode {sc.t_enc:.1f}s] top {g[0]['text'].splitlines()[0].strip()[:60]!r}", flush=True)
            margin = g[0]["score"] - g[1]["score"] if len(g) > 1 else 99.0
            for x in g[:1 if margin >= tau else 3]:
                lps, margins = p["lps"] + [x["score"]], p["margins"] + [margin]
                if x["kind"] == "stop":
                    s = p["table"].get(x["sid"])
                    finished.append({"answer": s.filled if s else None, "steps": p["steps"], "lps": lps, "margins": margins})
                    continue
                t2 = copy.deepcopy(p["table"])
                sid = f"$S{k + 1}"
                t2.spans.append(Span(sid, derivation(x["ops"]).replace("$", ""), -1, -1, "S", canon(float(x["value"]))))
                children.append({"table": t2, "steps": p["steps"] + [(x["ops"], canon(float(x["value"])))],
                                 "lps": lps, "margins": margins})
        paths = sorted(children, key=lambda c: -sum(c["lps"]))[:beam]
        if not paths:
            break
    return {"finished": finished, "scored": n_scored}


def pick(finished: list, how: str):
    if not finished:
        return None
    key = {"sum": lambda f: sum(f["lps"]), "mean": lambda f: sum(f["lps"]) / len(f["lps"]),
           "min-margin": lambda f: (min(f["margins"]), sum(f["lps"]))}[how]
    return max(finished, key=key)


def loop(a, sc: Scorer) -> None:
    from build_emitter_sft import gold_matches
    recs = [json.loads(l) for l in open(os.path.join(D, "pf_heldout_eval_w_slots.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if r.get("task") == "arithmetic" and r.get("split", "val") == "val" and "#" not in r["id"]]
    if a.limit:
        recs = recs[:a.limit]
    pat = re.compile(r"program (GSM\d+)")
    seen_t = train_templates()
    seen = {r["id"]: bool(set(pat.findall((r.get("reference") or "") + (r.get("program") or ""))) & seen_t) for r in recs}
    world = ArithmeticWorld()
    taus = [float(x) for x in a.taus.split(",")]
    if a.tau_from:                                       # the calibrated tau, read from the dev run (never set by hand)
        taus.append(float(json.load(open(a.tau_from, encoding="utf-8"))["tau"]))
    rankers = ("sum", "mean", "min-margin")
    summary = {}
    for tau in taus:
        label = "width 1 (menu argmax)" if tau == -math.inf else f"beam {a.beam}, branch top-3 when margin < {tau:.3f}"
        log(f"\nloop | {label} | {len(recs)} pf questions ({sum(seen.values())} on seen templates)")
        res, t0 = [], time.time()
        # every finished question is appended here at once, so a killed run resumes where it stopped
        part = os.path.join(HERE, "logs", f"exp_he19_beam_loop_{a.tag}_tau{'inf' if tau == -math.inf else f'{tau:.3f}'}.jsonl")
        done = {}
        if os.path.exists(part):
            for line in open(part, encoding="utf-8"):
                if line.strip():
                    x = json.loads(line)
                    done[x["id"]] = x
            log(f"  resuming: {len(done)} questions already in {os.path.basename(part)}")
        for i, r in enumerate(recs, 1):
            if r["id"] in done:
                res.append(done[r["id"]])
                continue
            out = solve(r["question"], sc, tau, a.beam, a.max_steps, world)
            fin = out["finished"]
            row = {"id": r["id"], "seen": seen[r["id"]], "n_finished": len(fin), "scored": out["scored"],
                   "oracle": any(f["answer"] is not None and bool(gold_matches(f["answer"], r.get("gold"))) for f in fin)}
            for how in rankers:
                f = pick(fin, how)
                row[how] = f is not None and f["answer"] is not None and bool(gold_matches(f["answer"], r.get("gold")))
                row[how + "_answer"] = f["answer"] if f else None
            res.append(row)
            with open(part, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if i % 20 == 0 or i == len(recs):
                log(f"  {i}/{len(recs)} ({time.time() - t0:.0f}s) " + "  ".join(
                    f"{h} {sum(x[h] for x in res) / len(res):.3f}" for h in rankers + ("oracle",)) +
                    f"  menus scored/question {sum(x['scored'] for x in res) / len(res):.1f}")
        split = lambda h, s: (lambda rr: sum(x[h] for x in rr) / max(1, len(rr)))([x for x in res if x["seen"] == s])
        log(f"  {'ranker':<12} {'all':>6} {'seen':>6} {'unseen':>7}")
        for h in rankers + ("oracle",):
            log(f"  {h:<12} {sum(x[h] for x in res) / len(res):>6.3f} {split(h, True):>6.3f} {split(h, False):>7.3f}")
        summary[str(tau)] = {"rows": res}
    out = os.path.join(HERE, "logs", f"exp_he19_beam_loop_{a.tag}")
    json.dump({"args": vars(a), "runs": summary}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    log(f"  -> {os.path.relpath(out, ROOT)}.{{log,json}}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("calibrate", "loop", "parity"))
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--chunk", type=int, default=160, help="menu candidates decoded together against one copied state")
    ap.add_argument("--taus", default="-inf", help="loop: comma list; -inf = width 1")
    ap.add_argument("--tau-from", default="", help="loop: also run with the tau of this calibrate json")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--prune", type=int, default=1)
    ap.add_argument("--to-line", type=int, default=0, help="score a step only through its plan line (faster)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args(argv)
    sc = Scorer(a.export, a.adapter, a.tokenizer, a.chunk, to_line=bool(a.to_line))
    log(f"he19 beam | {os.path.basename(os.path.normpath(a.adapter))} | mode {a.mode} | chunk {a.chunk} | "
        f"steps scored {'to the end of the plan line' if a.to_line else 'in full'}")
    {"calibrate": calibrate, "loop": loop, "parity": parity}[a.mode](a, sc)


if __name__ == "__main__":
    main()
