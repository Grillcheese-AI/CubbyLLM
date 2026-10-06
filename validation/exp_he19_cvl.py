"""H-E19, the maths world's scorer: rank the beam's paths by P(on track) of each step's VM-computed future.

The beam over the menu holds the right answer among its finished paths for 0.496 of the 236 (0.450 of the dev
problems) while the best ranker built from the adapter's own scores picks it for 0.309 (0.250): choosing between
paths is the bottleneck, and the adapter cannot judge its own text. In MoWM's terms (cubemind `HYLA` / `cvl.py`):
a state s_t (the question + the values held so far), an action a_t (one op on two held values, or a stop), a future
s' -- which in arithmetic the VM computes EXACTLY, so no transition model is learned -- and a value Q(s, a, s')
that scores the future. Here Q is the probability the future is ON TRACK: the step's value is one the gold program
still has to compute (a stop: the held value is the gold answer). Labels come from the VM's values on TRAIN
problems only; no model is involved in the data.

Encoding (block codes, D = 2048 of the fastword table's 10,240: the sign of its teacher part, a SimHash):
  word w         f(w) in {-1, +1}^D (unknown words: a seeded random code)
  a number $N    c = sum_o (1 + |o|)^-1/2 rho^o f(w_o) over the words around it (o = -6..+4, order kept by rotation)
  a value $S     c = OP[op] * (rho^1 c(a) + rho^2 c(b)), folded left over its derivation (`S1 * N3`)
  the question   q = the bundle of the words of its last sentence (what is asked)
  an action      (op, a, b): the model reads c(a), c(b) through op-specific maps, plus the future's numbers
                 (integer? magnitude? a value already held? which operands were used before? how many $N unused?)
Model: h = relu(A[op] c(a) + B[op] c(b) + W q + F f + b) -> logit (a stop: its own maps); trained with binary
cross-entropy on every menu entry plus a listwise term (InfoNCE over the menu, the on-track futures positive -- the
contrastive value rule of cubemind's CVL, with the VM supplying the future instead of a sampled one).
Off-track states are in the data too (a wrong step taken from a gold state), so a path that went wrong scores low.

Generalisation: trained on GSM8K-train / TinyGSM step rows; early stopping on the emitter's val problems 200:329;
the path weight lambda chosen on the dev paths (val 80:200, `exp_he19_beam.py --dev 80:200 --dump-paths`); the 236
are read once after, split seen / unseen templates. Val 0:80 (tau's calibration) is never used.

    python validation/exp_he19_cvl.py --mode words                 # cache the word codes (reads the 1.1 GB table)
    python validation/exp_he19_cvl.py --mode train --tag v1        # CPU, minutes
    python validation/exp_he19_cvl.py --mode rerank --tag v1 --paths validation/logs/exp_he19_beam_loop_dev_paths_tau1.405.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

D_DATA = os.path.join(ROOT, "standin", "data", "out")
MODELS = os.environ.get("CUBBY_MODELS", r"C:\CUBBY-TRAINED-MODELS")
TABLE = os.path.join(MODELS, "fastword_table_v4.npz")
WORDS = os.path.join(MODELS, "he19_cvl_words_v4.npz")
STEP_FILE = os.path.join(D_DATA, "emitter_sft_v12e_w_tg30_step.jsonl")
SLOTS_FILE = os.path.join(D_DATA, "emitter_sft_v12e_w_slots.jsonl")
DIM = 2048
OPS = ("add", "sub", "mul", "div")
SYM = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
WORD_RX = re.compile(r"[A-Za-z][A-Za-z']*|\d[\d,]*(?:\.\d+)?")
DERIV_RX = re.compile(r"([NSK]\d+)|([+\-*/])")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ---- word codes ------------------------------------------------------------------------------------------------
def words_of(text: str) -> list[str]:
    return [w.lower() for w in WORD_RX.findall(text)]


def forms(w: str) -> list[str]:
    """The word, then its possessive stripped, then singular guesses -- the table is short on everyday plurals
    (pencils, marbles, candies) that a word problem is made of."""
    out = [w]
    if w.endswith("'s"):
        w = w[:-2]
        out.append(w)
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")
    if w.endswith("ves") and len(w) > 4:
        out += [w[:-3] + "f", w[:-3] + "fe"]
    if w.endswith("es") and len(w) > 3:
        out.append(w[:-2])
    if w.endswith("s") and len(w) > 3:
        out.append(w[:-1])
    return out


def all_texts() -> list[str]:
    """Every question the scorer will ever encode (train, val, the 236) -- for the code cache only: the codes come
    from the fixed table, nothing is fitted to these texts."""
    out = []
    for p in (STEP_FILE, SLOTS_FILE, os.path.join(D_DATA, "pf_heldout_eval_w_slots.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("task") == "arithmetic" and r.get("question"):
                    out.append(r["question"])
    return out


def build_words(a) -> None:
    from exp_hm1_ngram_hv import Table
    vocab = sorted({w for t in all_texts() for w in words_of(t) if not w[0].isdigit()})
    log(f"words | {len(vocab)} distinct words in the arithmetic questions | table {os.path.basename(a.table)}")
    tab = Table(a.table)
    hit = {w: next((x for x in forms(w) if x in tab.index), None) for w in vocab}
    have = [w for w in vocab if hit[w] is not None]
    ids = np.array([tab.index[hit[w]] for w in have])
    codes = np.empty((len(have), DIM), dtype=np.int8)
    for s in range(0, len(ids), 2048):
        P = tab.proj(ids[s:s + 2048]).reshape(len(ids[s:s + 2048]), -1)[:, :DIM]
        codes[s:s + len(P)] = np.where(P > 0, 1, -1).astype(np.int8)
    np.savez_compressed(WORDS, words=np.array(have), codes=codes, dim=DIM, table=os.path.basename(a.table))
    log(f"  {len(have)} in the table, {len(vocab) - len(have)} not (seeded random codes at use) -> {WORDS}")


class Codes:
    """Word codes, value contexts, actions -- all {-1,+1}^D (bundles real-valued, then signed only where stated)."""

    def __init__(self, path: str = WORDS):
        z = np.load(path)
        self.index = {str(w): i for i, w in enumerate(z["words"])}
        self.codes = z["codes"].astype(np.float32)
        self.cache: dict = {}
        rng = np.random.default_rng(0xC0DE)
        self.op = {o: rng.choice([-1.0, 1.0], size=DIM).astype(np.float32) for o in OPS}
        self.num = rng.choice([-1.0, 1.0], size=DIM).astype(np.float32)          # any number token

    def word(self, w: str) -> np.ndarray:
        if w[0].isdigit():
            return self.num
        i = self.index.get(w)
        if i is not None:
            return self.codes[i]
        c = self.cache.get(w)
        if c is None:
            seed = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
            c = self.cache[w] = np.random.default_rng(seed).choice([-1.0, 1.0], size=DIM).astype(np.float32)
        return c


def unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


# ---- states, the menu, features ---------------------------------------------------------------------------------
N_FEAT = 25


def frac(x) -> Fraction | None:
    from cubbyllm.reasoning.slots import num_value
    v = num_value(str(x)) if x is not None else None
    return Fraction(str(v)) if v is not None else None


def span_dict(s) -> dict:
    get = (lambda k: getattr(s, k)) if not isinstance(s, dict) else (lambda k: s.get(k))
    filled = get("value") if get("value") is not None else get("text")
    return {"id": get("id"), "kind": get("kind"), "text": get("text") or "", "start": get("start"),
            "end": get("end"), "value": frac(filled)}


class State:
    """The question and the spans held (N from the question, S the VM's values with their derivations)."""

    def __init__(self, question: str, spans: list):
        self.question = question
        self.spans = [span_dict(s) for s in spans]
        self.by_id = {s["id"]: s for s in self.spans}

    def held(self) -> dict:
        from exp_he19_beam import held_values
        from cubbyllm.reasoning.slots import canon
        return held_values([{"id": s["id"], "kind": s["kind"], "text": s["text"],       # the beam's own spelling
                             "value": None if s["value"] is None else canon(float(s["value"]))} for s in self.spans])

    def step(self, ops: list, value: Fraction) -> "State":
        from cubbyllm.reasoning.step_loop import derivation
        k = sum(1 for s in self.spans if s["kind"] == "S")
        new = {"id": f"$S{k + 1}", "kind": "S", "text": derivation(ops).replace("$", ""), "start": -1, "end": -1,
               "value": Fraction(value)}
        st = State.__new__(State)
        st.question, st.spans = self.question, self.spans + [new]
        st.by_id = {**self.by_id, new["id"]: new}
        return st


def contexts(st: State, cd: Codes) -> tuple[dict, np.ndarray]:
    """({id: context code}, question code). $N: the words around it, order kept by rotation; $S: OP-bound fold of
    its operands; $K: the bag of its text."""
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in WORD_RX.finditer(st.question)]
    ctx: dict = {}
    for s in st.spans:
        if s["kind"] == "N" and s["start"] is not None and s["start"] >= 0:
            at = next((i for i, (_, a0, a1) in enumerate(toks) if a1 > s["start"]), len(toks) - 1)
            c = np.zeros(DIM, dtype=np.float32)
            for o in range(-6, 5):
                j = at + o
                if o == 0 or not (0 <= j < len(toks)) or (s["start"] <= toks[j][1] < s["end"]):
                    continue
                c += np.roll(cd.word(toks[j][0]), o) / math.sqrt(1 + abs(o))
            for w in words_of(s["text"]):                  # a number written in words carries meaning: half, twice
                if not w[0].isdigit():
                    c += cd.word(w)
            ctx[s["id"]] = unit(c)
        elif s["kind"] in ("N", "K"):
            ctx[s["id"]] = unit(sum((cd.word(w) for w in words_of(s["text"])), np.zeros(DIM, dtype=np.float32)))
    for s in st.spans:                                     # S in order: each folds earlier ones
        if s["kind"] != "S":
            continue
        acc, op = None, None
        for m in DERIV_RX.finditer(s["text"]):
            if m.group(2):
                op = SYM[m.group(2)]
                continue
            c = ctx.get("$" + m.group(1), np.zeros(DIM, dtype=np.float32))
            acc = c if acc is None else unit(cd.op[op or "add"] * (np.roll(acc, 1) + np.roll(c, 2)))
        ctx[s["id"]] = acc if acc is not None else np.zeros(DIM, dtype=np.float32)
    sents = [x for x in re.split(r"(?<=[.?!])\s+", st.question.strip()) if x.strip()]
    ask = next((x for x in reversed(sents) if "?" in x), sents[-1] if sents else st.question)
    q = unit(sum((cd.word(w) for w in words_of(ask)), np.zeros(DIM, dtype=np.float32)))
    return ctx, q


def used_ids(st: State) -> set:
    return {"$" + m.group(1) for s in st.spans if s["kind"] == "S" for m in DERIV_RX.finditer(s["text"]) if m.group(1)}


def candidates(st: State, only: list | None = None) -> tuple[list, list[str], np.ndarray, np.ndarray]:
    """The beam's own menu on this state (the same pruning), each as (op index 0-3 / 4 = stop, a index, b index) and
    N_FEAT numbers about its future. `only`: score just these menu entries (rerank: the path's own step)."""
    from exp_he19_beam import menu
    vals = st.held()
    ids = list(vals)
    k = sum(1 for i in ids if i.startswith("$S"))
    cands = only if only is not None else menu(vals, k)
    pos = {i: j for j, i in enumerate(ids)}
    sids = [i for i in ids if i.startswith("$S")]
    latest = sids[-1] if sids else None
    used = used_ids(st)
    nN = sum(1 for i in ids if i.startswith("$N"))
    held_v = [float(v) for v in vals.values()]
    hmax = max(held_v) if held_v else 1.0
    idx = np.zeros((len(cands), 3), dtype=np.int64)
    f = np.zeros((len(cands), N_FEAT), dtype=np.float32)
    for r, c in enumerate(cands):
        if c["kind"] == "stop":
            v = float(vals[c["sid"]])
            idx[r] = (4, pos[c["sid"]], 0)
            f[r, 4] = 1.0
            f[r, 21] = float(c["sid"] == latest)
            unused = sum(1 for i in ids if i.startswith("$N") and i not in used)
            f[r, 18] = unused / max(1, nN)
            f[r, 22] = float(unused == 0)
        else:
            a, b = c["ops"][0][1], c["ops"][1][1]
            o = OPS.index(c["ops"][1][0])
            v = float(c["value"])
            x, y = float(vals[a]), float(vals[b])
            idx[r] = (o, pos[a], pos[b])
            f[r, o] = 1.0
            f[r, 8] = float(any(abs(v - h) <= 1e-9 * max(1, abs(h)) for h in held_v))
            f[r, 9] = float(v > hmax)
            f[r, 11], f[r, 12] = float(a.startswith("$S")), float(b.startswith("$S"))
            f[r, 13], f[r, 14] = float(a == latest), float(b == latest)
            f[r, 15] = float(a == b)
            f[r, 16], f[r, 17] = float(a in used), float(b in used)
            after = used | {a, b}
            f[r, 18] = sum(1 for i in ids if i.startswith("$N") and i not in after) / max(1, nN)
            f[r, 20] = float(x > y)
            f[r, 24] = max(-1.0, min(1.0, math.log10(max(1e-9, abs(v)) / max(1e-9, abs(x), abs(y))) / 3))
        f[r, 5] = float(abs(v - round(v)) < 1e-9)
        f[r, 6] = math.log10(abs(v) + 1) / 4
        f[r, 7] = float(abs(v * 100 - round(v * 100)) < 1e-6)
        f[r, 10] = float(abs(v) < 1)
        f[r, 19] = k / 8
        f[r, 23] = float(abs(v - round(v)) < 1e-9 and round(v) % 5 == 0)
    return cands, ids, idx, f


def on_track(st: State, cands: list, gold_vals: list, final: Fraction) -> np.ndarray:
    """A step is on track when its value is one the gold program still has to compute; a stop when it returns the
    gold answer. Value-based, so a path that recovers from a wasted step is on track again."""
    near = lambda x, y: abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y)))
    held = [s["value"] for s in st.spans if s["kind"] == "S" and s["value"] is not None]
    remaining = [g for g in gold_vals if not any(near(h, g) for h in held)]
    vals = st.held()
    out = np.zeros(len(cands), dtype=np.float32)
    for r, c in enumerate(cands):
        if c["kind"] == "stop":
            out[r] = float(near(vals[c["sid"]], final))
        else:
            out[r] = float(any(near(c["value"], g) for g in remaining))
    return out


# ---- data: gold-path states (and one off-track state per problem) from the step rows ----------------------------
def problems(split: str, keep: set | None = None) -> list[dict]:
    """{question, states: [spans of each gold step row], gold_vals, final} per problem of the step file's split."""
    by = defaultdict(list)
    with open(STEP_FILE, encoding="utf-8") as f:
        for line in f:
            if f'"split": "{split}"' not in line:
                continue
            r = json.loads(line)
            if r.get("task") != "arithmetic" or "#" not in r.get("id", ""):
                continue
            base = r["id"].split("#")[0]
            if keep is None or base in keep:
                by[base].append(r)
    out = []
    for base, rows in by.items():
        rows.sort(key=lambda r: r["stats"]["k"])
        stop = rows[-1]
        if not stop["program"].lstrip().startswith("return"):
            continue
        sid = stop["program"].split()[1].rstrip(";")
        svals = {s["id"]: frac(s.get("value") or s.get("text")) for s in stop["spans"] if s["kind"] == "S"}
        if sid not in svals or svals[sid] is None:
            continue
        out.append({"id": base, "question": stop["question"], "states": [r["spans"] for r in rows],
                    "gold_vals": [v for v in svals.values() if v is not None], "final": svals[sid]})
    return out


def val_split_ids() -> list[str]:
    """The emitter's GSM8K val problems in the order the beam's --dev slices them."""
    out = []
    with open(SLOTS_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") == "arithmetic" and r.get("split") == "val":
                out.append(r["id"])
    return out


def encode_problem(p: dict, cd: Codes, rng: random.Random, corrupt: bool) -> list[dict]:
    """Every gold state of a problem as arrays; plus, if `corrupt`, one off-track state (a wrong step taken)."""
    states = [State(p["question"], sp) for sp in p["states"]]
    if corrupt:
        st = rng.choice(states)
        cands, _, _, _ = candidates(st)
        y = on_track(st, cands, p["gold_vals"], p["final"])
        wrong = [c for c, t in zip(cands, y) if c["kind"] == "step" and t == 0]
        if wrong:
            c = rng.choice(wrong)
            states.append(st.step(c["ops"], c["value"]))
    out = []
    for st in states:
        cands, ids, idx, f = candidates(st)
        if not cands:
            continue
        ctx, q = contexts(st, cd)
        C = np.stack([ctx.get(i, np.zeros(DIM, dtype=np.float32)) for i in ids]).astype(np.float16)
        out.append({"C": C, "q": q.astype(np.float16), "idx": idx.astype(np.int16), "f": f.astype(np.float16),
                    "y": on_track(st, cands, p["gold_vals"], p["final"]), "pid": p["id"]})
    return out


# ---- the model ---------------------------------------------------------------------------------------------------
def make_model(hidden: int = 128):
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        """logit(on track | state, action, VM future): op-specific reads of the operand contexts, the question's ask,
        the future's numbers."""

        def __init__(self):
            super().__init__()
            H = self.H = hidden
            self.A = nn.Linear(DIM, 5 * H, bias=False)     # operand a under add/sub/mul/div, or the stop's value
            self.B = nn.Linear(DIM, 4 * H, bias=False)     # operand b under add/sub/mul/div
            self.Q = nn.Linear(DIM, 2 * H)                 # the ask, for a step / for a stop
            self.F = nn.Linear(N_FEAT, H)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))

        def forward(self, C, q, op, ia, ib, f):
            Bsz, n, _ = C.shape
            H = self.H
            A = self.A(C).view(Bsz, n * 5, H)
            Bm = self.B(C).view(Bsz, n * 4, H)
            bi = torch.arange(Bsz)[:, None]
            stop = op == 4
            a_part = A[bi, ia * 5 + op]
            b_part = Bm[bi, ib * 4 + op.clamp(max=3)] * (~stop).unsqueeze(-1)
            qh = self.Q(q).view(Bsz, 2, H)
            q_part = qh[bi, stop.long()]
            return self.head(a_part + b_part + q_part + self.F(f)).squeeze(-1)

    return QNet()


def batch_tensors(items: list[dict]):
    import torch
    n = max(len(x["C"]) for x in items)
    M = max(len(x["idx"]) for x in items)
    B = len(items)
    C = np.zeros((B, n, DIM), dtype=np.float32)
    q = np.zeros((B, DIM), dtype=np.float32)
    idx = np.zeros((B, M, 3), dtype=np.int64)
    f = np.zeros((B, M, N_FEAT), dtype=np.float32)
    y = np.zeros((B, M), dtype=np.float32)
    mask = np.zeros((B, M), dtype=bool)
    for i, x in enumerate(items):
        C[i, :len(x["C"])] = x["C"]
        q[i] = x["q"]
        m = len(x["idx"])
        idx[i, :m], f[i, :m], y[i, :m], mask[i, :m] = x["idx"], x["f"], x["y"], True
    t = lambda a: torch.from_numpy(a)
    return t(C), t(q), t(idx[..., 0]), t(idx[..., 1]), t(idx[..., 2]), t(f), t(y), t(mask)


def loss_of(model, items, pos_weight: float):
    import torch
    import torch.nn.functional as F
    C, q, op, ia, ib, f, y, mask = batch_tensors(items)
    logit = model(C, q, op, ia, ib, f)
    bce = F.binary_cross_entropy_with_logits(logit[mask], y[mask], pos_weight=torch.tensor(pos_weight))
    neg_inf = torch.finfo(logit.dtype).min
    has = (y * mask).sum(1) > 0
    all_lse = torch.logsumexp(logit.masked_fill(~mask, neg_inf), 1)
    pos_lse = torch.logsumexp(logit.masked_fill(~(mask & (y > 0)), neg_inf), 1)
    lst = (all_lse - pos_lse)[has].mean() if has.any() else torch.zeros(())
    return bce + lst, logit.detach(), y, mask


def read(model, data: list[dict], per: int = 64) -> dict:
    """On-track states: is the top-scored entry on track (top-1), and AUC of on-track vs not over every entry."""
    import torch
    top, n_on, scores, labels = 0, 0, [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(data), per):
            part = data[s:s + per]
            _, logit, y, mask = loss_of(model, part, 1.0)
            for i in range(len(part)):
                m = mask[i]
                lg, yy = logit[i][m].numpy(), y[i][m].numpy()
                scores.append(lg)
                labels.append(yy)
                if yy.sum() > 0:
                    n_on += 1
                    top += int(yy[int(np.argmax(lg))] > 0)
    model.train()
    s, l = np.concatenate(scores), np.concatenate(labels)
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = l.sum(), len(l) - l.sum()
    auc = float((ranks[l > 0].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))
    return {"top1": top / max(1, n_on), "auc": auc, "states": len(data), "on_track_states": n_on}


def train(a) -> None:
    import torch
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    cd = Codes()
    t0 = time.time()
    tr = problems("train")
    rng.shuffle(tr)
    tr = tr[:a.problems]
    vids = val_split_ids()
    stop_ids = set(vids[200:])                            # early stopping: val 200:329 (dev 80:200 is lambda's)
    va = problems("val", keep=stop_ids)
    log(f"cvl train | {len(tr)} train problems, {len(va)} val problems (200:329) | encoding ...")
    data = [x for p in tr for x in encode_problem(p, cd, rng, corrupt=True)]
    vdata = [x for p in va for x in encode_problem(p, cd, random.Random(1), corrupt=True)]
    pos = sum(float(x["y"].sum()) for x in data)
    tot = sum(len(x["y"]) for x in data)
    pw = min(30.0, (tot - pos) / max(1.0, pos))
    log(f"  {len(data)} train states ({tot} menu entries, {pos / tot:.3f} on track; pos_weight {pw:.1f}), "
        f"{len(vdata)} val states | {time.time() - t0:.0f}s")
    model = make_model(a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_ep, bad = -1.0, 0, 0
    path = os.path.join(MODELS, f"he19_cvl_{a.tag}.pt")
    log(f"  epoch 0: val {read(model, vdata)}")
    for ep in range(1, a.epochs + 1):
        rng.shuffle(data)
        run, nb = 0.0, 0
        for s in range(0, len(data), a.batch):
            loss, _, _, _ = loss_of(model, data[s:s + a.batch], pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        r = read(model, vdata)
        log(f"  epoch {ep}: train loss {run / nb:.4f} | val top-1 {r['top1']:.3f} auc {r['auc']:.3f} | {time.time() - t0:.0f}s")
        score = r["top1"] + r["auc"]
        if score > best:
            best, best_ep, bad = score, ep, 0
            torch.save({"state": model.state_dict(), "hidden": a.hidden, "dim": DIM, "epoch": ep, "val": r}, path)
        else:
            bad += 1
            if bad >= a.patience:
                break
    log(f"  best epoch {best_ep} -> {path}")
    out = os.path.join(HERE, "logs", f"exp_he19_cvl_train_{a.tag}.log")
    open(out, "w", encoding="utf-8").write("\n".join(LOG) + "\n")


def load_model(tag: str):
    import torch
    z = torch.load(os.path.join(MODELS, f"he19_cvl_{tag}.pt"), map_location="cpu")
    m = make_model(z["hidden"])
    m.load_state_dict(z["state"])
    m.eval()
    return m


# ---- rerank the beam's finished paths ----------------------------------------------------------------------------
def path_logits(model, cd: Codes, question: str, path: dict, world) -> list[float]:
    """logit(on track) of each step of a finished path and of its stop, each in the state the path had reached."""
    import torch
    from cubbyllm.reasoning.slots import extract
    st = State(question, extract(question, constants=world.constants(question)).spans)
    out = []
    for ops, v in path["steps"]:
        ops = [tuple(o) for o in ops]
        c = {"kind": "step", "ops": ops, "value": Fraction(str(v))}
        out.append((st, c))
        st = st.step(ops, Fraction(str(v)))
    vals = st.held()
    ans = frac(path["answer"])
    sid = next((i for i in reversed(list(vals)) if i.startswith("$S") and ans is not None
                and abs(float(vals[i]) - float(ans)) <= 1e-9 * max(1, abs(float(ans)))), None)
    if sid is not None:
        out.append((st, {"kind": "stop", "sid": sid, "key": ("stop", sid)}))
    logits = []
    with torch.no_grad():
        for s, c in out:
            _, ids, idx, f = candidates(s, only=[c])
            ctx, q = contexts(s, cd)
            C = np.stack([ctx.get(i, np.zeros(DIM, dtype=np.float32)) for i in ids])
            item = {"C": C, "q": q, "idx": idx, "f": f, "y": np.zeros(1, dtype=np.float32)}
            Ct, qt, op, ia, ib, ft, _, _ = batch_tensors([item])
            logits.append(float(model(Ct, qt, op, ia, ib, ft)[0, 0]))
    return logits


def logsig(x: float) -> float:
    return -math.log1p(math.exp(-x)) if x > -30 else x


RULES = ["lm_mean", "q_sum", "q_min"] + [f"mix_{l:g}" for l in (0.5, 1, 2, 4, 8)] + ["got"]


def rank_paths(paths: list[dict], rule: str):
    """The pre-declared family: the beam's mean log-prob (the 0.250 / 0.309 baseline), the scorer alone (sum or weakest
    step), the mean log-prob + lambda x the scorer's mean log-probability, and HD-GoT (cubemind `hd_got`): paths as a
    graph, edges = the share of step values two paths agree on, node prior = the scorer, ranked by 3 hops of diffusion."""
    if not paths:
        return None
    lm = lambda p: sum(p["lps"]) / len(p["lps"])
    qs = lambda p: [logsig(x) for x in p["q"]] or [0.0]
    if rule == "lm_mean":
        return max(paths, key=lm)
    if rule == "q_sum":
        return max(paths, key=lambda p: (sum(qs(p)), lm(p)))
    if rule == "q_min":
        return max(paths, key=lambda p: (min(qs(p)), lm(p)))
    if rule.startswith("mix_"):
        lam = float(rule[4:])
        return max(paths, key=lambda p: lm(p) + lam * sum(qs(p)) / len(qs(p)))
    if rule == "got":
        vs = [{str(v) for _, v in p["steps"]} | {str(p["answer"])} for p in paths]
        n = len(paths)
        A = np.array([[len(vs[i] & vs[j]) / max(1, len(vs[i] | vs[j])) if i != j else 0.0 for j in range(n)]
                      for i in range(n)])
        prior = np.exp(np.array([sum(qs(p)) for p in paths]))
        r = prior.copy()
        for _ in range(3):
            r = prior + A @ r
            r = r / max(1e-12, r.sum())
        return paths[int(np.argmax(r))]
    raise ValueError(rule)


def rerank(a) -> None:
    from cubbyllm.reasoning.arith_world import ArithmeticWorld
    model, cd, world = load_model(a.tag), Codes(), ArithmeticWorld()
    rows = [json.loads(l) for l in open(a.paths, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "paths" in r]
    t0 = time.time()
    for r in rows:
        for p in r["paths"]:
            p["q"] = path_logits(model, cd, r["question"], p, world)
    log(f"rerank | scorer {a.tag} | {os.path.basename(a.paths)}: {len(rows)} questions, "
        f"{sum(len(r['paths']) for r in rows)} finished paths | {time.time() - t0:.0f}s")
    res = {}
    for rule in RULES:
        hit = [bool(rank_paths(r["paths"], rule) and rank_paths(r["paths"], rule)["correct"]) for r in rows]
        sp = lambda s: (lambda xs: sum(xs) / max(1, len(xs)))([h for h, r in zip(hit, rows) if r["seen"] == s])
        res[rule] = {"all": sum(hit) / len(rows), "seen": sp(True), "unseen": sp(False)}
    res["oracle"] = {"all": sum(any(p["correct"] for p in r["paths"]) for r in rows) / len(rows)}
    log(f"  {'rule':<10} {'all':>6} {'seen':>6} {'unseen':>7}")
    for k, v in res.items():
        log(f"  {k:<10} {v['all']:>6.3f} {v.get('seen', float('nan')):>6.3f} {v.get('unseen', float('nan')):>7.3f}")
    if a.choose:                                          # dev only: the rule to carry to the 236, written once
        best = max(RULES, key=lambda k: (res[k]["all"], -RULES.index(k)))
        json.dump({"rule": best, "dev": res}, open(os.path.join(HERE, "logs", f"exp_he19_cvl_rule_{a.tag}.json"), "w"),
                  indent=1)
        log(f"  chosen on dev: {best} ({res[best]['all']:.3f}; ties go to the earlier, simpler rule)")
    out = os.path.join(HERE, "logs", f"exp_he19_cvl_rerank_{a.tag}_{os.path.basename(a.paths).split('.')[0]}")
    json.dump({"args": vars(a), "results": res}, open(out + ".json", "w", encoding="utf-8"), indent=1)
    open(out + ".log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("words", "train", "rerank"))
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--problems", type=int, default=6000, help="train problems (each ~5 gold states + 1 off-track)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", default="")
    ap.add_argument("--choose", action="store_true", help="rerank on the DEV paths: write the chosen rule")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (the GPU beam may be running beside it)")
    a = ap.parse_args(argv)
    if a.mode != "words":
        import torch
        torch.set_num_threads(a.threads)
    {"words": build_words, "train": train, "rerank": rerank}[a.mode](a)


if __name__ == "__main__":
    main()
