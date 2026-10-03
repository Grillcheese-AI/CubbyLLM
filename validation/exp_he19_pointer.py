"""H-E19, the next-step pointer head: pick the next step from the host's menu instead of writing it.

The ranking read (exp_he19_rank.py) found the right next step on the adapter's short list but not at its top
(later steps: top-1 0.52-0.58, top-3 0.81-0.86). This trains a small head that scores the menu directly:

  trunk (450M + the write-back adapter, frozen) reads the prompt ONCE ->
    q    = the hidden state at the last prompt token ("Program:\n")
    v_i  = the mean hidden state over each held value's marker in the prompt ([N1: 35], [S1: 9 = N1 - N2])
  head scores every candidate:  op(a, b) for op in add/sub/mul/div over held values, and stop(S_j)
  loss = -log P(the right choice) over the row's menu (candidates computing the gold value all count).

Like a next-token head over a vocabulary, except the vocabulary is the row's menu and its entries are
pointers into the prompt (a pointer network). One trunk pass per step instead of ~150 scored emissions.

  extract  -> trunk features to an .npz (local grilly2, no credits)
  train    -> the head on CPU (torch), read on held-out rows; compare with the adapter's own ranking

    python validation/exp_he19_pointer.py extract --split train --n 24000 --out <npz> --export .. --adapter .. --tokenizer ..
    python validation/exp_he19_pointer.py extract --data <held step jsonl> --split val --n 0 --out <npz> ...
    python validation/exp_he19_pointer.py train --train <npz> --held <npz>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

OPS = ("add", "sub", "mul", "div")
MARK = re.compile(r"\[(?P<id>[NS]\d+):[^\]]*\]")
OUT = os.path.join(ROOT, "standin", "data", "out")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def _near(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ── the menu, as pointers ────────────────────────────────────────────────────

def structure(row: dict):
    """(ids, menu, facts): the held values' slot ids in order, and the menu as (op, a, b, correct) with op
    0-3 = add/sub/mul/div on value indices a, b, and op 4 = stop on value index a (an $S). Same menu as
    exp_he19_rank.menu, by index instead of by text."""
    from emitter_data import table_of
    from cubbyllm.reasoning.slots import num_value
    table = table_of(row["spans"], row.get("question", ""))
    vals: dict[str, float] = {}
    for s in table.spans:
        if s.kind in ("N", "S") and s.id not in vals:
            v = num_value(s.filled)
            if v is not None:
                vals[s.id] = v
    ids = list(vals)
    gold_prog = row["program"].strip()
    is_stop = gold_prog.startswith("return")
    gold = num_value(str(row["gold"]))
    menu, groups = [], {}                         # group: what a candidate computes (a+b and b+a are one choice)
    for o, op in enumerate(OPS):
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                x, y = vals[a], vals[b]
                if op == "div" and y == 0:
                    continue
                v = (x + y, x - y, x * y, x / y if y else None)[o]
                g = groups.setdefault(("v", round(v, 9)), len(groups))
                menu.append((o, i, j, int((not is_stop) and _near(v, gold)), g))
    for i, a in enumerate(ids):
        if a.startswith("$S"):
            g = groups.setdefault(("stop", a), len(groups))
            menu.append((4, i, i, int(is_stop and gold_prog.split()[1].rstrip(";") == a), g))
    k = sum(a.startswith("$S") for a in ids)
    return ids, menu, {"k": k, "is_stop": is_stop}


def marker_tokens(prompt: str, offsets, ids: list[str]) -> list[list[int]]:
    """For each held value, the prompt tokens that spell its marker(s): [N1: 35], [S1: 9 = N1 - N2]."""
    spans: dict[str, list[tuple[int, int]]] = {}
    for m in MARK.finditer(prompt):
        spans.setdefault("$" + m.group("id"), []).append((m.start(), m.end()))
    out = []
    for sid in ids:
        toks = [t for (s, e) in spans.get(sid, []) for t, (a, b) in enumerate(offsets) if a < e and b > s]
        out.append(sorted(set(toks)))
    return out


def trunk_hidden(model, x_np: np.ndarray) -> np.ndarray:
    """The adapter-wrapped trunk's last hidden states for a right-padded batch (B, T): CubbyForCausalLM's
    forward without the head. Right padding is safe: every layer is causal."""
    import grilly
    t = x_np.shape[1]
    with grilly.no_grad():
        ids = grilly.from_numpy(x_np.astype(np.int64))
        cos, sin = model._rotary.tables(t)
        h = model.model.embed_tokens(ids)
        ctx = model.context(h, None)
        for layer in model.model.layers:
            h = layer(h, cos, sin, None)
        h = model.model.adapter(h, ctx)
        return h.numpy().astype(np.float32)


# ── extract ──────────────────────────────────────────────────────────────────

def extract(a) -> None:
    rows = []
    for line in open(a.data, encoding="utf-8"):
        r = json.loads(line)
        if r.get("task") == "arithmetic" and r.get("split") == a.split:
            rows.append(r)
    random.Random(a.seed).shuffle(rows)
    if a.n:
        rows = rows[:a.n]
    from emitter import Cubby450mEmitter
    em = Cubby450mEmitter(a.export, a.adapter, a.tokenizer)
    em._load()
    model, tk = em._model, em._tk
    log(f"extract | {len(rows)} {a.split} rows from {os.path.basename(a.data)} | {os.path.basename(os.path.normpath(a.adapter))}")
    items = []
    for r in rows:
        ids, menu, facts = structure(r)
        if not any(c[3] for c in menu):
            continue
        enc = tk.encode(r["prompt"])
        items.append((r, ids, menu, facts, enc.ids, marker_tokens(r["prompt"], enc.offsets, ids)))
    order = sorted(range(len(items)), key=lambda i: len(items[i][4]))
    d = None
    Q, V, M, MENU, META = [None] * len(items), [None] * len(items), [], [None] * len(items), [None] * len(items)
    t0, missing = time.time(), 0
    for s in range(0, len(order), a.batch):
        part = order[s:s + a.batch]
        T = max(len(items[i][4]) for i in part)
        x = np.zeros((len(part), T), dtype=np.int64)
        for b, i in enumerate(part):
            x[b, :len(items[i][4])] = items[i][4]
        H = trunk_hidden(model, x)
        d = H.shape[-1]
        for b, i in enumerate(part):
            r, ids, menu, facts, tok, marks = items[i]
            L = len(tok)
            vs = []
            for tl in marks:
                if tl:
                    vs.append(H[b, tl].mean(0))
                else:
                    missing += 1
                    vs.append(np.zeros(d, dtype=np.float32))
            Q[i] = H[b, L - 1].astype(np.float16)
            V[i] = np.stack(vs).astype(np.float16)
            MENU[i] = np.asarray(menu, dtype=np.int32)
            META[i] = {"id": r["id"], "k": facts["k"], "is_stop": facts["is_stop"],
                       "kinds": [sid[1] for sid in ids]}
        if (s // a.batch) % 50 == 0:
            log(f"  {min(s + a.batch, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
    vo = np.cumsum([0] + [len(v) for v in V])
    mo = np.cumsum([0] + [len(m) for m in MENU])
    np.savez(a.out, q=np.stack(Q), v=np.concatenate(V), v_off=vo, menu=np.concatenate(MENU), menu_off=mo,
             meta=np.asarray([json.dumps(m) for m in META]))
    log(f"  -> {a.out}: {len(Q)} rows, d={d}, {missing} values with no marker in the prompt ({time.time() - t0:.0f}s)")


# ── the head ─────────────────────────────────────────────────────────────────

def load(path: str):
    z = np.load(path)                       # an NpzFile reads an array on EVERY index: take each one once
    q, v, vo, menu, mo, metas = z["q"], z["v"], z["v_off"], z["menu"], z["menu_off"], z["meta"]
    rows = []
    for i in range(len(q)):
        rows.append({"q": q[i], "v": v[vo[i]:vo[i + 1]], "menu": menu[mo[i]:mo[i + 1]],
                     **json.loads(str(metas[i]))})
    return rows


def batch_of(rows, torch):
    B, M = len(rows), max(len(r["v"]) for r in rows)
    C = max(len(r["menu"]) for r in rows)
    d = rows[0]["q"].shape[0]
    q = torch.zeros(B, d); v = torch.zeros(B, M, d); kind = torch.zeros(B, M, dtype=torch.long)
    menu = torch.zeros(B, C, 3, dtype=torch.long); corr = torch.zeros(B, C, dtype=torch.bool)
    valid = torch.zeros(B, C, dtype=torch.bool); grp = torch.full((B, C), -1, dtype=torch.long)
    for b, r in enumerate(rows):
        q[b] = torch.from_numpy(r["q"].astype(np.float32))
        v[b, :len(r["v"])] = torch.from_numpy(r["v"].astype(np.float32))
        kind[b, :len(r["kinds"])] = torch.tensor([1 if k == "S" else 0 for k in r["kinds"]])
        m = torch.from_numpy(r["menu"].astype(np.int64))
        menu[b, :len(m)] = m[:, :3]; corr[b, :len(m)] = m[:, 3] > 0; grp[b, :len(m)] = m[:, 4]
        valid[b, :len(m)] = True
    return q, v, kind, menu, corr, valid, grp


def make_head(d: int, h: int, torch):
    nn = torch.nn

    class Head(nn.Module):
        """score(op, a, b) = MLP([e_a + op, e_b, e_a * e_b, q]); e = proj(LayerNorm(value state)) + kind."""

        def __init__(self):
            super().__init__()
            self.nv, self.nq = nn.LayerNorm(d), nn.LayerNorm(d)
            self.pv, self.pq = nn.Linear(d, h), nn.Linear(d, h)
            self.kind, self.op = nn.Embedding(2, h), nn.Embedding(5, h)
            self.mlp = nn.Sequential(nn.Linear(4 * h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

        def forward(self, q, v, kind, menu):
            e = self.pv(self.nv(v)) + self.kind(kind)                         # (B, M, h)
            qq = self.pq(self.nq(q))                                          # (B, h)
            op, ia, ib = menu.unbind(-1)                                      # (B, C)
            ea = torch.gather(e, 1, ia.unsqueeze(-1).expand(-1, -1, h))
            eb = torch.gather(e, 1, ib.unsqueeze(-1).expand(-1, -1, h))
            x = torch.cat([ea + self.op(op), eb, ea * eb, qq.unsqueeze(1).expand_as(ea)], -1)
            return self.mlp(x).squeeze(-1)                                    # (B, C)

    return Head()


def nll(scores, corr, valid, torch):
    neg = torch.finfo(scores.dtype).min / 4
    all_ = torch.logsumexp(scores.masked_fill(~valid, neg), -1)
    right = torch.logsumexp(scores.masked_fill(~(corr & valid), neg), -1)
    return (all_ - right).mean()


def ranks(scores, corr, valid, grp):
    """Per row: the rank of the right GROUP among groups, each group scored by its best candidate."""
    out = []
    s, c, va, g = scores.detach().numpy(), corr.numpy(), valid.numpy(), grp.numpy()
    for b in range(len(s)):
        best: dict[int, list] = {}
        for j in np.nonzero(va[b])[0]:
            e = best.setdefault(int(g[b, j]), [-math.inf, False])
            e[0] = max(e[0], float(s[b, j])); e[1] = e[1] or bool(c[b, j])
        order = sorted(best.values(), key=lambda e: -e[0])
        out.append(next((i for i, e in enumerate(order) if e[1]), None))
    return out


def summarize(rows, rk, label):
    n = len(rows)
    if not n:
        return f"  {label:<12} n=0"
    hit = lambda t: sum(r is not None and r < t for r in rk) / n
    return f"  {label:<12} n={n:<4} top-1 {hit(1):.3f}  top-3 {hit(3):.3f}  top-5 {hit(5):.3f}"


def read(head, rows, torch, bs=128):
    head.eval()
    rk, loss = [], 0.0
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            q, v, kind, menu, corr, valid, grp = batch_of(rows[s:s + bs], torch)
            sc = head(q, v, kind, menu)
            loss += float(nll(sc, corr, valid, torch)) * len(rows[s:s + bs])
            rk += ranks(sc, corr, valid, grp)
    return rk, loss / max(1, len(rows))


def report(rows, rk, tag):
    log(f"\n{tag}")
    log(summarize(rows, rk, "all"))
    sel = lambda f: ([r for r in rows if f(r)], [x for r, x in zip(rows, rk) if f(r)])
    log(summarize(*sel(lambda r: r["is_stop"]), "stop rows"))
    log(summarize(*sel(lambda r: not r["is_stop"]), "step rows"))
    for k in (0, 1, 2):
        log(summarize(*sel(lambda r, k=k: not r["is_stop"] and r["k"] == k), f"  step {k}"))
    log(summarize(*sel(lambda r: not r["is_stop"] and r["k"] >= 3), "  step 3+"))


def train(a) -> None:
    import torch
    torch.manual_seed(a.seed)
    rows = load(a.train)
    random.Random(a.seed).shuffle(rows)
    nv = max(1, min(len(rows) // 5, max(200, len(rows) // 20)))     # 5% for the dev split, 200+ when there is room
    dev_rows, tr = rows[:nv], rows[nv:]
    held = load(a.held)
    d = tr[0]["q"].shape[0]
    head = make_head(d, a.hidden, torch)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    log(f"pointer head | train {len(tr)} rows, dev {len(dev_rows)}, held {len(held)} | d={d} hidden={a.hidden} "
        f"| {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M params | {a.epochs} epochs")
    best, best_state, t0 = math.inf, None, time.time()
    for ep in range(a.epochs):
        head.train()
        random.shuffle(tr)
        tot = 0.0
        for s in range(0, len(tr), a.bs):
            q, v, kind, menu, corr, valid, grp = batch_of(tr[s:s + a.bs], torch)
            loss = nll(head(q, v, kind, menu), corr, valid, torch)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(tr[s:s + a.bs])
        rk, dl = read(head, dev_rows, torch)
        top1 = sum(r == 0 for r in rk) / len(rk)
        log(f"  epoch {ep + 1}: train nll {tot / len(tr):.3f} | dev nll {dl:.3f} top-1 {top1:.3f} ({time.time() - t0:.0f}s)")
        if dl < best:
            best, best_state = dl, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    rk, hl = read(head, held, torch)
    report(held, rk, f"held-out ({os.path.basename(a.held)}), head alone, nll {hl:.3f}")
    if a.rank_json and os.path.exists(a.rank_json):
        ref = {r["id"]: r for r in json.load(open(a.rank_json, encoding="utf-8"))["rows"] if r.get("covered")}
        both = [(r, x) for r, x in zip(held, rk) if r["id"] in ref]
        if both:
            rows_b = [r for r, _ in both]
            report(rows_b, [x for _, x in both], f"the same {len(both)} rows as the adapter's ranking read: head")
            report(rows_b, [ref[r["id"]]["rank"] for r in rows_b], f"the same {len(both)} rows: the adapter's own ranking")
    os.makedirs(a.save, exist_ok=True)
    torch.save({"state": head.state_dict(), "d": d, "hidden": a.hidden}, os.path.join(a.save, "pointer_head.pt"))
    json.dump({"args": vars(a), "held_ranks": rk, "held_ids": [r["id"] for r in held]},
              open(os.path.join(HERE, "logs", f"exp_he19_pointer_{a.tag}.json"), "w", encoding="utf-8"))
    log(f"  -> head {os.path.join(a.save, 'pointer_head.pt')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", default=os.path.join(OUT, "emitter_sft_v12e_w_tg30_step.jsonl"))
    e.add_argument("--split", default="train")
    e.add_argument("--n", type=int, default=24000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--out", required=True)
    e.add_argument("--export", required=True)
    e.add_argument("--adapter", required=True)
    e.add_argument("--tokenizer", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--held", required=True)
    t.add_argument("--rank-json", default=os.path.join(HERE, "logs", "exp_he19_rank_held400.json"))
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save", default=os.path.join(OUT, "he19_pointer"))
    t.add_argument("--tag", default="v1")
    a = ap.parse_args(argv)
    extract(a) if a.cmd == "extract" else train(a)
    with open(os.path.join(HERE, "logs", f"exp_he19_pointer_{a.cmd}_{getattr(a, 'tag', os.path.basename(a.out) if a.cmd == 'extract' else '')}.log"),
              "a", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
