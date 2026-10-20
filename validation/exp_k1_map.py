"""exp_k1_map -- the fact store as a map: entities as places, relations as displacements, in the
same Hilbert space the trunk binds places into (H-K1).

An entity e is a place S(e) = IFFT(exp(i theta_e)); a relation r is a displacement D_r =
IFFT(exp(i phi_r)); the claim "o is the r of s" is S(o) ~ S(s) (*) D_r -- in the Fourier domain,
theta_o ~ theta_s + phi_r. The score is the SSP inner product <S(o), S(s) (*) D_r> =
mean_f cos(theta_s + phi_r - theta_o), 1 for an exact fit, ~0 for a random pair. Three things
follow from the algebra and are measured here, against TransE at the same width and budget:

  one hop      rank the true object among every entity (filtered): Hits@1/10, MRR
  unbinding    "how is o related to s": D = S(o) (*) S(s)^-1, rank over the 2,902 relations
  composition  two hops by path integration, S(e0) (*) D1 (*) D2, never touching the store:
               the end of the walk ranked among every entity, against the one-hop rate

Inverses are free (phi -> -phi), so a backward hop is the same operator. The same phases the
spatial link uses for cells (cubbyllm/spatial/ssp.py) would place a village and the graph in one
space; here the entity phases are learned, which is the only part that is not algebra.

First read (2026-10-20; F = 256, 10 epochs, 8,191 shared negatives, 3,000 held-out, 2,000 paths;
lrs probed at 2 epochs -- map 0.05 of 0.05/0.2/0.5/1.0 with relations started uniform on the
circle, TransE 0.01 of 0.003/0.01/0.03, which memorised the training set, loss 0.13):
  map     one hop MRR 0.383 Hits@1 0.290 Hits@10 0.542 | relation MRR 0.180 Hits@3 0.188 | two-hop Hits@10 0.980
  TransE  one hop MRR 0.185 Hits@1 0.091 Hits@10 0.328 | relation MRR 0.037 Hits@3 0.041 | two-hop Hits@10 0.956
Gate 1 (composition >= 0.6 x one hop) passes at 1.81; gate 3 (one hop within 10% of TransE)
passes at 2.1x; gate 2 (relation Hits@3 >= 0.5) fails at 0.188, 4.6x the baseline, on a graph
of mean degree 1.9 where most entities carry one or two facts.

    python validation/exp_k1_map.py [--freq 256] [--epochs 10] [--neg 8191] [--eval 3000] [--out <json>]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIKI = os.path.join(ROOT, "standin", "data", "out", "wikikg")


def load(rng: random.Random, heldout: float = 0.05):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(WIKI, "triplets.parquet"), columns=["subject", "relation", "object"])
    s, r, o = (t.column(c).to_pylist() for c in ("subject", "relation", "object"))
    ents = {e: i for i, e in enumerate(sorted(set(s) | set(o)))}
    rels = {x: i for i, x in enumerate(sorted(set(r)))}
    tri = np.unique(np.array([(ents[a], rels[b], ents[c]) for a, b, c in zip(s, r, o)], dtype=np.int64), axis=0)
    idx = list(range(len(tri)))
    rng.shuffle(idx)
    n_hold = int(len(tri) * heldout)
    hold, train = tri[idx[:n_hold]], tri[idx[n_hold:]]
    seen_e, seen_r = set(train[:, 0]) | set(train[:, 2]), set(train[:, 1])
    hold = hold[[a in seen_e and c in seen_e and b in seen_r for a, b, c in hold]]
    return ents, rels, train, hold


def paths_2hop(ents, rels, train_set, n: int, rng: random.Random):
    """Two-hop walks whose BOTH hops are training triples; a 'backward' hop is the inverse."""
    import pyarrow.parquet as pq
    p = pq.read_table(os.path.join(WIKI, "paths.parquet"))
    out = []
    E, R, D = p.column("entities").to_pylist(), p.column("relations").to_pylist(), p.column("directions").to_pylist()
    order = list(range(len(E)))
    rng.shuffle(order)
    for i in order:
        es, rs, ds = E[i], R[i], D[i]
        if len(rs) != 2 or any(e not in ents for e in es) or any(r not in rels for r in rs):
            continue
        hops = []
        ok = True
        for k in range(2):
            a, b, r = ents[es[k]], ents[es[k + 1]], rels[rs[k]]
            fwd = ds[k] == "forward"
            if ((a, r, b) if fwd else (b, r, a)) not in train_set:
                ok = False
                break
            hops.append((r, 1.0 if fwd else -1.0))
        if ok and es[0] != es[2]:
            out.append((ents[es[0]], hops, ents[es[2]]))
        if len(out) >= n:
            break
    return out


def _unit_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """<S(a), S(b)> for phase rows a (Q, F), b (N, F): (Q, N) = mean_f cos(a - b), as one matmul."""
    ac, bc = torch.polar(torch.ones_like(a), a), torch.polar(torch.ones_like(b), b)
    return (ac.real @ bc.real.T + ac.imag @ bc.imag.T) / a.shape[-1]


class Map(torch.nn.Module):
    """Phases: entities (E, F), relations (R, F). score = mean cos(theta_s + sign*phi_r - theta_o).
    The entity table is a sparse embedding (375k rows: a dense gradient per step is the whole cost)."""

    def __init__(self, n_e: int, n_r: int, freq: int, tau: float = 12.0, rel_init: float = math.pi) -> None:
        super().__init__()
        self.e = torch.nn.Embedding(n_e, freq, sparse=True)
        self.e.weight.data.uniform_(-math.pi, math.pi)
        self.r = torch.nn.Parameter(torch.empty(n_r, freq).uniform_(-rel_init, rel_init))   # pi: distinct from the start
        self.tau = tau

    def query(self, s, r, sign=1.0):
        """The phases of S(s) (*) D_r^sign: a place."""
        return self.e(s) + sign * self.r[r]

    def score(self, q, o):
        """<S(q), S(o)> row by row: (B,)."""
        return torch.cos(q - o).mean(-1)

    def against(self, q, idx):
        """Query rows against the entities ``idx``: (B, len(idx))."""
        return _unit_dot(q, self.e(idx))

    def all_scores(self, q):
        return _unit_dot(q, self.e.weight)

    def rel_scores(self, s, o):
        """"How is o related to s": unbind, rank every relation: (Q, R)."""
        return _unit_dot(self.e(o) - self.e(s), self.r)


class TransE(torch.nn.Module):
    """The standard at the same width (2F real dims): score = -||e_s + r - e_o||."""

    def __init__(self, n_e: int, n_r: int, dim: int, tau: float = 4.0) -> None:
        super().__init__()
        self.e = torch.nn.Embedding(n_e, dim, sparse=True)
        self.e.weight.data.copy_(F.normalize(torch.randn(n_e, dim), dim=-1))
        self.r = torch.nn.Parameter(torch.randn(n_r, dim) * 0.1)
        self.tau = tau

    def query(self, s, r, sign=1.0):
        return self.e(s) + sign * self.r[r]

    def score(self, q, o):
        return -(q - o).norm(dim=-1)

    def against(self, q, idx):
        return -torch.cdist(q, self.e(idx))

    def all_scores(self, q):
        return -torch.cdist(q, self.e.weight)

    def rel_scores(self, s, o):
        return -torch.cdist(self.e(o) - self.e(s), self.r)


def train(model, tri: np.ndarray, epochs: int, neg: int, batch: int, lr: float, lr_rel: float, log) -> None:
    """Softmax over the true answer and ``neg`` negatives shared by the batch, both directions
    (s, r -> o and o, r^-1 -> s); a negative that is the answer is masked out. An entity row is
    touched about twice an epoch (mean degree 1.9), so ``lr`` is a step per touch: for phases, a
    fraction of a radian; relations are dense and stepped every batch."""
    opt_e = torch.optim.SparseAdam([model.e.weight], lr=lr)
    opt_r = torch.optim.Adam([model.r], lr=lr_rel)
    n_e = model.e.weight.shape[0]
    t = torch.from_numpy(tri)
    target = torch.zeros(batch, dtype=torch.long)
    for ep in range(epochs):
        perm = torch.randperm(len(t))
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(t), batch):
            b = t[perm[i:i + batch]]
            s, r, o = b[:, 0], b[:, 1], b[:, 2]
            negs = torch.randint(0, n_e, (neg,))
            loss = 0.0
            for q, true in ((model.query(s, r, 1.0), o), (model.query(o, r, -1.0), s)):   # both directions
                pos = model.score(q, model.e(true)).unsqueeze(1)                              # (B, 1)
                against = model.against(q, negs).masked_fill(negs.unsqueeze(0) == true.unsqueeze(1), -1e9)
                logits = model.tau * torch.cat([pos, against], 1)                            # (B, 1+neg)
                loss = loss + F.cross_entropy(logits, target[:len(b)])
            opt_e.zero_grad()
            opt_r.zero_grad()
            loss.backward()
            opt_e.step()
            opt_r.step()
            tot += float(loss)
            nb += 1
        log(f"  epoch {ep + 1}/{epochs}: loss {tot / nb:.3f} ({time.time() - t0:.0f}s)")


@torch.no_grad()
def ranks(scores: torch.Tensor, true: torch.Tensor, mask: list) -> np.ndarray:
    """1-based filtered ranks: entries in ``mask[i]`` (other true answers) do not count."""
    out = []
    for i in range(scores.shape[0]):
        row = scores[i].clone()
        s_true = float(row[true[i]])
        if mask[i]:
            row[torch.tensor(sorted(mask[i]), dtype=torch.long)] = -1e9
        out.append(int((row > s_true).sum()) + 1)
    return np.array(out)


def metrics(r: np.ndarray) -> dict:
    return {"n": int(len(r)), "mrr": float((1.0 / r).mean()), "hits1": float((r <= 1).mean()),
            "hits3": float((r <= 3).mean()), "hits10": float((r <= 10).mean())}


@torch.no_grad()
def evaluate(model, hold: np.ndarray, known: dict, known_rel: dict, paths: list, chunk: int = 256) -> dict:
    model.eval()
    h = torch.from_numpy(hold)
    one, rel = [], []
    for i in range(0, len(h), chunk):
        b = h[i:i + chunk]
        sc = model.all_scores(model.query(b[:, 0], b[:, 1], 1.0))
        one.append(ranks(sc, b[:, 2], [known.get((int(s), int(r)), set()) - {int(o)} for s, r, o in b.tolist()]))
        rs = model.rel_scores(b[:, 0], b[:, 2])
        rel.append(ranks(rs, b[:, 1], [known_rel.get((int(s), int(o)), set()) - {int(r)} for s, r, o in b.tolist()]))
    out = {"one_hop": metrics(np.concatenate(one)), "relation": metrics(np.concatenate(rel))}
    if paths:
        two = []
        for i in range(0, len(paths), chunk):
            ps = paths[i:i + chunk]
            q = model.e(torch.tensor([p[0] for p in ps]))
            for k in range(2):
                r = torch.tensor([p[1][k][0] for p in ps])
                sg = torch.tensor([p[1][k][1] for p in ps]).unsqueeze(-1)
                q = q + sg * model.r[r]
            two.append(ranks(model.all_scores(q), torch.tensor([p[2] for p in ps]), [set() for _ in ps]))
        out["two_hop"] = metrics(np.concatenate(two))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, default=256, help="phases per entity; the width is 2*freq real dims")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--neg", type=int, default=8191, help="negatives shared by a batch")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.05, help="map: radians per touch of an entity's phases "
                    "(SparseAdam's first touch steps ~3x lr; probed 2026-10-20 at 2 epochs: 0.05 > 0.2 > 0.5 > 1.0)")
    ap.add_argument("--lr-rel", type=float, default=0.05, help="map: the relations' displacement phases")
    ap.add_argument("--rel-init", type=float, default=math.pi, help="map: relation phases start uniform in +-this "
                    "(pi: distinct displacements from the start; 0.1: near-identity, one-hop fits faster)")
    ap.add_argument("--lr-transe", type=float, default=0.01, help="TransE: entities and relations")
    ap.add_argument("--eval", type=int, default=3000, help="held-out triples scored (all entities ranked)")
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=["map", "transe"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "exp_k1_map.json"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    ents, rels, tr, hold = load(rng)
    train_set = {tuple(x) for x in tr.tolist()}
    known, known_rel = collections.defaultdict(set), collections.defaultdict(set)
    for s, r, o in tr.tolist() + hold.tolist():
        known[(s, r)].add(o)
        known_rel[(s, o)].add(r)
    hold_eval = hold[:args.eval]
    paths = paths_2hop(ents, rels, train_set, args.paths, rng)
    log(f"wiki world: {len(ents):,} entities, {len(rels):,} relations, {len(tr):,} train triples, "
        f"{len(hold):,} held out ({len(hold_eval):,} scored), {len(paths):,} two-hop paths over training hops")
    report = {"args": vars(args), "entities": len(ents), "relations": len(rels), "train": int(len(tr)), "held": int(len(hold))}
    for name in args.only:
        torch.manual_seed(args.seed)
        model = (Map(len(ents), len(rels), args.freq, rel_init=args.rel_init) if name == "map"
                 else TransE(len(ents), len(rels), 2 * args.freq))
        lr, lr_rel = (args.lr, args.lr_rel) if name == "map" else (args.lr_transe, args.lr_transe)
        log(f"--- {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, width {2 * args.freq}, "
            f"lr {lr} (entities) / {lr_rel} (relations)")
        t0 = time.time()
        train(model, tr, args.epochs, args.neg, args.batch, lr, lr_rel, log)
        res = evaluate(model, hold_eval, known, known_rel, paths)
        res["train_s"] = round(time.time() - t0, 1)
        for k, v in res.items():
            if isinstance(v, dict):
                log(f"  {k:9s} MRR {v['mrr']:.3f}  Hits@1 {v['hits1']:.3f}  Hits@3 {v['hits3']:.3f}  "
                    f"Hits@10 {v['hits10']:.3f}  (n={v['n']})")
        if "two_hop" in res:
            res["composition_ratio"] = res["two_hop"]["hits10"] / max(res["one_hop"]["hits10"], 1e-9)
            log(f"  two-hop / one-hop Hits@10: {res['composition_ratio']:.2f}")
        report[name] = res
    report["log"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
