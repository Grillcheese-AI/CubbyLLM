"""H-M1: specialise a teacher's word vectors for SIMILARITY with WordNet constraints (Attract-Repel, Mrksic et al.,
TACL 2017), as a GLOBAL linear map so it reaches words no constraint names (post-specialisation, Vulic et al. 2018).

Why: a paraphrase teacher pulls big/large together but also france/germany (H-M1 teacher swap: paraphrase recall
3.7x, different-fact leak 10x). Attract = WordNet synonyms (same synset); repel = antonyms and co-hyponyms (lemmas of
sibling synsets under one hypernym: france/germany are both instances of 'European country').

GENERALISATION GUARD: every word of the H-M1 screen's test classes (PARAPHRASE pairs, ENTITY_SETS) is removed from
every constraint, and the map is linear, so the screen can only improve through generalisation. The constraint
words are also split 90/10 by word: the dev pairs (words never trained on) report the map's own generalisation.

    python validation/hm1_attract_repel.py --teacher teacher_potion.npz --out teacher_potion_ar.npz
then build the table:  build_fastword_table.py --teacher npz:<out> ...   (same vocabulary, same projection)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hm1_ngram_hv import ENTITY_SETS, PARAPHRASE  # noqa: E402


def constraints(vocab: set[str], banned: set[str], max_sib: int, rng: random.Random):
    from nltk.corpus import wordnet as wn
    ok = lambda w: w in vocab and w not in banned and w.isalpha()
    syn, ant, sib = set(), set(), set()
    for s in wn.all_synsets():
        lem = [l.name().lower() for l in s.lemmas() if ok(l.name().lower())]
        for i in range(len(lem)):
            for j in range(i + 1, len(lem)):
                if lem[i] != lem[j]:
                    syn.add(tuple(sorted((lem[i], lem[j]))))
        for l in s.lemmas():
            a = l.name().lower()
            for b in l.antonyms():
                b = b.name().lower()
                if ok(a) and ok(b) and a != b:
                    ant.add(tuple(sorted((a, b))))
    for h in wn.all_synsets("n"):
        kids = h.hyponyms() + h.instance_hyponyms()
        words = [[l.name().lower() for l in k.lemmas() if ok(l.name().lower())] for k in kids]
        words = [w for w in words if w]
        pairs = [(a, b) for i in range(len(words)) for j in range(i + 1, len(words)) for a in words[i][:1] for b in words[j][:1]]
        rng.shuffle(pairs)
        for a, b in pairs[:max_sib]:
            if a != b:
                sib.add(tuple(sorted((a, b))))
    sib -= syn
    return sorted(syn), sorted(ant), sorted(sib)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", required=True, help="npz with vocab, E (build_fastword_table --dump-teacher)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sib", type=int, default=20, help="co-hyponym pairs kept per hypernym")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--att", type=float, default=0.6, help="attract margin")
    ap.add_argument("--rep", type=float, default=0.0, help="repel margin")
    ap.add_argument("--reg", type=float, default=0.1, help="weight of ||W - I||^2")
    ap.add_argument("--obj", default="direct", choices=("direct", "ar"))
    ap.add_argument("--keep", type=float, default=10.0, help="direct: weight of keeping random-pair cosines")
    ap.add_argument("--rep-w", type=float, default=1.0, help="direct: weight of the repel term")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    import torch
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    t0 = time.time()
    z = np.load(a.teacher, allow_pickle=True)
    vocab = [str(w) for w in z["vocab"]]
    E = np.asarray(z["E"], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    idx = {w: i for i, w in enumerate(vocab)}
    banned = {w for p in PARAPHRASE for w in p} | {w for ws in ENTITY_SETS.values() for w in ws}
    syn, ant, sib = constraints(set(vocab), banned, a.max_sib, rng)
    words = sorted({w for p in syn + ant + sib for w in p})
    dev_words = set(rng.sample(words, len(words) // 10))
    split = lambda ps: ([p for p in ps if p[0] not in dev_words and p[1] not in dev_words],
                        [p for p in ps if p[0] in dev_words or p[1] in dev_words])
    (syn_tr, syn_dv), (ant_tr, ant_dv), (sib_tr, sib_dv) = split(syn), split(ant), split(sib)
    lines = [f"attract-repel | {os.path.basename(a.teacher)}: {len(vocab)} words, dim {E.shape[1]} | WordNet constraints "
             f"(test-class words banned: {len(banned)}): synonyms {len(syn)}, antonyms {len(ant)}, co-hyponyms {len(sib)} | "
             f"train {len(syn_tr)}/{len(ant_tr)}/{len(sib_tr)}, dev (unseen words) {len(syn_dv)}/{len(ant_dv)}/{len(sib_dv)}"]
    print(lines[0], flush=True)
    X = torch.from_numpy(E)
    d = E.shape[1]
    W = torch.nn.Parameter(torch.eye(d))
    opt = torch.optim.Adam([W], lr=a.lr)

    def enc(ids):
        y = X[ids] @ W
        return y / (y.norm(dim=1, keepdim=True) + 1e-9)

    def ids_of(ps):
        return torch.tensor([[idx[x], idx[y]] for x, y in ps], dtype=torch.long)

    T_att, T_rep = ids_of(syn_tr), ids_of(ant_tr + sib_tr)

    def ar_loss(P, attract: bool):
        L, R = enc(P[:, 0]), enc(P[:, 1])
        both = torch.cat([L, R])
        S = L @ both.T                                   # each left word against every in-batch word
        n = len(P)
        S[torch.arange(n), torch.arange(n)] = float("nan")       # itself
        pos = (L * R).sum(1)
        if attract:                                      # hardest negative: the most similar other word
            neg = torch.nan_to_num(S, nan=-2.0)
            neg[torch.arange(n), n + torch.arange(n)] = -2.0
            return torch.relu(a.att + neg.max(1).values - pos).mean()
        neg = torch.nan_to_num(S, nan=2.0)               # repel: the partner must be LESS similar than the farthest
        neg[torch.arange(n), n + torch.arange(n)] = 2.0
        return torch.relu(a.rep + pos - neg.min(1).values).mean()

    def report(tag):
        out = {}
        with torch.no_grad():
            for name, ps in (("syn", syn_dv), ("ant", ant_dv), ("sib", sib_dv), ("train syn", syn_tr[:3000]),
                             ("train sib", sib_tr[:3000])):
                if ps:
                    P = ids_of(ps)
                    out[name] = float((enc(P[:, 0]) * enc(P[:, 1])).sum(1).mean())
            for name, ps in (("test paraphrase", [p for p in PARAPHRASE if p[0] in idx and p[1] in idx]),
                             ("test entity", [(ws[i], ws[j]) for ws in ENTITY_SETS.values() for i in range(len(ws))
                                              for j in range(i + 1, len(ws)) if ws[i] in idx and ws[j] in idx])):
                P = ids_of(ps)
                out[name] = float((enc(P[:, 0]) * enc(P[:, 1])).sum(1).mean())
        line = f"  {tag:<10} mean cosine | dev synonyms {out.get('syn', 0):.3f}  dev antonyms {out.get('ant', 0):.3f}  " \
               f"dev co-hyponyms {out.get('sib', 0):.3f} | train syn {out['train syn']:.3f} sib {out['train sib']:.3f} | test paraphrase {out['test paraphrase']:.3f}  test entity {out['test entity']:.3f}"
        print(line, flush=True)
        lines.append(line)
        return out

    hist = {"before": report("before")}
    I = torch.eye(d)
    with torch.no_grad():                                 # the unrelated level: mean cosine of random word pairs
        rp = torch.randint(0, len(vocab), (20000, 2))
        base_rand = float((X[rp[:, 0]] * X[rp[:, 1]]).sum(1).mean())
    lines.append(f"  random-pair cosine (the repel target): {base_rand:.3f}")
    for ep in range(a.epochs):
        ra = torch.randperm(len(T_att)); rr = torch.randperm(len(T_rep))
        nb = max(len(T_att), len(T_rep)) // a.batch
        tot = 0.0
        for b in range(nb):
            pa = T_att[ra[(b * a.batch) % len(T_att):][:a.batch]]
            pr = T_rep[rr[(b * a.batch) % len(T_rep):][:a.batch]]
            if a.obj == "ar":
                loss = ar_loss(pa, True) + ar_loss(pr, False) + a.reg * ((W - I) ** 2).sum()
            else:                                         # direct: attract, repel to the unrelated level, keep geometry
                ca = (enc(pa[:, 0]) * enc(pa[:, 1])).sum(1)
                cr = (enc(pr[:, 0]) * enc(pr[:, 1])).sum(1)
                rp = torch.randint(0, len(vocab), (a.batch, 2))
                cn = (enc(rp[:, 0]) * enc(rp[:, 1])).sum(1)
                c0 = (X[rp[:, 0]] * X[rp[:, 1]]).sum(1)
                loss = (1 - ca).mean() + a.rep_w * torch.relu(cr - base_rand).mean() + a.keep * ((cn - c0) ** 2).mean() \
                    + a.reg * ((W - I) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        hist[f"epoch{ep + 1}"] = report(f"epoch {ep + 1}")
        lines.append(f"    loss {tot / max(1, nb):.4f}")
    with torch.no_grad():
        Y = (X @ W).numpy()
    np.savez(a.out, vocab=np.array(vocab, dtype=object), E=Y.astype(np.float32))
    lines.append(f"  -> {a.out} ({time.time() - t0:.0f}s)")
    print(lines[-1])
    base = os.path.join(HERE, "logs", f"hm1_attract_repel_{a.tag or os.path.splitext(os.path.basename(a.teacher))[0]}")
    open(base + ".log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"args": vars(a), "history": hist}, open(base + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
