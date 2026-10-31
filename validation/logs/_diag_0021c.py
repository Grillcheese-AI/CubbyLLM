"""Scratch (order 0021): a hard switch on the read's best match alone (copy where best >= thr, the trunk below)."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation")
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM")
import exp_k10_free_text as ft
import exp_k10_switch as sw
from exp_k10_far_facts import fwd
model, a, lay, held = sw.load(sys.argv[1])
ids, key, plc, R, _g, spans = held
calls, picks = [], {}
def hk(_m, inp, o):
    r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
    e = F.normalize(table.float()[cid], dim=-1); rn = F.normalize(r.float(), dim=-1)
    s = (rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)).masked_fill(~ok, float("-inf"))
    calls.append((idx, s.max(-1).values, o[1].argmax(-1), inp[0].argmax(-1)))
h = model.spatial.copy_path.register_forward_hook(hk)
rows = []
with torch.no_grad():
    for b in range(0, ids.shape[0], 16):
        sl, S = slice(b, b + 16), ids.shape[1] - 1
        calls.clear()
        fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
        got = {}
        for ix, best, cp, lm in calls:
            for j, v, c, m in zip(ix.tolist(), best.tolist(), cp.tolist(), lm.tolist()):
                got[divmod(j, S)] = (v, c, m)
        for i in range(ids[sl].shape[0]):
            here, o, info = spans[b + i]
            rows.append((info, o, [got.get((i, here + j)) for j in range(len(o))]))
for thr in (0.16, 0.17, 0.18, 0.19, 0.20):
    ex, ab, wr, rf, n_p, n_a = 0, 0, 0, 0, 0, 0
    for info, o, g in rows:
        say = [(c if v >= thr else m) if x is not None else None for x in g for (v, c, m) in [x or (0, None, None)]]
        if info["absent"]:
            n_a += 1; ab += say[0] != lay.none
        else:
            n_p += 1; e = say == list(o); r_ = say[0] == lay.none
            ex += e; rf += r_; wr += (not e and not r_)
    print(f"thr {thr:.2f}: exact {ex / n_p:.3f}  present wrong {wr / n_p:.3f}  refused {rf / n_p:.3f}  absent answered {ab / n_a:.3f}")
print("-- copy or refuse: the copy's answer when every piece clears the threshold, else NONE")
for thr in (0.14, 0.16, 0.17, 0.18):
    ex, ab, wr, rf, n_p, n_a = 0, 0, 0, 0, 0, 0
    for info, o, g in rows:
        ok = all(x is not None and x[0] >= thr for x in g[:1 if info["absent"] else len(g)])
        say = [x[1] for x in g] if ok else [lay.none]
        if info["absent"]:
            n_a += 1; ab += say[0] != lay.none
        else:
            n_p += 1; e = say == list(o); r_ = say[0] == lay.none
            ex += e; rf += r_; wr += (not e and not r_)
    print(f"thr {thr:.2f}: exact {ex / n_p:.3f}  present wrong {wr / n_p:.3f}  refused {rf / n_p:.3f}  absent answered {ab / n_a:.3f}")
