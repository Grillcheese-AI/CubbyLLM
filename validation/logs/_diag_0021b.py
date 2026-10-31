"""Scratch (order 0021): the read's best match by answer piece (present) against the absent first token."""
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
calls, by = [], {}
def hk(_m, inp, o):
    r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
    e = F.normalize(table.float()[cid], dim=-1); rn = F.normalize(r.float(), dim=-1)
    s = (rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)).masked_fill(~ok, float("-inf"))
    calls.append((idx, s.max(-1).values, o[2][:, 0]))
h = model.spatial.copy_path.register_forward_hook(hk)
with torch.no_grad():
    for b in range(0, ids.shape[0], 16):
        sl, S = slice(b, b + 16), ids.shape[1] - 1
        calls.clear()
        fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
        for ix, best, lam in calls:
            for j, v, g in zip(ix.tolist(), best.tolist(), lam.tolist()):
                i, t = divmod(j, S)
                here, o, info = spans[b + i]
                if info["absent"] and t == here:
                    by.setdefault("absent", []).append((v, g))
                elif not info["absent"] and here <= t < here + len(o):
                    by.setdefault(f"piece {t - here}", []).append((v, g))
                elif not info["absent"] and t == here + len(o):
                    by.setdefault("after the answer (EOS)", []).append((v, g))
for k in sorted(by):
    v = np.array(by[k])
    print(k.ljust(24), len(v), "best p5/p50/p95", np.round(np.percentile(v[:, 0], [5, 50, 95]), 3).tolist(), "lam p5/p50", np.round(np.percentile(v[:, 1], [5, 50]), 3).tolist())
