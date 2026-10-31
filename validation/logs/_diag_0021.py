"""Scratch (order 0021): on a saved joint model, at absent questions' first answer token -- does the trunk alone say
NONE, what is the gate, and what does the copy alone pick? And at present facts' first piece, the gate's spread."""
import sys
import numpy as np
import torch
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation")
sys.path.insert(0, r"C:\Users\grill\Documents\GitHub\CubbyLLM")
import exp_k10_free_text as ft
import exp_k10_switch as sw
from exp_k10_far_facts import fwd

model, a, lay, held = sw.load(sys.argv[1])
ids, key, plc, R, _g, spans = held
calls, out = [], {"absent_lm_none": [], "absent_lam": [], "absent_mixed_none": [], "present_lam_first": []}
hook = model.spatial.copy_path.register_forward_hook(lambda _m, inp, o: calls.append((inp[4], inp[0].argmax(-1), o[0].argmax(-1), o[2][:, 0])))
with torch.no_grad():
    for b in range(0, ids.shape[0], 16):
        sl, S = slice(b, b + 16), ids.shape[1] - 1
        calls.clear()
        fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
        for ix, lm, mix, lam in calls:
            for j, l, m, g in zip(ix.tolist(), lm.tolist(), mix.tolist(), lam.tolist()):
                i, t = divmod(j, S)
                here, o, info = spans[b + i]
                if t == here and info["absent"]:
                    out["absent_lm_none"].append(l == lay.none); out["absent_lam"].append(g); out["absent_mixed_none"].append(m == lay.none)
                elif t == here:
                    out["present_lam_first"].append(g)
hook.remove()
for k, v in out.items():
    v = np.array(v, dtype=float)
    print(k, len(v), round(float(v.mean()), 3), "p10/p50/p90", np.round(np.percentile(v, [10, 50, 90]), 3).tolist())
# the read's best-match cosine (the gate's own `best`) and its margin over the second best: absent HERE vs present first piece
import torch.nn.functional as F
rec = {"absent": [], "present": []}
def hk(_m, inp, _o):
    r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
    e = F.normalize(table.float()[cid], dim=-1)
    rn = F.normalize(r.float(), dim=-1)
    s = rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)
    s = s.masked_fill(~ok, float("-inf"))
    top = s.topk(2, -1).values
    calls.append((idx, top[:, 0], top[:, 0] - top[:, 1]))
h2 = model.spatial.copy_path.register_forward_hook(hk)
with torch.no_grad():
    for b in range(0, ids.shape[0], 16):
        sl, S = slice(b, b + 16), ids.shape[1] - 1
        calls.clear()
        fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
        for ix, best, marg in calls:
            for j, v, mg in zip(ix.tolist(), best.tolist(), marg.tolist()):
                i, t = divmod(j, S)
                here, o, info = spans[b + i]
                if t == here:
                    rec["absent" if info["absent"] else "present"].append((v, mg))
h2.remove()
for k, v in rec.items():
    v = np.array(v)
    print(k, "best p5/p50/p95", np.round(np.percentile(v[:, 0], [5, 50, 95]), 3).tolist(), "margin p5/p50/p95", np.round(np.percentile(v[:, 1], [5, 50, 95]), 3).tolist())
pb, ab = np.array(rec["present"])[:, 0], np.array(rec["absent"])[:, 0]
best_thr = max(((pb >= t).mean() + (ab < t).mean(), t) for t in np.linspace(0, 0.6, 121))
print("one threshold on best: present kept + absent closed =", round(best_thr[0], 3), "of 2 at", round(best_thr[1], 3))
