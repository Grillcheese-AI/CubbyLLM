"""scratch: why copy-or-refuse refuses on the toy -- which condition, per present answer."""
import json, sys, os
import numpy as np
import torch
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import k10_refuse as kr
import exp_k10_switch as sw

path = sys.argv[1]
torch.set_num_threads(2)
model, a, lay, held = sw.load(path)
rep = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"k10_refuse_{a.arm}_s{a.seed}_2026-10-22.json")))
th, te = rep["theta"], rep["theta_end"]
rows = kr.capture(model, a, held, lay)
why = {"piece_unsure": 0, "end_unsure": 0, "first_empty": 0, "no_end": 0, "copied": 0}
copied_wrong = {"wrong_piece": 0, "too_long": 0, "too_short": 0}
piece_b, end_b = [], []
for info, o, b, picks in rows:
    if info["absent"]:
        continue
    n = len(o)
    piece_b += list(b[:n]); end_b.append(b[n])
    L = int(kr.sr.decide(torch.tensor(b)[None], th, te)[0])
    if L:
        why["copied"] += 1
        if L == n and picks[:n] != o: copied_wrong["wrong_piece"] += 1
        elif L > n: copied_wrong["too_long"] += 1
        elif L < n: copied_wrong["too_short"] += 1
        continue
    if b[0] < te: why["first_empty"] += 1
    elif any(te <= x < th for x in b[:n]): why["piece_unsure"] += 1
    elif te <= b[n] < th: why["end_unsure"] += 1
    else: why["no_end"] += 1
q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 25, 50, 75, 95])]
print(json.dumps({"arm": a.arm, "seed": a.seed, "theta": th, "theta_end": te, "refused_by": why, "copied_wrong": copied_wrong,
                  "piece_best_p5_25_50_75_95": q(piece_b), "end_best_p5_25_50_75_95": q(end_b),
                  "end_reads_above_theta": float(np.mean(np.array(end_b) >= th)),
                  "piece_reads_below_theta": float(np.mean(np.array(piece_b) < th))}, indent=1))
