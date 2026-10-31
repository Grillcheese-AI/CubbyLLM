"""scratch: U arm -- where the model's own answer under the contract differs from the simulation."""
import json, os, sys
import torch
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]
import k10_refuse as kr, exp_k10_switch as sw, exp_k10_free_text as ft, exp_k10_hypertoken as kh
from exp_k10_far_facts import fwd
torch.set_num_threads(2)
model, a, lay, held = sw.load(sys.argv[1])
rep = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"k10_refuse_{a.arm}_s{a.seed}_2026-10-22.json")))
th, te = rep["theta"], rep["theta_end"]
rows = kr.capture(model, a, held, lay)
ids, key, plc, R, _g, spans = held
model.spatial.copy_path.set_refuse(th, te, lay.none, kr.STEP, kr.K)
diff = []
with torch.no_grad():
    for b in range(0, ids.shape[0], 16):
        sl = slice(b, b + 16)
        pred = fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1]).argmax(-1)
        for i in range(pred.shape[0]):
            here, o, info = spans[b + i]
            info_, o_, bests, picks = rows[b + i]
            L = int(kr.sr.decide(torch.tensor(bests)[None], th, te)[0])
            sim = [lay.none] if L == 0 else picks[:L]
            said = pred[i, here:here + kr.K].tolist()
            cut = next((j for j, t in enumerate(said) if t in (kh.EOS, lay.none)), len(said))
            model_ans = said[:cut] or said[:1]
            if (model_ans != sim) and not (L == 0 and said[:1] == [lay.none]):
                diff.append({"absent": info["absent"], "L": L, "n": len(o), "sim_ok": sim == list(o), "said": said[:L + 2],
                             "picks": picks[:L + 1], "end_tok": said[L] if L < len(said) else None, "eos": kh.EOS})
print(len(diff)); print(json.dumps(diff[:6]))
