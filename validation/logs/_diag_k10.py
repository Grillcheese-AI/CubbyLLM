"""Does the scene built from a K10 document read back the object at the fact's address? (algebra only)"""
import sys
import numpy as np, torch
sys.path.insert(0, "validation"); sys.path.insert(0, ".")
import exp_k10_hypertoken as k
from train_base import build_model
from cubbyllm.model.spatial import NO_PLACE

torch.manual_seed(0)
m = build_model(k.meta(256, True), torch.device("cpu"), seed=0)
rng = np.random.default_rng(0)
lang = k.Language(np.random.default_rng(1))
ids, P, ask_at, answers = k.batch(rng, lang, 4, 8, 120, 256)
placed = P[..., 0] != NO_PLACE
positions = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
link = m.spatial
print("placed per doc:", placed.sum(1).tolist(), " kernel S(0).S(6e1) =",
      float((link.encode(torch.zeros(1, 3)) * link.encode(torch.tensor([[6., 0., 0.]]))).sum()),
      " S(0).S(2e1) =", float((link.encode(torch.zeros(1, 3)) * link.encode(torch.tensor([[2., 0., 0.]]))).sum()))
with torch.no_grad():
    ctx = m.infer_context(ids)
    x = m.embedding.embed(ids, ctx)
    bound = link.bind(x, positions)
    scene = link.scene(bound, placed)
    E = m.embedding.embed(torch.arange(k.ENT0, k.ENT0 + k.N_ENT).unsqueeze(0), m.infer_context(ids[:1]))[0]   # (N_ENT, d)
    hits = 0
    for b in range(ids.shape[0]):
        for at, o in zip(ask_at[b], answers[b]):
            q = link.encode(positions[b, at])                    # S(addr), exact
            read = link.unbind(scene[b, at], q)                  # what the scene holds there
            e_o = x[b, at + 1]
            cos = torch.nn.functional.cosine_similarity(read, e_o, dim=0)
            sims = torch.nn.functional.cosine_similarity(read.unsqueeze(0), E, dim=1)
            best = int(sims.argmax()) + k.ENT0
            hits += int(best == o)
            if b == 0:
                print(f"  doc0 q@{at}: cos(read, e(o)) {float(cos):.3f}; nearest entity {best} (true {o}); 2nd {float(sims.topk(2).values[1]):.3f}")
    print(f"nearest-entity decode over {ids.shape[0] * len(ask_at[0])} questions: {hits / (ids.shape[0] * len(ask_at[0])):.2f}")
