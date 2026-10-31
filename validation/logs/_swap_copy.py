"""Scratch (Cowork, 2026-10-21): is run 4's gain in the copy's parameters or in the trunk?
Arms on the same held-out docs: run 4 as trained; run 3's trunk with run 4's trained copy (gate, tau, beta)."""
import json, os, sys, time
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import torch
import train_base as tb
R3, R4 = os.environ["R3"], os.environ["R4"]
dev, out, t0 = torch.device("cpu"), {}, time.time()
model = tb.load_base(R4, "cpu")
cp = model.spatial.copy_path
print("run 4 copy:", {n: (round(float(p.detach().flatten()[0]), 4) if p.numel() == 1 else round(float(p.detach().norm()), 4)) for n, p in cp.named_parameters()}, flush=True)
out["run4"] = tb.fact_probe(model, dev); print("run4", out["run4"], f"[{time.time()-t0:.0f}s]", flush=True)
ck = torch.load(R3, map_location="cpu", weights_only=False, mmap=True)
own = {id(p) for p in cp.parameters()}
rest = [p for p in model.parameters() if id(p) not in own]
assert len(rest) == len(ck["params"]) and all(p.shape == s.shape for p, s in zip(rest, ck["params"]))
with torch.no_grad():
    for p, s in zip(rest, ck["params"]): p.copy_(s)
    for p, s in zip(model.context_source.parameters(), ck["router"]): p.copy_(s)
del ck
out["run3_trunk_run4_copy"] = tb.fact_probe(model, dev); print("run3 trunk + run4 copy", out["run3_trunk_run4_copy"], f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open(os.environ["OUT"], "w"), indent=1)
