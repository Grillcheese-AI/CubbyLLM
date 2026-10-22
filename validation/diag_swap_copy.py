"""diag_swap_copy -- is run 4's far-facts gain in the copy's parameters or in the trunk? (Cowork, 2026-10-21, order 0001)

Two arms on the same held-out documents, scored by ``train_base.fact_probe`` (teacher-forced, as
``exp_hypertoken_recall`` scores them): run 4 as trained, and run 3's trunk (every tensor but the copy's)
with run 4's trained copy (gate, tau, beta). Run 4 continued the whole model at a re-warmed learning rate;
if the second arm holds, the gain lives in the copy alone and the trunk can stay frozen.

    R3=<spatial_link_3_final.pt> R4=<spatial_link_4_final.pt> OUT=<json> CB_CORPUS=<token cache>
    CUBBY_SPM=<tokenizer.json> CB_PROBE_SHARDS=facts_far1k_lead,spatial_here CB_PROBE_DOCS=64
    python validation/diag_swap_copy.py
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
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
