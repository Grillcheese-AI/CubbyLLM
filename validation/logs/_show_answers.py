"""Scratch (Cowork, 2026-10-22): free-running answers to held-out far-fact questions, greedy, for two models:
run 3 as trained (no copy) and run 3's trunk with run 4's copy. Prints question / gold / each model's answer."""
import json, os, sys, time
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, torch
import train_base as tb
import exp_hypertoken_recall as hr
os.environ.pop("CB_NO_FLEX", None)
from tokenizers import Tokenizer
from cubbyllm.model.spatial import NO_PLACE
R3, R4, N = os.environ["R3"], os.environ["R4"], int(os.environ.get("NQ", "10"))
hr.CORPUS = os.environ["CB_CORPUS"]
tk = Tokenizer.from_file(os.environ["CUBBY_SPM"])
NL = tk.encode("\n", add_special_tokens=False).ids[-1]
QTAG = tk.encode("Question:", add_special_tokens=False).ids
STEP = 6

def models():
    m3 = tb.load_base(R3, "cpu")
    mh = tb.load_base(R4, "cpu")
    cp = mh.spatial.copy_path
    own = {id(p) for p in cp.parameters()}
    rest = [p for p in mh.parameters() if id(p) not in own]
    ck = torch.load(R3, map_location="cpu", weights_only=False, mmap=True)
    with torch.no_grad():
        for p, s in zip(rest, ck["params"]): p.copy_(s)
        for p, s in zip(mh.context_source.parameters(), ck["router"]): p.copy_(s)
    return {"run 3 (no copy)": m3, "run 3 + copy": mh}

@torch.no_grad()
def greedy(model, ids, places, roles, lead_place, lams):
    ids, places, roles, out = list(ids), [list(p) for p in places], list(roles), []
    for i in range(10):
        x = torch.tensor([ids])
        P = torch.tensor([places], dtype=torch.long)
        placed = P[..., 0] != NO_PLACE
        pos = torch.where(placed.unsqueeze(-1), P, torch.zeros_like(P)).float()
        R = torch.tensor([roles])
        lams.clear()
        logits = model.forward(x, positions=pos, placed=placed, writes=placed & ((R & 1) != 0), asks=placed & ((R & 2) != 0))
        t = int(logits[0, -1].argmax())
        if t == NL or i == 9:
            break
        out.append(t)
        ids.append(t); places.append([lead_place[0], lead_place[1], lead_place[2] + STEP * (i + 1)]); roles.append(2)
    return out

x, pl = hr.heldout_docs("facts_far1k_lead", 40)
places, roles = pl
ms = models()
lams = []
rows, t0, shown = [], time.time(), 0
for r in range(len(x)):
    if shown >= N: break
    row = x[r]
    spans = hr.answers(tk, row)
    if not spans: continue
    j, k = spans[0]
    q = max(i for i in range(j) if list(row[i:i + len(QTAG)]) == QTAG)
    rec = {"question": tk.decode([int(t) for t in row[q:j]]).strip(), "gold": tk.decode([int(t) for t in row[j:k + 1]]).strip()}
    for name, m in ms.items():
        g = greedy(m, row[:j], places[r, :j], roles[r, :j], places[r, j - 1], lams)
        rec[name] = tk.decode(g).strip()
    rows.append(rec); shown += 1
    print(json.dumps(rec, ensure_ascii=False), f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(rows, open(os.environ["OUT"], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
