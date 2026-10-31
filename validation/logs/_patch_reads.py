import io, sys
path, inwin = sys.argv[1], sys.argv[2]
text = io.open(path, encoding="utf-8").read()

def swap(old, new):
    global text
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)

# H-K1 status -> first read
swap("Status: **OPEN — running 2026-10-20 (F = 256, 12 epochs); first read below when it lands.**",
     "**First read (2026-10-20; F = 256, 10 epochs, 8,191 shared negatives, 3,000 held-out triples, 2,000 two-hop paths; "
     "`validation/logs/exp_k1_map_{map,transe}_2026-10-20.json`):** the map — one hop MRR 0.383 / Hits@1 0.290 / Hits@10 0.542, "
     "relation discovery by unbinding MRR 0.180 / Hits@3 0.188, two-hop by path integration Hits@10 0.980; TransE at the same width "
     "and budget — one hop MRR 0.185 / Hits@1 0.091 / Hits@10 0.328, relation MRR 0.037 / Hits@3 0.041, two-hop Hits@10 0.956. "
     "Gate 1 passes (two-hop / one-hop Hits@10 = 1.81, the paths being over training hops); gate 3 passes at 2.1× TransE instead of "
     "within 10%; **gate 2 fails** at Hits@3 0.188 — 4.6× the baseline but far from 0.5, on a graph of mean degree 1.9 where most "
     "entities carry one or two facts and the displacement between a held-out pair is generalisation, not fit. Caveats, stated: the "
     "map's learning rate was probed at 2 epochs (0.05 > 0.2 > 0.5 > 1.0; SparseAdam's first touch steps ~3× lr on a row touched "
     "twice an epoch) and its relations start uniform on the circle (a relation is a displacement, distinct from the start — it cost "
     "one hop at 2 epochs and bought relation discovery); TransE's rate was probed the same way (0.01 of 0.003 / 0.01 / 0.03) and it "
     "memorised the training set (loss 0.13) — a stronger baseline would be regularised, so the 2× is a first read, not a claim. "
     "Status: **OPEN — gates 1 and 3 passed, gate 2 failed as pre-registered; next: the map's phases as the fact link's codes (H-K10 "
     "gate 4), and relation discovery re-read on the dense core of the graph (entities with ≥ 5 facts).**")

# H-K10 status -> first read
swap("Status: **OPEN — written 2026-10-20; gate 1 is a CPU hour on the tiny trunk, next.**",
     "**First read (2026-10-20, `validation/exp_k10_hypertoken.py`, a 3.4M-parameter `train_base` trunk, D = 256, window 64, 8 random "
     "facts per document shown once, questions 120+ tokens later, 1,000 steps × 16 documents, 25 min on the CPU per arm; "
     "`validation/logs/exp_k10_hypertoken_*_2026-10-20.json`):** before any training the scene decodes the object at the fact's "
     "address at 0.84 by nearest entity (cos 0.17–0.35 to the true embedding: the attenuated read of H-K8, with `HERE`'s own bound "
     "embedding at the same address). *The hypertoken arm* — the question's `HERE` token carries its address and the read's query "
     "**is** that address (the gate and the decode `o` learned, the gate started open at 0 for a trunk trained from scratch): held-out "
     "answers 0.00 → 0.21 (step 200) → 0.53 → 0.81 → 0.86 → 0.92 → **0.94** at 1,000 — **gate 1's first clause passes** (≥ 90%); "
     "*the text control*, the same documents with nothing placed: **0.00** throughout, the answer beyond attention's reach; the "
     "filler language's held-out loss 2.075 in both arms — **gate 2 passes**; the in-window control (4 facts, no filler): " + inwin + ". "
     "*The learned-query arm* — H-S12's route, the trunk asked to recover the address from its own state through `q = W rms(h)`: "
     "**0.00** for all 1,000 steps, the gate drifting shut (−4 → −4.08 from the design's start; 0 → −0.36 when started open). "
     "Read plainly: the algebra delivers, the trunk learns to decode an algebraic read in a few hundred steps when the address arrives "
     "with the token, and *recovering* an address from the hidden state is the hard part at this scale — which is exactly H-S12's "
     "gate 2 risk, now measured small. Status: **OPEN — gate 1 (given address) and gate 2 passed; the learned route failed at this "
     "scale and budget; next: the fact link with H-K1's phases as entity codes (gate 4), spans (gate 3), and the learned route on "
     "the 450M (H-S12 gate 1–2).**")

# section 10 item
swap("H-K1 running (`validation/exp_k1_map.py`, map vs TransE on the wiki world).",
     "H-K1 read the same day (`validation/exp_k1_map.py`: the map 2× TransE on one hop, 4.6× on relation discovery, both compose; "
     "gate 2 failed as pre-registered); H-K7 gate 1 read (`exp_k7_exposure.py`: 0.00 / 0.00 / 0.50 / 0.99 at 1 / 10 / 100 / 1,000 "
     "sightings); H-K10 gate 1 read (`exp_k10_hypertoken.py`: a fact shown once answered at 0.94 beyond the window through the "
     "hypertoken, 0.00 for text; the learned route 0.00).")
swap("H-K10 gate 1 (hypertokens: a fact shown once, read back beyond the window) is a CPU hour on the tiny trunk and comes right after K1.",
     "H-K10's fact link with H-K1's phases (gate 4) and the learned route at the 450M's scale (H-S12 gates 1–2) are the next builds.")
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched")
