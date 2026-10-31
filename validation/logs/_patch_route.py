import io, sys
path = sys.argv[1]
text = io.open(path, encoding="utf-8").read()

def swap(old, new):
    global text
    assert text.count(old) == 1, old[:70]
    text = text.replace(old, new)

swap("Status: **OPEN — gate 1 (the address with the token) and gate 2 passed; the learned route failed at this scale and budget; next: "
     "the fact link with H-K1's phases as entity codes (gate 4), spans (gate 3), and the learned route on the 450M (H-S12 gates 1–2) — "
     "with the shard's questions carrying their addresses, so the route is measured apart from the read.**",
     "**Second read — the route (2026-10-20, same trunk and budget; `validation/logs/exp_k10_hypertoken_*_2026-10-20.json`).** "
     "The learned route is teachable: a *route loss*, `1 − cos(unitary(q), S(p))` at the placed tokens — the side file knows every "
     "question's address, so the trunk's own query is taught where to look, for free at training time — takes `--query learned` from "
     "0.00 to **0.82** (route cosine 0.54). What did not work, recorded: the gate pinned open for 250 steps then released (an outside "
     "suggestion, freeze-and-release) 0.00 alone and 0.77 with the route loss; a per-frequency complex query head (the shape of the "
     "deconvolution, d parameters instead of d²) 0.69 with the route loss — the dense head can also project the mixing layers' noise "
     "out, the diagonal one cannot; a *matched* read (the query's conjugate at its own magnitudes instead of unitary(), to stop a "
     "noise-dominated frequency being stretched to modulus 1) 0.19 — unitary() is right for this read; the dense head started at the "
     "identity 0.66. Controls asked for by the outside review and run: attention over the *whole* document (window 256) **0.03** at "
     "1,000 steps — the algebraic read is the sample-efficient path, not a stand-in for a longer window; a *mismatched address* (every "
     "`HERE` carrying another fact's valid address) **0.00** on every arm — the channel relies on the address alone, no other path "
     "carries the answer; and the trunk's own gated recurrence (MinGRU, a dense state past the window) is the text control that sat at "
     "0.00. Built: `CB_ROUTE_LOSS=λ` in `train_base` (the query tapped in the forward; needs `CB_GRAD_CKPT=0`), "
     "`validation/make_fact_shard.py` (the wiki world's facts as hypertokens: a fact's address is the subject's hashed cell plus the "
     "relation's step, a name a span up the z axis read piece by piece; `facts` / `facts_text` built, 20,000 documents, 240,208 facts, "
     "119,839 questions, on the cache and the Drive), and `make_spatial_shard.py --ask placed` (`spatial_here` / `spatial_text_here`: "
     "every village question carries its address). Status: **OPEN — gates 1–2 passed with the address given, the learned route at 0.82 "
     "when taught; next: the 450M continued on `spatial_here` + `facts` with `CB_SPATIAL=1 CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1` "
     "against the text controls at the same tokens (H-S12 gates 1–2 and this entry's gate 4 in one run), then spans (gate 3).**")

swap("H-K10's fact link with H-K1's phases (gate 4) and the learned route at the 450M's scale (H-S12 gates 1–2) are the next builds.",
     "H-K10's route read the same evening (the learned route 0.00 → 0.82 when taught by the route loss; attention over the whole "
     "document 0.03; a mismatched address 0.00) and its shards built (`facts`, `spatial_here` + text controls): the 450M run on them "
     "(`CB_SPATIAL=1 CB_SPATIAL_QUERY=placed CB_ROUTE_LOSS=1`) is the next Colab job — H-S12 gates 1–2 and H-K10 gate 4 in one.")
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched")
