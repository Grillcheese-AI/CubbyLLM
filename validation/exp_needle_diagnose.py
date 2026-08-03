"""Why is the needle probe at chance? Separate 'broken instrument' from 'model can't'.

The needle sweep returned 12.5% (= chance, 1 of 8) at EVERY depth including
depth 1.0, where the fact sits ~15 tokens before the question. That is not a
long-context result; nothing about a fixed-size state should fail at 15 tokens.
So before concluding anything about the architecture, rule out the probe.

Four checks, cheapest first, each with a pass condition stated up front:

  1. TOKENISER — does encode() inject specials, and does the answer tokenise the
     same way inside the needle as it does standalone? If BPE splits " 47213"
     differently in the two positions, the candidate we score is not the string
     the model saw, and no amount of training would produce a hit.
  2. ZERO-DISTANCE — needle immediately followed by the question, no filler at
     all. This is pure sequence continuation, not retrieval. Chance here means
     the probe is broken OR the model cannot copy adjacent tokens.
  3. COPY FLOOR — can the model copy a token it saw two tokens ago? The minimal
     in-context ability. If this fails the model has no copying circuit yet, and
     needle-in-a-haystack cannot possibly work regardless of state design.
  4. DISCRIMINATION — are the candidate scores actually different from each
     other, or is the model assigning ~identical likelihood to all eight? Flat
     scores mean the test has no resolution, not that the answer is wrong.

  CB_CKPT=... CUBBY_SPM=... python validation/exp_needle_diagnose.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.argv = sys.argv[:1]
from exp_needle_recall import _clone, build, prefix_state, score_from  # noqa: E402

CKPT = os.environ.get("CB_CKPT", "")
SPM = os.environ.get("CUBBY_SPM", "")
ANS = ["47213", "3092", "8814", "60517", "27384", "91650", "5573", "10428"]
FACT = "The secret access code for the north vault is {}."
PROBE = "The secret access code for the north vault is"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu")
    model, d, L, V, kind = build(ck["meta"], dev)
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    del ck
    from cubbyllm.training.data import _load_tokenizer
    encode, decode, eos, _ = _load_tokenizer(SPM)
    amp = dev.type == "cuda"

    print("=== 1. TOKENISER — does the candidate match what the model saw? ===")
    needle = encode(FACT.format(ANS[0]))
    cand = encode(" " + ANS[0])
    print(f"  needle ids  : {needle}")
    print(f"  decoded back: {decode(needle)!r}")
    print(f"  candidate   : {cand} -> {decode(cand)!r}")
    sub = any(needle[i:i+len(cand)] == cand for i in range(len(needle)-len(cand)+1))
    print(f"  candidate appears verbatim inside needle? {sub}")
    if not sub:
        print("  *** MISMATCH: the tokens we score are NOT the tokens in the needle.")
        print("      BPE is splitting the answer differently in the two positions.")
        print("      This alone would pin the probe at chance. FIX BEFORE ANY RERUN.")
    print()

    print("=== 2. ZERO-DISTANCE — needle then question, no filler ===")
    print("  (pure continuation, not retrieval. chance = 12.5%)")
    hits = 0
    for i, a in enumerate(ANS):
        ctx = encode(FACT.format(a)) + encode(" " + PROBE)
        pre = torch.tensor(ctx, dtype=torch.long, device=dev)
        lg, st = prefix_state(model, pre, dev, amp)
        sc = [score_from(model, lg, st,
                         torch.tensor(encode(" " + c), dtype=torch.long, device=dev), amp)
              for c in ANS]
        win = max(range(len(sc)), key=lambda j: sc[j])
        hits += int(win == i)
        if i == 0:
            rng = max(sc) - min(sc)
            print(f"  trial 0 scores: {[f'{s:.3f}' for s in sc]}")
            print(f"  spread max-min = {rng:.4f}   (near 0 => no discrimination at all)")
    print(f"  zero-distance accuracy: {hits}/{len(ANS)} = {hits/len(ANS):.1%}")
    print("  PASS if well above 12.5%. At chance here, retrieval at 4k is moot.\n")

    print("=== 3. COPY FLOOR — repeat a token seen 2 positions back ===")
    print("  sequence 'A B C A B ?' -> is C the top candidate?")
    torch.manual_seed(0)
    ok = 0
    for _ in range(20):
        a, b, c = torch.randint(0, V, (3,)).tolist()
        ids = torch.tensor([a, b, c, a, b], dtype=torch.long, device=dev)
        lg, _ = prefix_state(model, ids, dev, amp)
        ok += int(int(lg[0].argmax()) == c)
    print(f"  exact-copy top-1: {ok}/20 = {ok/20:.0%}  (random would be ~0%)")
    print("  If 0%, the model has no in-context copying circuit yet — expected for")
    print("  a heavily undertrained model, and it makes needle tests uninformative.\n")

    print("=== 4. VERDICT ===")
    print("  broken instrument  -> check 1 mismatched, or check 2 at chance with a")
    print("                        near-zero score spread")
    print("  model not ready    -> checks 1 and 2 fine but check 3 at 0%: no copying")
    print("                        circuit has emerged; rerun on the final checkpoint")
    print("  genuine limitation -> checks 1-3 all pass and accuracy still decays with")
    print("                        DEPTH. Only this reading supports a claim about the")
    print("                        architecture.")


if __name__ == "__main__":
    main()
