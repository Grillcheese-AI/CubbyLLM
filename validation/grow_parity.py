"""A grown base on grilly2: does it load, and does it give the base's logits?

Step 0's last check (docs/research/2026-09-25-grow-450m.md). Exports are made
with ``validation/export_base.py``; this compares, on the same prompts:
- the grown model in torch (float32, CPU) against the grown model on grilly2,
  the parity every base export gets;
- the grown model on grilly2 against the base on grilly2: the function
  preserved on the serving path too, not only in torch.

    python validation/grow_parity.py --grown <grown.pt> --grown-export <dir> --base-export <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

PROMPTS = [
    "The history of Quebec City begins with the founding of a trading post on the St. Lawrence River,",
    "La ville de Lévis est située sur la rive sud du fleuve Saint-Laurent, en face de Québec.",
    "Question: What caused the Thirty Years' War?\nAnswer:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grown", required=True, help="the grown checkpoint (grow_base.py --save)")
    ap.add_argument("--grown-export", required=True)
    ap.add_argument("--base-export", required=True)
    args = ap.parse_args()
    import gc

    import grilly
    import torch
    from tokenizers import Tokenizer
    from exp_e5_base450m_probes import load_grilly
    from train_base import load_base

    tk = Tokenizer.from_file(os.path.join(args.base_export, "tokenizer.json"))
    ids = [tk.encode(p).ids for p in PROMPTS]

    tm = load_base(args.grown, "cpu")
    with torch.no_grad():
        t_grown = [tm.forward(torch.tensor([x])).float().numpy()[0] for x in ids]
    del tm
    gc.collect()

    def on_grilly(export):
        gm, _ = load_grilly(export)
        with grilly.no_grad():
            out = [gm(grilly.tensor([x])).logits.numpy()[0] for x in ids]
        del gm
        gc.collect()
        return out

    g_grown = on_grilly(args.grown_export)
    g_base = on_grilly(args.base_export)
    for i, (t, g, b) in enumerate(zip(t_grown, g_grown, g_base)):
        scale = float(np.abs(t).max())
        d_tg = float(np.abs(g - t).max())
        d_bg = float(np.abs(g - b).max())
        a_tg = float((g.argmax(-1) == t.argmax(-1)).mean())
        a_bg = float((g.argmax(-1) == b.argmax(-1)).mean())
        print(f"prompt {i} ({len(ids[i])} tokens, logit scale {scale:.1f}): "
              f"grown torch vs grilly2 max |dlogit| {d_tg:.2e}, argmax agrees {a_tg:.4f} | "
              f"grown vs base on grilly2 max |dlogit| {d_bg:.2e}, argmax agrees {a_bg:.4f}", flush=True)


if __name__ == "__main__":
    main()
