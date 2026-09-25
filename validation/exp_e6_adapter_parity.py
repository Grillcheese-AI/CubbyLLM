"""Does one talk-adapter file mean the same thing in torch and in grilly2?

The adapter can be trained on either side (`train_talk_torch.py` on Colab, `train_talk_grilly.py` on the
local card) and is gated and served on grilly2. So the file's layout -- grilly2's module names, the
(alpha / r) * B (A x) convention, which projection each name is -- has to mean the same model on both.
This loads one adapter into this repo's torch model (CPU, float32) and into grilly2's native model (the
local GPU) and compares their logits on talk prompts; the adapter's own effect (with vs without) is
printed beside it, so a mismatch cannot hide under a small adapter.

    python validation/exp_e6_adapter_parity.py --ckpt <base450m_final.pt> --export <export dir> \
        --adapter <adapter dir> --data standin/data/out/ground_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "standin"), os.path.join(ROOT, "standin", "data")):
    sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    import torch
    import grilly
    from tokenizers import Tokenizer
    from exp_e5_base450m_probes import load_grilly
    from exp_e6_talk_gate import load_adapter as load_grilly_adapter
    from train_base import load_base
    from train_talk_torch import load_adapter as load_torch_adapter

    tk = Tokenizer.from_file(os.path.join(args.export, "tokenizer.json"))
    prompts = [json.loads(l)["prompt"] for l, _ in zip(open(args.data, encoding="utf-8"), range(args.n))]
    ids = [tk.encode(p).ids for p in prompts]

    tm = load_base(args.ckpt, "cpu")
    with torch.no_grad():
        t_base = [tm.forward(torch.tensor([x])).float().numpy()[0] for x in ids]
    load_torch_adapter(tm, args.adapter)
    with torch.no_grad():
        t_lora = [tm.forward(torch.tensor([x])).float().numpy()[0] for x in ids]
    del tm

    gm, _ = load_grilly(args.export)
    load_grilly_adapter(gm, args.adapter)
    with grilly.no_grad():
        g_lora = [gm(grilly.tensor([x])).logits.numpy()[0] for x in ids]

    for i, (tb, tl, gl) in enumerate(zip(t_base, t_lora, g_lora)):
        scale = float(np.abs(tl).max())
        effect = float(np.abs(tl - tb).max())
        diff = float(np.abs(gl - tl).max())
        agree = float((gl.argmax(-1) == tl.argmax(-1)).mean())
        print(f"prompt {i} ({len(ids[i])} tokens): torch vs grilly2 with the adapter max |dlogit| {diff:.2e} "
              f"(scale {scale:.1f}); the adapter's own effect {effect:.2f}; argmax agrees at {agree:.4f}")


if __name__ == "__main__":
    main()
