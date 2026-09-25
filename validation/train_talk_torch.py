"""Train the talk adapter in torch (Colab's A100; CPU for a smoke), LoRA over the frozen 450M base.

The same job, data, loss and adapter file as `train_talk_grilly.py` -- the shared parts are in
`talk_data.py` -- so an adapter trained here loads into grilly2 on the local card for the gate
(`exp_e6_talk_gate.py`) and for serving, and one trained there loads here:

- LoRA on every projection of every layer (attention qkv + output, MinGRU gate / value / decay, the
  FFN's three), y = W x + b + (alpha / r) * B (A x), A as nn.Linear's default init, B zero;
- loss on the answer tokens only (the answer and its </s>), the head applied to those positions only;
- AdamW, warmup then cosine to 10%, gradient norm clipped at 1.0, bf16 autocast on CUDA;
- the adapter file keyed by grilly2's module paths (`model.layers.{i}.mixer.qkv.lora_A.weight`).

    python validation/train_talk_torch.py --ckpt <base450m_final.pt> --tokenizer <bbpe128k.json> \
        --data <ground_sft.jsonl> --out <adapter dir> [--epochs 2] [--batch 32] [--tag _colab]
    python validation/train_talk_torch.py --smoke --tokenizer <bbpe128k.json> --data <ground_sft.jsonl>
                                            # a tiny random base on CPU: the code path, not a result
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

if "--smoke" in sys.argv:                    # train_base reads its shape from the environment at import
    os.environ.update(CB_D="64", CB_L="4", CB_HEADS="4", CB_WINDOW="16", CB_GEN_BASIS="4", CB_GEN_RANK="2")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from talk_data import batches, encode_records, lora_targets, make_batch, torch_attr  # noqa: E402

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus (alpha / r) * B (A x) -- grilly.infer.lora.LoRALinear's arithmetic."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.lora_B.weight)     # the adapter starts as the identity
        self.scaling = float(alpha) / float(rank)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(model, targets, rank, alpha) -> dict:
    wrapped = {}
    for t in targets:
        i, group, attr = torch_attr(t)
        owner = getattr(model.backbone, group)[i]
        base = getattr(owner, attr)
        if isinstance(base, LoRALinear):
            raise RuntimeError(f"{t} already carries a LoRA")
        w = LoRALinear(base, rank, alpha)
        setattr(owner, attr, w)
        wrapped[t] = w
    return wrapped


def save_adapter(wrapped, out_dir, meta) -> None:
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for t, w in wrapped.items():
        tensors[f"{t}.lora_A.weight"] = w.lora_A.weight.detach().float().cpu().numpy()
        tensors[f"{t}.lora_B.weight"] = w.lora_B.weight.detach().float().cpu().numpy()
    save_file(tensors, os.path.join(out_dir, "talk_lora.safetensors"))
    with open(os.path.join(out_dir, "talk_lora.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def load_adapter(model, adapter_dir) -> dict:
    """An adapter file (from either trainer) into the torch model; returns its json."""
    from safetensors.numpy import load_file
    meta = json.load(open(os.path.join(adapter_dir, "talk_lora.json"), encoding="utf-8"))
    wrapped = apply_lora(model, meta["targets"], meta["rank"], meta["alpha"])
    tensors = load_file(os.path.join(adapter_dir, "talk_lora.safetensors"))
    with torch.no_grad():
        for t, w in wrapped.items():
            w.lora_A.weight.copy_(torch.from_numpy(tensors[f"{t}.lora_A.weight"]))
            w.lora_B.weight.copy_(torch.from_numpy(tensors[f"{t}.lora_B.weight"]))
    return meta


def masked_loss(model, x, y, m, dev, amp):
    """Mean cross-entropy over the answer tokens; the head sees those positions only."""
    x, y, sel = (torch.from_numpy(a).to(dev) for a in (x, y, m > 0))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h = model.features(x)                              # (B, T, D)
        logits = model.logits_from(h[sel].unsqueeze(0))[0]  # (N, V)
    return F.cross_entropy(logits.float(), y[sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=320)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = dev.type == "cuda"
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    eos = tk.token_to_id("</s>")
    t0 = time.time()
    if args.smoke:
        from train_base import arch_meta, build_model
        meta = arch_meta(tk.get_vocab_size(), ((tk.get_vocab_size() + 127) // 128) * 128)
        model = build_model(meta, dev)
        args.steps = args.steps or 3
        args.batch, args.eval_n = 2, 4
    else:
        from train_base import load_base
        model = load_base(args.ckpt, str(dev))
        meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)["meta"]
    for p in model.parameters():
        p.requires_grad_(False)
    targets = lora_targets(meta["L"], meta["attn_every"])
    wrapped = apply_lora(model, targets, args.rank, args.alpha)
    params = [p for w in wrapped.values() for p in (w.lora_A.weight, w.lora_B.weight)]
    for p in params:
        p.requires_grad_(True)

    train = encode_records(args.data, tk, eos, "train", args.max_len)
    held = encode_records(args.data, tk, eos, "held", args.max_len)
    rng.shuffle(held)
    held = held[:args.eval_n]
    if args.smoke:
        train = train[:16]
    total = args.steps or int(math.ceil(args.epochs * len(train) / args.batch))
    log(f"train_talk_torch | {dev} | {len(targets)} LoRA modules, {sum(p.numel() for p in params) / 1e6:.2f}M "
        f"trainable (r={args.rank}, alpha={args.alpha}) | train {len(train):,} held-eval {len(held)} | "
        f"{total} steps of {args.batch} | lr {args.lr} warmup {args.warmup} | load {time.time() - t0:.0f}s")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, **({"fused": True} if amp else {}))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    def evaluate():
        tot, n = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(held), args.batch):
                x, y, m, _ = make_batch(held[s:s + args.batch])
                k = float(m.sum())
                tot += float(masked_loss(model, x, y, m, dev, amp)) * k
                n += k
        return tot / max(n, 1.0)

    def meta_out(step):
        return {"step": step, "rank": args.rank, "alpha": args.alpha, "targets": targets, "trainer": "torch",
                "ckpt": os.path.basename(args.ckpt) if args.ckpt else "smoke", "data": os.path.basename(args.data),
                "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "tag": args.tag}

    logs = os.path.join(HERE, "logs")
    os.makedirs(logs, exist_ok=True)
    mf = open(os.path.join(logs, f"train_talk_torch{args.tag}.jsonl"), "w", encoding="utf-8")
    ev = evaluate()
    log(f"  step 0 held answer loss {ev:.4f}")
    mf.write(json.dumps({"step": 0, "held_loss": ev}) + "\n")
    order, step, tokens, t_start, run = [], 0, 0, time.time(), []
    while step < total:
        if not order:
            order = batches(train, args.batch, rng)
        x, y, m, _ = make_batch([train[i] for i in order.pop()])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = masked_loss(model, x, y, m, dev, amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            log(f"  step {step}: loss {lv} -- stopping without saving")
            break
        run.append(lv)
        tokens += int((x != 0).sum())
        step += 1
        if step % 20 == 0 or step == total:
            el = time.time() - t_start
            mf.write(json.dumps({"step": step, "loss": float(np.mean(run)), "gnorm": gn, "lr": lr_at(step - 1),
                                 "s_per_step": el / step, "tok_s": tokens / el}) + "\n")
            mf.flush()
            log(f"  step {step:5d}/{total}  loss {np.mean(run):.4f}  gnorm {gn:.3f}  lr {lr_at(step - 1):.2e}  "
                f"{el / step:.2f} s/step  {tokens / el:,.0f} tok/s  eta {(total - step) * el / step / 60:.1f} min")
            run = []
        if step % args.eval_every == 0 or step == total:
            ev = evaluate()
            log(f"  step {step} held answer loss {ev:.4f}")
            mf.write(json.dumps({"step": step, "held_loss": ev}) + "\n")
        if args.out and (step % args.save_every == 0 or step == total):
            save_adapter(wrapped, args.out, meta_out(step))
    mf.close()
    with open(os.path.join(logs, f"train_talk_torch{args.tag}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    log(f"done: {step} steps in {(time.time() - t_start) / 60:.1f} min" + (f" -> {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
