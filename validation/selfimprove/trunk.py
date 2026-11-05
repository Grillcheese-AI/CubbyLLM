"""validation/selfimprove/trunk.py -- order 0049: the frozen 450M with one set of LoRA modules whose weights are swapped
per adapter (the base adapter, a probe, a gap's adapter), sampling batched across prompts, the held loss, and training.

  sampling   every row of a batch starts at the first token; a row feeds its own prompt token at each step until its
             prompt is read, then its own choices. No padding token ever enters a row's recurrent state, so prompts of
             any length share a batch (the step school's ``sample_programs`` batches one prompt only)
  training   SFT on (prompt, right program), from the weights in place. With negatives (T3's arms B and C), half of
             each batch's rows are wrong programs, and their loss is unlikelihood, -log(1 - p), on the wrong program's
             tokens from its first divergence from the problem's right program only: the shared prefix is right and is
             not pushed down (Cowork, 2026-11-04). Equal compute across arms: the same steps, the same rows a step.
  the trunk  frozen; only the LoRA weights move (0049's constraint)

Wired: STANDALONE (order 0049's harness).
"""
from __future__ import annotations

import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.dirname(HERE)
for _p in (VAL, os.path.dirname(VAL), os.path.join(os.path.dirname(VAL), "standin", "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__wiring__ = "STANDALONE"

ADAPTER = "emitter_lora"


def first_divergence(wrong: list[int], right: list[int]) -> int:
    """The first token index where the wrong program leaves the right one (both end in </s>)."""
    for i, (a, b) in enumerate(zip(wrong, right)):
        if a != b:
            return i
    return min(len(wrong), len(right))


def sample_many(model, prompts: list[list[int]], k: int, temperature: float, caps: list[int], eos: int, dev,
                top_k: int = 50, gen=None) -> list[list[list[int]]]:
    """k programs for each prompt (token ids), all rows in one batch; ``caps``: each prompt's token budget."""
    import torch
    seqs = [p for p in prompts for _ in range(k)]
    cap = torch.tensor([c for c in caps for _ in range(k)], device=dev)
    B, T = len(seqs), max(len(s) for s in seqs)
    L = torch.tensor([len(s) for s in seqs], device=dev)
    P = torch.full((B, T), eos, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        P[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    out = torch.full((B, int(cap.max())), eos, dtype=torch.long, device=dev)
    n = torch.zeros(B, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    state, cur = model.init_state(B, dev), P[:, 0]
    with torch.no_grad():
        for t in range(T - 1 + int(cap.max())):
            logits, state = model.step(cur, state)
            reading = (t + 1) < L
            live = ~reading & ~done
            lg = logits.float()
            if temperature > 0:
                if top_k:
                    kth = torch.topk(lg, min(top_k, lg.shape[-1]), dim=-1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(lg / temperature, dim=-1), 1, generator=gen).squeeze(1)
            else:
                nxt = lg.argmax(-1)
            stop = live & (nxt == eos)
            put = live & ~stop
            idx = put.nonzero().flatten()
            out[idx, n[idx]] = nxt[idx]
            n += put.long()
            done |= stop | (live & (n >= cap))
            if bool(done.all()):
                break
            cur = torch.where(reading, P[:, min(t + 1, T - 1)], torch.where(done, torch.full_like(nxt, eos), nxt))
    rows = [out[i, :int(n[i])].tolist() for i in range(B)]
    return [rows[j * k:(j + 1) * k] for j in range(len(prompts))]


class Trunk:
    """The base checkpoint, frozen, with the base adapter's LoRA modules; ``state()`` / ``use()`` swap their weights."""

    def __init__(self, ckpt: str, tokenizer: str, base_adapter: str, dev: str | None = None, seed: int = 0,
                 model=None, tk=None) -> None:
        import torch
        self.dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = self.dev.type == "cuda"
        if self.amp:
            torch.backends.cuda.matmul.allow_tf32 = True
        if tk is None:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(tokenizer)
        self.tk, self.eos = tk, tk.token_to_id("</s>")
        if model is None:
            from train_base import load_base
            from train_talk_torch import load_adapter
            model = load_base(ckpt, str(self.dev))
            for p in model.parameters():
                p.requires_grad_(False)
            self.meta = load_adapter(model, base_adapter, ADAPTER)
        else:                                            # tests: a model that already carries its LoRA modules
            self.meta = {"targets": list(model.lora_targets), "rank": 1, "alpha": 1.0}
        self.model = model
        from train_emitter_torch import _find
        self.wrapped = {t: (_find(model, t) if not hasattr(model, "lora_find") else model.lora_find(t))
                        for t in self.meta["targets"]}
        self.params = [p for w in self.wrapped.values() for p in (w.lora_A.weight, w.lora_B.weight)]
        self.base = self.state()
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.rng = random.Random(seed)

    # -- the adapters -------------------------------------------------------------------------------------------------
    def state(self) -> list:
        return [p.detach().clone() for p in self.params]

    def use(self, st: list) -> None:
        import torch
        with torch.no_grad():
            for p, s in zip(self.params, st):
                p.copy_(s)

    def save(self, st: list, out_dir: str, meta: dict) -> None:
        from train_talk_torch import save_adapter
        keep = self.state()
        self.use(st)
        save_adapter(self.wrapped, out_dir, {**self.meta, **meta, "adapter": ADAPTER}, ADAPTER)
        self.use(keep)

    # -- generation ---------------------------------------------------------------------------------------------------
    def programs(self, prompts: list[str], caps: list[int], k: int = 1, temperature: float = 0.0,
                 rows: int = 64) -> list[list[str]]:
        """k programs a prompt (greedy when temperature is 0), batched by length; slotted text, as emitted."""
        from emitter_data import decode_program
        enc = [self.tk.encode(p).ids for p in prompts]
        order = sorted(range(len(enc)), key=lambda i: (caps[i], len(enc[i])))
        per, res = max(1, rows // k), [None] * len(enc)
        for s in range(0, len(order), per):
            ids = order[s:s + per]
            got = sample_many(self.model, [enc[i] for i in ids], k, temperature, [caps[i] for i in ids], self.eos,
                              self.dev, gen=self.gen)
            for i, g in zip(ids, got):
                res[i] = [decode_program(self.tk, o, self.eos) for o in g]
        return res

    # -- loss and training --------------------------------------------------------------------------------------------
    def encode(self, prompt: str, program: str) -> dict:
        return {"p": self.tk.encode(prompt).ids, "t": self.tk.encode(program.rstrip() + "\n").ids + [self.eos]}

    def negative(self, prompt: str, wrong: str, right: str) -> dict | None:
        row = self.encode(prompt, wrong)
        d = first_divergence(row["t"], self.encode(prompt, right)["t"])
        return None if d >= len(row["t"]) else {**row, "d": d}

    def _ce(self, rows):
        from talk_data import make_batch
        from train_talk_torch import masked_loss
        x, y, m, _ = make_batch(rows)
        return masked_loss(self.model, x, y, m, self.dev, self.amp), float(m.sum())

    def _ul(self, rows):
        import torch
        from talk_data import make_batch
        x, y, m, _ = make_batch(rows)
        for i, r in enumerate(rows):                       # only the wrong program's tokens from its divergence on
            m[i, :len(r["p"]) - 1 + r["d"]] = 0.0
        xt, yt, sel = (torch.from_numpy(a).to(self.dev) for a in (x, y, m > 0))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            h = self.model.features(xt)
            logits = self.model.logits_from(h[sel].unsqueeze(0))[0]
        p = torch.softmax(logits.float(), -1).gather(1, yt[sel][:, None])[:, 0]
        return -torch.log1p(-p.clamp(max=1 - 1e-6)).mean()

    def loss(self, items: list[tuple[str, str]], batch: int = 16) -> float:
        """Mean token cross-entropy of (prompt, program) items under the weights in place."""
        import torch
        rows, tot, n = [self.encode(p, t) for p, t in items], 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                lv, k = self._ce(rows[s:s + batch])
                tot, n = tot + float(lv) * k, n + k
        return tot / max(n, 1.0)

    def train(self, right: list[tuple[str, str]], negatives: list[tuple[str, str, str]] = (), steps: int = 100,
              batch: int = 16, lr: float = 2e-4, ul: float = 1.0, warmup: int = 10) -> dict:
        """From the weights in place: ``right`` (prompt, program); ``negatives`` (prompt, wrong, right)."""
        import torch
        pos = [self.encode(p, t) for p, t in right]
        neg = [r for r in (self.negative(*x) for x in negatives) if r is not None]
        if not pos:
            raise ValueError("no right program to train on")
        n_neg = batch // 2 if neg else 0
        opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        for p in self.params:
            p.requires_grad_(True)
        last = []
        for step in range(steps):
            f = (step + 1) / warmup if step < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
            for g in opt.param_groups:
                g["lr"] = lr * f
            ce, _ = self._ce(self.rng.choices(pos, k=batch - n_neg))
            loss = ce + (ul * self._ul(self.rng.choices(neg, k=n_neg)) if n_neg else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            opt.step()
            last.append(float(loss.detach()))
        for p in self.params:
            p.requires_grad_(False)
        return {"steps": steps, "batch": batch, "negative_rows": n_neg, "right": len(pos), "negatives": len(neg),
                "loss_last": round(sum(last[-10:]) / max(1, len(last[-10:])), 4)}


__all__ = ["ADAPTER", "Trunk", "first_divergence", "sample_many"]
