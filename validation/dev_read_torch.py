"""dev_read_torch - order 0078's dev read on a CUDA card (Colab): the torch base and the emitter adapter behind the step
loop's emitter protocol, with a batched decode for the 16 tries (`exp_0078_dev_read.py --ckpt`).

**Why the batching is by length.** The model's decode state (`CubbyModel.step`) keeps its position, ring slot, ring
liveness and token count as ONE value for the whole batch, so rows can share a batch only when they have read the same
number of tokens. `emit_rows` therefore (1) reads the prompts' longest common token prefix once, at batch 1 (the
question, written the same for every try); (2) copies that state to each bucket of rows whose prompts have the same
length; (3) reads each bucket's remaining tokens and samples its rows together. That is exact: each row sees exactly
its own prompt, as `train_school_torch.sample_programs` gives it (pinned in
tests/validation/test_dev_read_torch.py, greedy, against per-row decoding). Before (2026-11-24, a T4): the tries were
decoded one prompt group at a time, each prompt read token by token, 286 s for one question's 16 tries.

Wiring: STANDALONE (order 0078's read).
"""
from __future__ import annotations

import os

__wiring__ = "STANDALONE"


def clone_state(x, batch: int):
    """A decode state copied for ``batch`` rows: float tensors with a batch dimension of 1 are repeated; the shared
    counters (position, slot, token count, liveness) are copied as they are."""
    import torch
    if isinstance(x, dict):
        return {k: clone_state(v, batch) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_state(v, batch) for v in x)
    if torch.is_tensor(x):
        if x.is_floating_point() and x.dim() >= 2 and x.shape[0] == 1:
            return x.repeat(batch, *([1] * (x.dim() - 1))).contiguous()
        return x.clone()
    return x


def decode(model, state, cur, eos: int, max_new: int, temperature: float, top_k: int) -> list[list[int]]:
    """Rows from ``cur`` (each row's last prompt token) until </s>: the loop of `sample_programs`."""
    import torch
    k = cur.shape[0]
    out: list[list[int]] = [[] for _ in range(k)]
    done = torch.zeros(k, dtype=torch.bool, device=cur.device)
    for _ in range(max_new):
        logits, state = model.step(cur, state)
        logits = logits.float()
        if temperature > 0:
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        for i in (~done).nonzero().flatten().tolist():
            if int(nxt[i]) != eos:
                out[i].append(int(nxt[i]))
        done |= nxt == eos
        if bool(done.all()):
            break
        cur = nxt
    return out


def rows_ids(model, ids: list[list[int]], eos: int, max_new: int, temperature: float, top_k: int, dev,
             cache: "list | None" = None) -> list:
    """Each prompt's continuation (token ids): the common prefix read once, then one batch per prompt length.
    ``cache`` (a list, kept by the caller): [(prefix ids, state)] of earlier calls; the longest one that starts this
    call's prefix is continued instead of reading it again (the question, between a question's steps)."""
    import torch
    n = min(len(i) for i in ids)
    lcp = 0
    while lcp < n - 1 and all(i[lcp] == ids[0][lcp] for i in ids):
        lcp += 1
    out: list = [None] * len(ids)
    with torch.no_grad():
        head, st = (), None
        for pre, s0 in (cache or []):
            if len(pre) <= lcp and len(pre) > len(head) and tuple(ids[0][:len(pre)]) == pre:
                head, st = pre, s0
        st = clone_state(st, 1) if st is not None else model.init_state(1, dev)
        marks = {lcp}                                   # snapshot where this prefix leaves each cached one too
        for pre, _ in (cache or []):
            c = 0
            while c < min(len(pre), lcp) and pre[c] == ids[0][c]:
                c += 1
            if c > len(head):
                marks.add(c)
        for p in range(len(head), lcp):
            _, st = model.step(torch.tensor([ids[0][p]], device=dev), st)
            if cache is not None and p + 1 in marks and not any(len(k) == p + 1 and k == tuple(ids[0][:p + 1])
                                                                for k, _ in cache):
                cache.append((tuple(ids[0][:p + 1]), clone_state(st, 1)))
        if cache is not None:
            del cache[:-8]
        buckets: dict = {}
        for j, i in enumerate(ids):
            buckets.setdefault(len(i), []).append(j)
        for length, js in sorted(buckets.items()):
            s = clone_state(st, len(js))
            for p in range(lcp, length - 1):
                _, s = model.step(torch.tensor([ids[j][p] for j in js], device=dev), s)
            cur = torch.tensor([ids[j][length - 1] for j in js], device=dev)
            for j, r in zip(js, decode(model, s, cur, eos, max_new, temperature, top_k)):
                out[j] = r
    return out


class TorchStepEmitter:
    """`--ckpt`: the torch base and the emitter adapter. `emit` is greedy (`sample_programs`, as the school's gate);
    `emit_rows` samples one continuation per prompt, all prompts in as few batches as their lengths allow."""

    def __init__(self, ckpt: str, adapter: str, tokenizer: str):
        import torch
        from tokenizers import Tokenizer
        from train_base import load_base
        from train_talk_torch import load_adapter
        self.torch, self.dev = torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tk = Tokenizer.from_file(tokenizer)
        self.eos = self.tk.token_to_id("</s>")
        self.model = load_base(ckpt, str(self.dev))
        load_adapter(self.model, adapter, "emitter_lora")      # load_base already placed it on dev
        self.name = f"torch:{os.path.basename(os.path.normpath(adapter))}"

        self.cache: list = []

    def emit(self, prompt: str, max_new_tokens: int = 96, **kw) -> str:
        """Greedy, one prompt (the greedy chain): `rows_ids` at temperature 0, the question's state reused."""
        from emitter_data import decode_program
        ids = self.tk.encode(prompt).ids
        row = rows_ids(self.model, [ids], self.eos, max_new_tokens, 0.0, 0, self.dev, self.cache)[0]
        return decode_program(self.tk, row, self.eos)

    def emit_rows(self, prompts: list[str], max_new_tokens: int = 96, temperature: float = 0.7, top_k: int = 40,
                  seed: int | None = None) -> list[str]:
        from emitter_data import decode_program
        if seed is not None:
            self.torch.manual_seed(int(seed))
        ids = [self.tk.encode(p).ids for p in prompts]
        rows = rows_ids(self.model, ids, self.eos, max_new_tokens, temperature, top_k, self.dev, self.cache)
        return [decode_program(self.tk, r, self.eos) for r in rows]


__all__ = ["TorchStepEmitter", "clone_state", "decode", "rows_ids"]
