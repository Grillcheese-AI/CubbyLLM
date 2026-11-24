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

**CUDA graphs (`GraphStep`, 2026-11-24).** On an L4 the batched decode still took ~75 s a question: a token is one
`model.step`, thousands of small kernels launched from Python, and `CubbyModel.graphed_step` replays only on grilly2
(`ops.graph_step` is the identity on torch). The step is built to be captured (fixed-size buffers, device counters), so
on CUDA each batch size (rounded up to a power of two; the padding rows are copies of row 0 and are never sampled) is
recorded once and replayed. Each recording is checked against the eager step when it is made; a capture that fails or
differs is dropped and that size runs eagerly (printed). `CB_DEV_READ_GRAPHS=0` turns it off.

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


def copy_into(dst, src) -> None:
    """``src`` written into ``dst``'s buffers (same structure); a batch-1 row is broadcast to ``dst``'s rows, a smaller
    batch fills the first rows and row 0 the rest."""
    import torch
    if isinstance(dst, dict):
        for k in dst:
            copy_into(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for d, s in zip(dst, src):
            copy_into(d, s)
    elif torch.is_tensor(dst):
        if dst.shape == src.shape:
            dst.copy_(src)
        else:
            n = src.shape[0]
            dst[:n].copy_(src)
            dst[n:].copy_(src[:1].expand_as(dst[n:]))


class Eager:
    """The plain step: ``start`` takes a state (batch 1 or n), ``step`` returns the n rows' logits."""

    def __init__(self, model):
        self.model = model

    def start(self, state, n: int):
        self.st = clone_state(state, n)
        return self

    def step(self, tok):
        logits, self.st = self.model.step(tok, self.st)
        return logits

    def state(self):
        return self.st


class GraphStep:
    """`model.step` recorded once per batch size (a power of two) and replayed; see the module docstring."""

    def __init__(self, model, dev):
        self.model, self.dev, self.g, self.off = model, dev, {}, set()
        self.eager = Eager(model)

    def _pair(self, out, st):
        import torch
        if isinstance(st, dict):
            for k in st:
                self._pair(out[k], st[k])
        elif isinstance(st, (list, tuple)):
            for o, s in zip(out, st):
                self._pair(o, s)
        elif torch.is_tensor(st) and out is not st:
            st.copy_(out)                                   # the step's new buffer back into the recorded one

    def _capture(self, B: int):
        import torch
        st, tok = self.model.init_state(B, self.dev), torch.zeros(B, dtype=torch.long, device=self.dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _, out = self.model.step(tok, st)
                self._pair(out, st)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits, out = self.model.step(tok, st)
            self._pair(out, st)
        ref = self.model.init_state(B, self.dev)            # the check: four tokens, eager against replayed
        copy_into(st, self.model.init_state(B, self.dev))
        diff = 0.0
        for t in (5, 9, 13, 17):
            tok.fill_(t)
            g.replay()
            want, ref = self.model.step(torch.full((B,), t, dtype=torch.long, device=self.dev), ref)
            diff = max(diff, float((logits.float() - want.float()).abs().max()))
        if diff > 1e-3:
            raise RuntimeError(f"replay differs from the eager step by {diff:.2e}")
        return g, st, tok, logits

    def start(self, state, n: int):
        B = 1 << max(0, (n - 1).bit_length())
        if B not in self.g and B not in self.off:
            try:
                with __import__("torch").no_grad():
                    self.g[B] = self._capture(B)
                print(f"  [graphs] batch {B} recorded", flush=True)
            except Exception as e:                          # noqa: BLE001 - any capture failure: eager for this size
                self.off.add(B)
                print(f"  [graphs] batch {B} eager: {type(e).__name__}: {str(e)[:120]}", flush=True)
        if B in self.off:
            return self.eager.start(state, n)
        self.n, (self.graph, self.st, self.tok, self.logits) = n, self.g[B]
        copy_into(self.st, state)
        return self

    def step(self, tok):
        self.tok[:self.n].copy_(tok)
        self.tok[self.n:].copy_(tok[:1].expand(self.tok.shape[0] - self.n))
        self.graph.replay()
        return self.logits[:self.n]

    def state(self):
        return self.st


def decode(live, cur, eos: int, max_new: int, temperature: float, top_k: int) -> list[list[int]]:
    """Rows from ``cur`` (each row's last prompt token) until </s>: the loop of `sample_programs`. ``live`` is a started
    `Eager` / `GraphStep`."""
    import torch
    k = cur.shape[0]
    toks, done = [], torch.zeros(k, dtype=torch.bool, device=cur.device)
    for _ in range(max_new):
        logits = live.step(cur).float()
        if temperature > 0:
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        toks.append(nxt)
        done |= nxt == eos
        if bool(done.all()):
            break
        cur = nxt
    if not toks:
        return [[] for _ in range(k)]
    grid = torch.stack(toks, 1).tolist()
    return [r[:r.index(eos)] if eos in r else r for r in grid]


def rows_ids(model, ids: list[list[int]], eos: int, max_new: int, temperature: float, top_k: int, dev,
             cache: "list | None" = None, stepper=None) -> list:
    """Each prompt's continuation (token ids): the common prefix read once, then one batch per prompt length.
    ``cache`` (a list, kept by the caller): [(prefix ids, state)] of earlier calls; the longest one that starts this
    call's prefix is continued instead of reading it again (the question, between a question's steps).
    ``stepper``: `GraphStep` on CUDA; `Eager` by default."""
    import torch
    stepper = stepper or Eager(model)
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
        live = stepper.start(st if st is not None else model.init_state(1, dev), 1)
        marks = {lcp}                                   # snapshot where this prefix leaves each cached one too
        for pre, _ in (cache or []):
            c = 0
            while c < min(len(pre), lcp) and pre[c] == ids[0][c]:
                c += 1
            if c > len(head):
                marks.add(c)
        for p in range(len(head), lcp):
            live.step(torch.tensor([ids[0][p]], device=dev))
            if cache is not None and p + 1 in marks and not any(len(k) == p + 1 and k == tuple(ids[0][:p + 1])
                                                                for k, _ in cache):
                cache.append((tuple(ids[0][:p + 1]), clone_state(live.state(), 1)))
        st = clone_state(live.state(), 1)
        if cache is not None:
            del cache[:-8]
        buckets: dict = {}
        for j, i in enumerate(ids):
            buckets.setdefault(len(i), []).append(j)
        for length, js in sorted(buckets.items()):
            live = stepper.start(st, len(js))
            for p in range(lcp, length - 1):
                live.step(torch.tensor([ids[j][p] for j in js], device=dev))
            cur = torch.tensor([ids[j][length - 1] for j in js], device=dev)
            for j, r in zip(js, decode(live, cur, eos, max_new, temperature, top_k)):
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
        graphs = self.dev.type == "cuda" and os.environ.get("CB_DEV_READ_GRAPHS", "1") != "0"
        self.stepper = GraphStep(self.model, self.dev) if graphs else Eager(self.model)
        self.decode_kind = "rows-graphs" if graphs else "rows"
        self.cache: list = []

    def emit(self, prompt: str, max_new_tokens: int = 96, **kw) -> str:
        """Greedy, one prompt (the greedy chain): `rows_ids` at temperature 0, the question's state reused."""
        from emitter_data import decode_program
        ids = self.tk.encode(prompt).ids
        row = rows_ids(self.model, [ids], self.eos, max_new_tokens, 0.0, 0, self.dev, self.cache, self.stepper)[0]
        return decode_program(self.tk, row, self.eos)

    def emit_rows(self, prompts: list[str], max_new_tokens: int = 96, temperature: float = 0.7, top_k: int = 40,
                  seed: int | None = None) -> list[str]:
        from emitter_data import decode_program
        if seed is not None:
            self.torch.manual_seed(int(seed))
        ids = [self.tk.encode(p).ids for p in prompts]
        rows = rows_ids(self.model, ids, self.eos, max_new_tokens, temperature, top_k, self.dev, self.cache,
                        self.stepper)
        return [decode_program(self.tk, r, self.eos) for r in rows]


__all__ = ["Eager", "GraphStep", "TorchStepEmitter", "clone_state", "copy_into", "decode", "rows_ids"]
