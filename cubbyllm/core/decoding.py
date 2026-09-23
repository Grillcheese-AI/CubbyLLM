"""Text generation on ``CubbyModel.step``: O(1) per token, with the guards that
keep a small model from looping.

Wired: WIRED — ``CubbyModel.generate`` calls it.

WHY THIS EXISTS. Until now the only sampler was ``train_colab.sample_text``, a
training-time display helper that re-runs ``forward`` over a window for every
token (O(S) per token) and starts from EOS. Serving needs the opposite: continue a
prompt, carry the decode state, cost independent of context. And greedy decoding
of the 151M ``hd5_mem21`` checkpoint loops within ~15 tokens ("The history of the
city of Quebec is a story of the history of the city of Quebec. The history ...")
in both EN and FR. That is a decoding problem before it is a training problem, and
per the no-retraining rule the fix lives here, over frozen weights.

THE GUARDS are ``sample_text``'s, with the same semantics, so what
``validation/exp_gen_repetition.py`` measures is what this runs:

  repetition_penalty  frequency-aware CTRL penalty over GENERATED tokens: a logit
                      is divided (if > 0) or multiplied (if <= 0) by
                      ``penalty ** count``, so a tight loop is hit harder each time
                      round. 1.0 = off.
  no_repeat_ngram     a token that would complete an n-gram already present in
                      prompt + output is banned. Kept as a (n-1)-gram -> next-token
                      index, so it is O(1) per step rather than a rescan. 0 = off.

Then temperature (0 = greedy), top-k, top-p. If the guards ban every candidate
the head left (it can return top-K-masked logits), they are lifted for that step
rather than sampling from nothing.

WHERE THE GUARDS RUN. Both, and the split is deliberate.

The two guards are elementwise over the vocabulary, so they run on the
DEVICE: a (V,) count vector the penalty reads, and a padded index list the
n-gram ban writes ``-inf`` through. That is what turns a whole-vocabulary
readback per token -- 512 KB at V=127996 -- into a single token id.

Both are **bit-identical** to the numpy forms below, which stay as the
reference and as the ``device_guards=False`` path, and one detail is what
makes that true rather than nearly true: the penalty's ``penalty ** count``
comes from a table numpy built, gathered by the count, not from a device
``pow``. A device ``pow`` is a different implementation of the same
function and sits about an ulp away -- measured 3e-8 relative at penalty
1.3 -- which is a difference an argmax over 128k candidates can see. The
table is also cheaper than a pow over the whole vocabulary.

The CHOICE splits. Greedy is an argmax, deterministic and backend-
independent, so it runs on the device and the step reads back 8 bytes.
Sampling does NOT: the point of a numpy RNG seeded from the config is that
a cross-backend comparison compares models rather than samplers, and a
device RNG would end that. With ``top_k > 0`` the device still does the
work -- it returns the top-k values and indices, a few hundred bytes -- and
the draw happens on the host from exactly the candidate set the numpy path
would have built. With ``top_k = 0`` there is no candidate set to shrink to
and the whole vocabulary comes back, as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .protocols import Wiring

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DecodeConfig:
    """Decoding knobs. Defaults are ``train_colab.sample_text``'s guards, greedy."""

    max_new_tokens: int = 64
    temperature: float = 0.0          # 0 -> greedy argmax
    top_k: int = 40                   # 0 -> off (sampling only)
    top_p: float = 1.0                # 1.0 -> off (sampling only)
    repetition_penalty: float = 1.3   # 1.0 -> off
    no_repeat_ngram: int = 3          # 0 -> off
    eos_id: "int | None" = None       # stop when produced (not included in output)
    seed: int = 0
    device_guards: bool = True        # False -> the numpy reference path
    max_banned: int = 64              # padded n-gram ban width (device path)


class NGramBlock:
    """The (n-1)-gram -> {next token} index behind ``no_repeat_ngram``."""

    def __init__(self, n: int, ids: "list[int]") -> None:
        self.n = int(n)
        self.seen: dict[tuple, set] = {}
        if self.n > 0:
            for i in range(len(ids) - self.n + 1):
                self._add(ids[i:i + self.n])

    def _add(self, gram) -> None:
        self.seen.setdefault(tuple(gram[:-1]), set()).add(int(gram[-1]))

    def push(self, ids: "list[int]") -> None:
        """Record the n-gram ending at the last token of ``ids``."""
        if self.n > 0 and len(ids) >= self.n:
            self._add(ids[-self.n:])

    def banned(self, ids: "list[int]") -> "set[int]":
        if self.n <= 0 or len(ids) < self.n - 1:
            return set()
        return self.seen.get(tuple(ids[len(ids) - (self.n - 1):]), set())


def process(logits: np.ndarray, generated: "list[int]", banned: "set[int]",
            penalty: float) -> np.ndarray:
    """The two guards on one step's logits (a new array; the input is untouched)."""
    out = np.array(logits, dtype=np.float32, copy=True)
    if penalty and penalty != 1.0 and generated:
        idx, counts = np.unique(np.asarray(generated, dtype=np.int64), return_counts=True)
        factor = np.power(np.float32(penalty), counts.astype(np.float32))
        v = out[idx]
        out[idx] = np.where(v > 0, v / factor, v * factor)
    if banned:
        guarded = out.copy()
        guarded[np.fromiter(banned, dtype=np.int64)] = -np.inf
        if np.isfinite(guarded).any():      # never ban every surviving candidate
            out = guarded
    return out


def choose(logits: np.ndarray, cfg: DecodeConfig, rng: np.random.Generator) -> int:
    """Greedy when temperature is 0, else temperature -> top-k -> top-p -> draw."""
    if cfg.temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64)
    finite = np.flatnonzero(np.isfinite(z))
    k = min(cfg.top_k, finite.size) if cfg.top_k > 0 else finite.size
    cand = finite[np.argpartition(z[finite], -k)[-k:]] if k < finite.size else finite
    return _draw(cand, z[cand], cfg, rng)


def _host(t) -> np.ndarray:
    return t.detach().cpu().numpy()


class DeviceGuards:
    """The two guards and the greedy choice, as tensors that stay put.

    Holds a ``(V+1,)`` count vector for the repetition penalty and a
    fixed-width index list for the n-gram ban. Both are padded by one row:
    the ban list points its unused slots at row ``V``, so a step with two
    banned tokens and a step with twenty issue the same commands over the
    same buffers and nothing has to be reallocated or reshaped.

    Every operation here is one the reference above performs, in the same
    order, on the same values -- ``tests/core/test_decoding.py`` asserts
    the two agree bit for bit. What it buys is the readback: the logits
    stay on the device and the host learns one token id.
    """

    def __init__(self, vocab: int, cfg: DecodeConfig, device) -> None:
        import torch

        self.vocab, self.cfg = int(vocab), cfg
        self.counts = torch.zeros(self.vocab + 1, dtype=torch.int64, device=device)
        self.ban = torch.full((int(cfg.max_banned),), self.vocab,
                              dtype=torch.int64, device=device)
        self.device = device
        # penalty ** k for every count a run can reach, built by the same
        # numpy call the reference makes. A device `pow` would be a
        # DIFFERENT implementation of x**y and lands about an ulp away --
        # measured 3e-8 relative at penalty 1.3 -- which is a difference
        # the argmax can see. A gather from an exact table is both exact
        # and cheaper than a pow over the whole vocabulary.
        counts = np.arange(int(cfg.max_new_tokens) + 1, dtype=np.float32)
        self.factors = torch.from_numpy(
            np.power(np.float32(cfg.repetition_penalty), counts)).to(device)

    def observe(self, tok: int) -> None:
        """Record a generated token for the frequency-aware penalty."""
        self.counts[tok] += 1

    def _banned(self, banned: "set[int]"):
        """The banned ids, padded to a fixed width with the spare row.

        Overflow is truncation, not an error: ``max_banned`` bounds the
        work per token, and a prefix that has already banned 64 successors
        of the same (n-1)-gram is not one more ban away from looping."""
        import torch

        self.ban.fill_(self.vocab)
        ids = list(banned)[: self.ban.shape[0]]
        if ids:
            self.ban[: len(ids)] = torch.tensor(ids, dtype=torch.int64, device=self.device)
        return self.ban

    def apply(self, logits):
        """``process`` on the device. logits: (V,) -> (V,)."""
        import torch

        out = logits
        penalty = self.cfg.repetition_penalty
        if penalty and penalty != 1.0:
            factor = self.factors[self.counts[: self.vocab]]
            out = torch.where(out > 0, out / factor, out * factor)
        return out

    def guard(self, penalized, banned: "set[int]"):
        """The n-gram ban, and the rule that it never bans everything."""
        import torch

        if self.cfg.no_repeat_ngram <= 0 or not banned:
            return penalized
        wide = torch.cat([penalized, torch.zeros(1, device=penalized.device)])
        wide = wide.index_fill(0, self._banned(banned), float("-inf"))
        guarded = wide[: self.vocab]
        # never ban every surviving candidate -- decided on the device, so
        # the logits still do not come back
        return torch.where(torch.isfinite(guarded).any(), guarded, penalized)

    def choose(self, logits, rng: np.random.Generator) -> int:
        """The chosen id, reading back as little as the choice allows."""
        import torch

        if self.cfg.temperature <= 0:
            return int(_host(torch.argmax(logits)))
        if self.cfg.top_k <= 0:
            return choose(_host(logits), self.cfg, rng)   # nothing to shrink to
        top = torch.topk(logits, min(self.cfg.top_k, self.vocab))
        values, idx = _host(top.values), _host(top.indices)
        finite = np.isfinite(values)
        return int(_draw(idx[finite], values[finite].astype(np.float64), self.cfg, rng))


def _draw(cand: np.ndarray, z: np.ndarray, cfg: DecodeConfig,
          rng: np.random.Generator) -> int:
    """Temperature -> top-p -> draw, over an already-chosen candidate set.

    Split out of :func:`choose` so the device path can hand it the top-k
    the device selected and get the same answer the whole-vocabulary path
    would have given."""
    z = z / cfg.temperature
    order = np.argsort(-z)
    cand, z = cand[order], z[order]
    p = np.exp(z - z[0])
    p /= p.sum()
    if cfg.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(p), cfg.top_p) + 1)
        cand, p = cand[:keep], p[:keep] / p[:keep].sum()
    return int(cand[rng.choice(cand.size, p=p)])


def generate(model, prompt_ids: "list[int]", cfg: DecodeConfig = DecodeConfig(),
             state: "dict | None" = None,
             on_token: "Callable[[int], None] | None" = None,
             step: "Callable | None" = None) -> "tuple[list[int], dict]":
    """Continue ``prompt_ids`` with ``model.step``; returns (new token ids, state).

    The returned state has consumed the whole prompt and every returned token (an
    EOS that stopped generation is not consumed), so passing it back with the next
    turn as ``prompt_ids`` resumes the conversation from the carried
    recurrent/KV/episodic state instead of re-reading its history. The n-gram
    guard only sees this call's prompt and output. ``on_token`` is called with
    each new id as it is chosen (streaming).

    ``step`` replaces ``model.step`` when given. It exists for one caller:
    ``CubbyModel.generate`` passes the step through ``ops.graph_step``, so
    a backend with graph capture records it once and replays it. This
    module stays backend-agnostic — it takes a callable and does not ask
    what wrapped it.
    """
    import torch

    step = model.step if step is None else step
    if state is None and hasattr(model, "init_state"):
        # allocate before the loop: a capture cannot contain an allocation
        # whose buffers the replay would have to make again
        state = model.init_state()
    prompt = [int(t) for t in prompt_ids]
    if not prompt:
        raise ValueError("generate needs at least one prompt token")
    rng = np.random.default_rng(cfg.seed)
    block = NGramBlock(cfg.no_repeat_ngram, prompt)
    ids, out = list(prompt), []
    guards = None
    with torch.no_grad():
        for tok in prompt[:-1]:                               # prefill, one step each
            _, state = step(torch.tensor([tok]), state)
        pending = prompt[-1]                                  # chosen, not yet consumed
        for _ in range(int(cfg.max_new_tokens)):
            logits, state = step(torch.tensor([pending]), state)
            pending = None
            if cfg.device_guards:
                if guards is None:
                    guards = DeviceGuards(logits.shape[-1], cfg, logits.device)
                row = guards.guard(guards.apply(logits[0]), block.banned(ids))
                tok = guards.choose(row, rng)
            else:
                lg = process(_host(logits)[0], out, block.banned(ids), cfg.repetition_penalty)
                tok = choose(lg, cfg, rng)
            if cfg.eos_id is not None and tok == cfg.eos_id:
                break
            ids.append(tok)
            out.append(tok)
            block.push(ids)
            if guards is not None:
                guards.observe(tok)
            pending = tok
            if on_token is not None:
                on_token(tok)
        if pending is not None:                               # keep the state whole
            _, state = step(torch.tensor([pending]), state)
    return out, state


__wiring__ = Wiring.WIRED
