"""Shared helpers for the Group P (prospection) experiments.

Standalone by the validation/ rule: never imported by cubbyllm/. The sibling
scripts do ``sys.path.insert(0, THIS_DIR)`` and ``import _common``.

What lives here and why:
  * ``BranchGrammar`` — a toy corpus with GROUND-TRUTH choice points. After SEP
    the next token is a free choice among K branch tokens; every other token is
    deterministic given the branch. The whole Group P thread rests on "branch
    only where there is a real choice" being measurable, so the toy makes the
    choice points a known quantity instead of an assumption.
  * ``build_model`` / ``train_on`` / ``trained_toy`` — the smallest CubbyModel
    that exercises the REAL decode path (``CubbyModel.step``), CPU-cheap, plus a
    per-process cache so the five scripts (and the smoke test) train it once.
  * ``load_checkpoint`` — the ``{"meta", "params"}`` checkpoint layout that
    ``exp_needle_recall.py`` reads, so every script here also runs on a trained
    checkpoint via CB_CKPT (plus CUBBY_SPM and CB_TEXT / CB_CORPUS for tokens).
  * ``clone_state`` / ``prefix_state`` / ``continue_from`` — fork a decode state
    exactly (tensors clone; an EpisodicStore deep-copies) and roll it forward.
    Forking IS the operation H-P1 is about, and the primitive H-P4/H-P5 use.
  * ``retention_profile`` / ``time_constants`` / ``chrono_init_`` — per-unit
    MinGRU retention ``a`` and the implied time constant tau = -1/ln(a); a
    chrono-style log-uniform re-init of the retention-gate bias (Tallec &
    Ollivier 2018, "Can recurrent neural networks warp time?").
"""
from __future__ import annotations

import copy
import math
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:                                          # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                             # pragma: no cover - non-tty / captured
    pass

CKPT = os.environ.get("CB_CKPT", "")
SPM = os.environ.get("CUBBY_SPM", "")
TEXT = os.environ.get("CB_TEXT", "")          # plain-text file to tokenize (simplest real input)
CORPUS = os.environ.get("CB_CORPUS", "")      # token-cache dir (WeightedCorpusPipeline, as needle uses)
SOURCES = os.environ.get("CB_SOURCES", "")
CTX_DIM = int(os.environ.get("CB_CTX_DIM", "32"))

# mingru.py: a = 0.001 + 0.998 * sigmoid(proj_d(x)). Retention can never exceed
# A_MAX, so no unit can hold a value with a time constant longer than TAU_CAP —
# a property of the parameterization, measured and pinned in exp_p3.
A_MIN, A_MAX = 0.001, 0.999
TAU_CAP = -1.0 / math.log(A_MAX)              # ~999.5 steps


# ── toy corpus with ground-truth choice points ──────────────────────────────

class BranchGrammar:
    """[SEP, B_k, t_k1 .. t_kM] repeated, k drawn uniformly per segment.

    Token ids: SEP=0, branch tokens 1..K, content tokens K+1..K+C. All templates
    share their first ``shared`` content tokens; the token right after that
    prefix is DISTINCT per branch. So continuing a template correctly past the
    prefix requires the branch id to have survived in state — the previous
    token alone is ambiguous there — which makes the counterfactual probe
    (H-P5) a state-tracking test, not a bigram lookup. The only genuine choice
    in the whole stream is which B_k follows a SEP (entropy ln K); every other
    next-token is deterministic (entropy 0).
    """

    def __init__(self, n_branches: int = 4, template_len: int = 6,
                 content_vocab: int = 8, shared: int = 2, seed: int = 0):
        assert n_branches <= content_vocab and 0 <= shared < template_len
        rng = np.random.default_rng(seed)
        self.K, self.M, self.C = int(n_branches), int(template_len), int(content_vocab)
        self.shared = int(shared)
        self.SEP = 0
        self.branch_ids = list(range(1, 1 + self.K))
        self.content_off = 1 + self.K
        self.V = self.content_off + self.C
        prefix = rng.integers(0, self.C, size=self.shared)
        firsts = rng.permutation(self.C)[: self.K]           # distinct divergent token per branch
        self.templates = []
        for k in range(self.K):
            rest = rng.integers(0, self.C, size=max(0, self.M - self.shared - 1))
            body = np.concatenate([prefix, [firsts[k]], rest]).astype(np.int64)
            self.templates.append(body + self.content_off)
        self.seg_len = 2 + self.M
        # per-token entropy floor of the stream (nats/token): ln K once per segment
        self.entropy_floor = math.log(self.K) / self.seg_len

    def segment(self, k: int) -> np.ndarray:
        return np.concatenate([[self.SEP, 1 + k], self.templates[k]]).astype(np.int64)

    def sample(self, n_segments: int, rng) -> tuple[np.ndarray, np.ndarray]:
        ks = rng.integers(0, self.K, size=n_segments)
        return np.concatenate([self.segment(int(k)) for k in ks]), ks

    def choice_mask(self, tokens: np.ndarray) -> np.ndarray:
        """True at positions whose NEXT token is a free choice (input == SEP)."""
        m = np.zeros(len(tokens), dtype=bool)
        m[:-1] = tokens[:-1] == self.SEP
        return m


# ── models ───────────────────────────────────────────────────────────────────

def build_model(d: int = 32, L: int = 2, V: int = 16, backbone: str = "mingru",
                window: int = 8, heads: int = 4, ctx_dim: int = 8, seed: int = 0,
                mem_every: int = 0, mem_topk: int = 4, mem_key: int = 16):
    """The smallest CubbyModel with the real decode path. ``retrieval_k=V`` so
    logits are a FULL softmax (entropy over the whole vocab, not a top-K set).
    backbone: "mingru" (pure recurrence) or "hybrid" (windowed attention every
    3rd layer; ``mem_every>0`` adds the episodic MemoryRead, B=1 decode only)."""
    from cubbyllm.core.config import CubbyConfig
    from cubbyllm.core.context import FrozenSlotRouter
    from cubbyllm.core.generation import HyperGenerator, SnapshotHardener
    from cubbyllm.model.assembly import CubbyModel
    from cubbyllm.model.backbone import HybridBackbone, MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    torch.manual_seed(seed)
    if backbone == "hybrid":
        bb = HybridBackbone(d, L, attn_every=3, window=window, heads=heads,
                            mem_every=mem_every, mem_topk=mem_topk, mem_key=mem_key)
    elif backbone == "mingru":
        bb = MinGRUBackbone(d, L)
    else:
        raise ValueError(backbone)
    return CubbyModel(
        config=CubbyConfig(d_model=d, n_layers=L, ctx_dim=ctx_dim, vocab_core=V),
        context_source=FrozenSlotRouter(input_dim=d, n_slots=4, ctx_dim=ctx_dim).freeze(),
        backbone=bb,
        memory=MemoryLayer(HyperGenerator(ctx_dim=ctx_dim, n_out=d * d), SnapshotHardener(),
                           d_model=d),
        binding=BindingHead(),
        embedding=HybridEmbedding(V, d),
        head=TopKRetrievalHead(torch.randn(V, d) * 0.02, learnable=True),
        retrieval_k=V,
    )


def train_on(model, tokens, steps: int = 400, lr: float = 3e-3, batch_size: int = 8,
             seq_len: int = 48, seed: int = 0) -> list[float]:
    """Next-token training through the package's own TrainLoop. Returns losses."""
    from cubbyllm.core.generation import SnapshotHardener
    from cubbyllm.training import InMemoryDataPipeline, TrainLoop

    data = InMemoryDataPipeline(np.asarray(tokens, dtype=np.int64),
                                manifest="prospection/branch-grammar", seed=seed)
    loop = TrainLoop(model, data, SnapshotHardener(), lr=lr, batch_size=batch_size,
                     seq_len=seq_len)
    return [loop.step() for _ in range(steps)]


_TOY_CACHE: dict = {}


def trained_toy(steps: int = 400, seed: int = 0, backbone: str = "mingru",
                d: int = 32, L: int = 2, n_segments: int = 600):
    """(model, grammar, train_losses) — a toy model trained on the branch grammar,
    cached per process so several scripts/tests share one training run."""
    key = (steps, seed, backbone, d, L, n_segments)
    if key not in _TOY_CACHE:
        g = BranchGrammar(seed=seed)
        toks, _ = g.sample(n_segments, np.random.default_rng(seed))
        m = build_model(d=d, L=L, V=g.V, backbone=backbone, seed=seed)
        losses = train_on(m, toks, steps=steps, seed=seed)
        _TOY_CACHE[key] = (m, g, losses)
    return _TOY_CACHE[key]


def load_checkpoint(path: str, dev=None):
    """Rebuild + load a CubbyModel from the ``{"meta", "params"}`` checkpoint
    layout — the same construction as ``exp_needle_recall.build`` so a hybrid
    checkpoint is never silently rebuilt as pure MinGRU."""
    from cubbyllm.core.config import CubbyConfig
    from cubbyllm.core.context import FrozenSlotRouter
    from cubbyllm.core.device import move_model
    from cubbyllm.core.generation import (BasisHyperGenerator, HyperGenerator,
                                          SnapshotHardener)
    from cubbyllm.model.assembly import CubbyModel
    from cubbyllm.model.backbone import HybridBackbone, MinGRUBackbone
    from cubbyllm.model.binding import BindingHead
    from cubbyllm.model.memory import MemoryLayer
    from cubbyllm.model.vocab import HybridEmbedding, TopKRetrievalHead

    dev = dev or torch.device("cpu")
    ck = torch.load(path, map_location="cpu", mmap=True)   # pages in on copy; ~1.8GB files
    meta = dict(ck["meta"])
    d, L, V = int(meta["D"]), int(meta["L"]), int(meta["vocab"])
    kind = meta.get("gen", "flat")
    gen = (HyperGenerator(ctx_dim=CTX_DIM, n_out=d * d) if kind == "flat" else
           BasisHyperGenerator(ctx_dim=CTX_DIM, d_model=d, n_layers=L))
    if meta.get("backbone") == "hybrid":
        bb = HybridBackbone(d, L, attn_every=int(meta.get("attn_every", 3)),
                            window=int(meta.get("window", 512)),
                            heads=int(meta.get("heads", 8)),
                            mem_every=int(meta.get("mem_every", 0)),
                            mem_topk=int(meta.get("mem_topk", 8)),
                            mem_key=int(meta.get("mem_key", 64)))
    else:
        bb = MinGRUBackbone(d, L)
    torch.manual_seed(0)
    m = CubbyModel(
        config=CubbyConfig(d_model=d, n_layers=L, ctx_dim=CTX_DIM, vocab_core=V),
        context_source=FrozenSlotRouter(input_dim=d, n_slots=8, ctx_dim=CTX_DIM).freeze(),
        backbone=bb, memory=MemoryLayer(gen, SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(V, d),
        head=TopKRetrievalHead(torch.randn(V, d) * 0.02, learnable=True),
        retrieval_k=V)
    move_model(m, dev)
    with torch.no_grad():
        for p, s in zip(m.parameters(), ck["params"]):
            p.copy_(s.to(p.device))
    return m, meta


def real_tokens(n: int = 4096) -> np.ndarray | None:
    """Tokens for a checkpoint run: CB_TEXT (plain text, tokenized with CUBBY_SPM)
    or CB_CORPUS (token cache via WeightedCorpusPipeline). None if neither set."""
    if TEXT and SPM:
        from cubbyllm.training.data import _load_tokenizer
        encode, _, _, _ = _load_tokenizer(SPM)
        with open(TEXT, encoding="utf-8", errors="replace") as f:
            ids = encode(f.read())
        return np.asarray(ids[:n], dtype=np.int64)
    if CORPUS and SPM:
        import glob
        import json
        from cubbyllm.training import WeightedCorpusPipeline
        srcs = (json.load(open(SOURCES, encoding="utf-8"))["sources"] if SOURCES else
                [{"name": os.path.splitext(os.path.basename(p))[0], "weight": 1.0}
                 for p in sorted(glob.glob(os.path.join(CORPUS, "*.u32")))])
        pipe = WeightedCorpusPipeline(srcs, SPM, CORPUS, seed=99, cache_only=True).prepare()
        x, _ = next(pipe.batches(1, n))
        return x[0].numpy().astype(np.int64)
    return None


# ── decode-state forking ─────────────────────────────────────────────────────

def _clone_store(store):
    """Deep-copy an EpisodicStore (duck-typed) so a branch's writes stay isolated."""
    new = copy.copy(store)
    new.load_state_dict({k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in store.state_dict().items()})
    return new


def _clone_any(s):
    """Deep-clone a backbone decode state of ANY shape: MinGRU (1, d) tensors,
    hybrid attention (k, v, pos) tuples, episodic-memory dicts with a store."""
    if torch.is_tensor(s):
        return s.clone()
    if hasattr(s, "state_dict") and hasattr(s, "retrieve_cosine"):     # EpisodicStore
        return _clone_store(s)
    if isinstance(s, dict):
        return {k: _clone_any(v) for k, v in s.items()}
    if isinstance(s, (list, tuple)):
        return type(s)(_clone_any(e) for e in s)
    return s                                                            # int / None


def clone_state(state: dict) -> dict:
    """Snapshot a ``CubbyModel.step`` state. This is a FORK: the copy can be
    rolled forward independently and the original stays re-enterable."""
    return {"bb": _clone_any(state["bb"]), "ctx_sum": state["ctx_sum"].clone(),
            "ctx_n": state["ctx_n"]}


def state_numel(state: dict) -> int:
    """Floats carried in the backbone state (what a fork has to copy)."""
    def walk(s):
        if torch.is_tensor(s):
            return s.numel()
        if isinstance(s, dict):
            return sum(walk(v) for k, v in s.items() if k != "store")
        if isinstance(s, (list, tuple)):
            return sum(walk(e) for e in s)
        return 0
    return walk(state["bb"])


@torch.no_grad()
def prefix_state(model, ids: torch.Tensor, state=None):
    """Feed 1-D ``ids`` through ``model.step``; returns (last_logits, state)."""
    logits = None
    for t in range(ids.shape[0]):
        logits, state = model.step(ids[t].view(1), state)
    return logits, state


@torch.no_grad()
def continue_from(model, state, ids: torch.Tensor, track=None):
    """Roll ``ids`` forward from a CLONE of ``state``. Returns (logits (n, V),
    final_state). If ``track`` is a list, the per-step MinGRU layer states are
    appended to it as (n_layers_gru, d) tensors — used by the divergence probes."""
    s = clone_state(state)
    outs = []
    for t in range(ids.shape[0]):
        lg, s = model.step(ids[t].view(1), s)
        outs.append(lg[0])
        if track is not None:
            track.append(gru_states(s))
    return torch.stack(outs), s


def gru_states(state: dict) -> torch.Tensor:
    """Stack the MinGRU layers' (1, d) states -> (n_gru_layers, d)."""
    bb = state["bb"]
    mix = bb["mix"] if isinstance(bb, dict) else bb
    return torch.stack([s[0] for s in mix if torch.is_tensor(s)])


@torch.no_grad()
def state_trajectory(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-step MinGRU states along ``ids``: (T, n_gru_layers, d)."""
    track = []
    state = None
    for t in range(ids.shape[0]):
        _, state = model.step(ids[t].view(1), state)
        track.append(gru_states(state))
    return torch.stack(track)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats over the finite support of the last dim (the
    retrieval head marks non-top-K entries -inf; those contribute 0)."""
    lp = F.log_softmax(logits.float(), dim=-1)
    p = lp.exp()
    return -torch.where(torch.isfinite(lp), p * lp, torch.zeros_like(p)).sum(-1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUROC (Mann-Whitney), ties averaged. labels: bool."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):                           # average ranks over ties
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ── MinGRU retention gate: time constants and chrono init ───────────────────

def mingru_mixers(backbone) -> list:
    """[(layer_idx, mixer)] for every MinGRU mixer in a MinGRU or Hybrid backbone."""
    from cubbyllm.model.backbone.mingru import _MinGRUMixer
    return [(i, m) for i, m in enumerate(backbone.mix) if isinstance(m, _MinGRUMixer)]


def retention_from_gate_logits(z: torch.Tensor) -> torch.Tensor:
    """The mixer's own gate expression (kept in sync with mingru.py by a pin in
    exp_p3): a = 0.001 + 0.998 * sigmoid(proj_d(x))."""
    return A_MIN + (A_MAX - A_MIN) * torch.sigmoid(z)


def time_constants(a: torch.Tensor) -> torch.Tensor:
    """tau = -1 / ln(a): the number of steps for a retained value to decay to 1/e.
    Elementwise; input is a retention in (0, 1)."""
    return -1.0 / torch.log(a.clamp(min=1e-6, max=A_MAX))


@torch.no_grad()
def retention_profile(model, tokens: torch.Tensor) -> dict:
    """Run one parallel trunk pass (``model.features`` — no output head, so this
    is cheap even at V=128k) and capture every MinGRU layer's per-unit retention
    over all (batch, position): {layer_idx: (B*S, d)}. Hooks the gate
    projection, so what is measured is exactly what the scan multiplies by."""
    caught, hooks = {}, []
    for i, m in mingru_mixers(model.backbone):
        def hook(mod, inp, out, i=i):
            caught[i] = retention_from_gate_logits(out).reshape(-1, out.shape[-1])
        hooks.append(m.proj_d.register_forward_hook(hook))
    try:
        model.features(tokens)
    finally:
        for h in hooks:
            h.remove()
    return caught


def unit_time_constants(profile: dict) -> dict:
    """{layer_idx: (d,) tau} from a retention profile, via the mean of ln(a) over
    inputs (the geometric-mean retention is what a value decays by per step on
    average, so it is the right average to take before inverting)."""
    return {i: time_constants(torch.exp(torch.log(a).mean(0))) for i, a in profile.items()}


def unit_latch_fraction(profile: dict, tau_at_least: float = 100.0) -> dict:
    """{layer_idx: (d,) fraction of positions at which the unit's INSTANTANEOUS
    retention implies tau >= tau_at_least}. A conditional latch (a ~ 1 on most
    steps, a ~ 0 on reset events) has a short geometric-mean tau but a high
    latch fraction — the two together separate 'always short' from
    'long, with input-driven resets'."""
    a_min = math.exp(-1.0 / tau_at_least)
    return {i: (a >= a_min).float().mean(0) for i, a in profile.items()}


@torch.no_grad()
def chrono_init_(mixer, t_min: float = 1.0, t_max: float = 500.0, seed: int = 0,
                 weight_scale: float = 0.1) -> torch.Tensor:
    """Re-initialise the retention-gate bias so the per-unit time constant at
    init is log-uniform in [t_min, t_max] — the chrono init of Tallec & Ollivier
    (2018), adapted to this gate's affine sigmoid. ``weight_scale`` shrinks the
    gate's input weights so the intended spectrum is legible at init (inputs can
    still modulate it). t_max must stay under the parameterization's ceiling.
    Returns the target tau per unit."""
    assert t_max < TAU_CAP, f"t_max {t_max} exceeds the gate ceiling {TAU_CAP:.1f}"
    g = torch.Generator().manual_seed(seed)
    d = mixer.proj_d.bias.shape[0]
    tau = torch.exp(torch.empty(d).uniform_(math.log(t_min), math.log(t_max), generator=g))
    a = torch.exp(-1.0 / tau)
    p = ((a - A_MIN) / (A_MAX - A_MIN)).clamp(1e-6, 1 - 1e-6)
    mixer.proj_d.bias.copy_(torch.log(p / (1 - p)))
    mixer.proj_d.weight.mul_(weight_scale)
    return tau


def pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float().flatten(), q))


def banner(title: str, lines: list[str]) -> None:
    print(title)
    for ln in lines:
        print("  " + ln)
    print(flush=True)
