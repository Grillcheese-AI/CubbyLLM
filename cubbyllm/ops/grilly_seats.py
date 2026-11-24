"""cubbyllm/ops/grilly_seats.py -- the 450M's seats on grilly2 behind the ops seam (order 0031): load a base export, wrap
its projections with LoRA, copy a role's A/B pair in, decode a batch; and (order 0068) attach route 5, the vocabulary
shortlist greedy one-row decodes pick their tokens with.

Only ``cubbyllm/ops/`` names grilly (``test_ops_is_sole_grilly_importer``); the seats' trunk (``Cubby450mTrunk``, bound
for ``cubbyllm/emit/seats_450m.py``) calls these. Each function is the code the trunk (and
``validation/exp_e5_base450m_probes.load_grilly``) ran before, verbatim. grilly is imported when a function runs, so
``import cubbyllm`` stays grilly-free.

Wired: WIRED (``Cubby450mTrunk``).
"""
from __future__ import annotations

import json
import os

from ..core.protocols import Wiring

__wiring__ = Wiring.WIRED


def grilly():
    """grilly2's module, the handle the trunk keeps (a test passes its own: ``no_grad``, ``from_numpy``, ...)."""
    import grilly as g
    return g


def load_export(export_dir: str):
    """The base export (``validation/export_base.py``: config.json + model.safetensors) as grilly2's
    ``CubbyForCausalLM`` -> (model, the config dict)."""
    from grilly.infer.models import CubbyConfig, CubbyForCausalLM
    from grilly.infer.safetensors import load_model
    with open(os.path.join(export_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    fields = {k: v for k, v in cfg.items() if k not in ("model_type", "source")}
    model = CubbyForCausalLM(CubbyConfig(**fields))
    missing, unexpected = load_model(model, os.path.join(export_dir, "model.safetensors"))
    if missing or unexpected:
        raise SystemExit(f"export does not match the model: missing {missing}, unexpected {unexpected}")
    return model, cfg


def attach_route(model, index_path: str, near: int = 2048, context: int = 8192, ranks: int = 256):
    """Order 0068's route 5 on ``model`` (grilly2's ``CubbyForCausalLM``): a ``ShortlistIndex`` (``.npz`` with arm
    (d)'s screen lists and the successors) as grilly2's ``ShortlistHead`` over the model's own head rows, set as
    ``model.shortlist`` -> the head. grilly2's ``generate`` then picks each token of a greedy one-row decode with no
    repetition penalty and no processors with it; every other decode runs the full head. None, and nothing set, when
    this grilly2 has no shortlist decode. Refused (ValueError) when the index is not over this head's rows."""
    try:
        from grilly.infer.shortlist import ShortlistHead
    except ImportError:
        return None
    if not callable(getattr(model, "hidden", None)) or not hasattr(model, "shortlist"):
        return None
    from ..model.vocab.shortlist import ShortlistIndex
    idx = ShortlistIndex.load(index_path)
    rows = int(model.lm_head.weight.shape[0])
    name = os.path.basename(index_path)
    if idx.screen is None:
        raise ValueError(f"{name}: no screen lists (route 5's context list is arm (d)'s)")
    if len(idx.assign) != rows:
        raise ValueError(f"{name} is over {len(idx.assign)} head rows; this head has {rows}")
    if idx.succ is not None and idx.succ.shape[0] != rows:
        raise ValueError(f"{name}'s successors cover {idx.succ.shape[0]} tokens; this head has {rows}")
    s = idx.screen
    head = ShortlistHead(model.lm_head.weight, s.centroids, s.ids, s.offsets, idx.succ, near=near, context=context,
                         ranks=ranks)
    model.shortlist = head
    return head


def wrap_lora(model, rank, alpha, targets) -> dict:
    """Every target projection of ``model`` wrapped with a LoRA pair, in place -> {path: wrapped module}."""
    from grilly.infer.lora import apply_lora_
    return apply_lora_(model, rank, alpha, targets)


def copy_pair(wrapped: dict, weights: dict, g=None) -> None:
    """A role's A/B tensors ({path: (A, B)}, numpy, kept on the host) copied into the wrapped projections: the delta
    moves, the base does not. ``g``: the grilly module (``grilly()`` when None)."""
    g = g or grilly()
    with g.no_grad():
        for path, mod in wrapped.items():
            a, b = weights[path]
            mod.lora_A.weight.copy_(g.from_numpy(a))
            mod.lora_B.weight.copy_(g.from_numpy(b))


def suppress_processor(token_ids: list):
    """grilly2's processor that keeps `token_ids` from ever being written (order 0071's hook; order 0067's steering),
    or None when this grilly2 has no hook."""
    try:
        from grilly.infer.logits_process import SuppressTokensLogitsProcessor
    except ImportError:
        return None
    return SuppressTokensLogitsProcessor(token_ids)


def ranked_processor(choose, widths=(32, 512)):
    """grilly2's processor that lets ``choose(ids_so_far, ranked_ids) -> allowed ids`` pick among a row's best-scored
    ids (the best ``widths[0]`` first, then more, then every id best first), every other id masked out (order 0067's
    second reader: decoding held to a grammar). None when this grilly2 has no hook."""
    try:
        from grilly import full, int64, tensor
    except ImportError:
        return None
    import math

    import numpy as np

    class _Ranked:
        def __call__(self, input_ids, scores):
            rows, vocab = scores.shape
            allowed = []
            for i in range(rows):
                ids = [int(t) for t in input_ids[i].tolist()]
                s = np.asarray(scores[i].tolist(), dtype=np.float64)
                pick, order = [], None
                for w in widths:
                    top = np.argpartition(-s, min(w, vocab - 1))[:w]
                    pick = choose(ids, [int(t) for t in top[np.argsort(-s[top])]])
                    if pick:
                        break
                if not pick:
                    order = np.argsort(-s)
                    pick = choose(ids, [int(t) for t in order])
                if not pick:
                    raise ValueError("no token keeps the answer in its grammar")
                allowed.append(pick)
            width = max(len(a) for a in allowed)
            mask = full((rows, vocab), -math.inf)
            mask.scatter_(1, tensor([a + [a[0]] * (width - len(a)) for a in allowed], dtype=int64), value=0.0)
            return scores + mask
    return _Ranked()


def generate(model, rows: list, max_new_tokens: int, eos: int, temperature: float = 0.0, top_k: int = 40,
             seed: "int | None" = None, g=None, logits_processor=None, prefix_allowed_tokens_fn=None) -> list:
    """Rows of token ids -> each row with its continuation (as lists), greedy; sampled at ``temperature`` > 0, seeded
    by ``seed``. Stops a row at ``eos``. ``g``: the grilly module (``grilly()`` when None).

    ``logits_processor`` / ``prefix_allowed_tokens_fn`` (order 0071): grilly2's hook, transformers' names, handed to
    ``model.generate`` only when given (processors ``(input_ids, scores) -> scores`` on device tensors, run after the
    repetition penalty and before sampling; ``grilly.infer.logits_process`` has the common ones). Both None: the call
    is the one before the hook, word for word."""
    g = g or grilly()
    if seed is not None and temperature > 0:
        g.manual_seed(int(seed))
    hook = {k: v for k, v in (("logits_processor", logits_processor),
                              ("prefix_allowed_tokens_fn", prefix_allowed_tokens_fn)) if v is not None}
    with g.no_grad():
        if temperature > 0:
            out = model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=eos,
                                 do_sample=True, temperature=float(temperature), top_k=int(top_k), top_p=1.0, **hook)
        else:
            out = model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=eos, **hook)
    return out.tolist()


def allowed_processor(allowed, start: int, g=None):
    """Order 0081: a logits processor for ONE row that keeps only the ids ``allowed(row ids after ``start``)``
    returns, and leaves the scores as they are when it returns None (outside a constrained literal: no mask is built,
    so an unconstrained step costs one read of the row's ids). grilly2's ``prefix_allowed_tokens_fn`` must return a
    non-empty list every step, so "no constraint" would be the whole vocabulary's mask at every step."""
    import math
    g = g or grilly()

    def process(input_ids, scores):
        rows, vocab = scores.shape
        if rows != 1:
            raise ValueError("allowed_processor constrains one row")
        ids = allowed(input_ids[0].tolist()[start:])
        if ids is None:
            return scores
        if not ids:
            raise ValueError("allowed_processor: the constraint allows nothing")
        mask = g.full((1, vocab), -math.inf)
        mask.scatter_(1, g.tensor([list(ids)], dtype=g.int64), value=0.0)
        return scores + mask
    return process


def device_name(g=None) -> str:
    """The card grilly2 decodes on (Cowork, 2026-11-24: every log names its card), or "unknown"."""
    try:
        g = g or grilly()
        return str(g._C.device_name())
    except Exception:                                    # noqa: BLE001 -- a log line, never a failure
        return "unknown"
