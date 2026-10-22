"""cubbyllm/ops/grilly_seats.py -- the 450M's seats on grilly2 behind the ops seam (order 0031): load a base export, wrap
its projections with LoRA, copy a role's A/B pair in, decode a batch.

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


def generate(model, rows: list, max_new_tokens: int, eos: int, temperature: float = 0.0, top_k: int = 40,
             seed: "int | None" = None, g=None) -> list:
    """Rows of token ids -> each row with its continuation (as lists), greedy; sampled at ``temperature`` > 0, seeded
    by ``seed``. Stops a row at ``eos``. ``g``: the grilly module (``grilly()`` when None)."""
    g = g or grilly()
    if seed is not None and temperature > 0:
        g.manual_seed(int(seed))
    with g.no_grad():
        if temperature > 0:
            out = model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=eos,
                                 do_sample=True, temperature=float(temperature), top_k=int(top_k), top_p=1.0)
        else:
            out = model.generate(g.tensor(rows), max_new_tokens=max_new_tokens, eos_token_id=eos)
    return out.tolist()
