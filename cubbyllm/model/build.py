"""cubbyllm/model/build.py -- the model a checkpoint's meta describes (``validation/train_base.py`` builds it to train,
the 450M's loci reader to read; order 0031 moved it here, verbatim but for the trainer's gradient checkpointing,
now an argument).

Wired: WIRED (the loci reader; the trainer imports it).
"""
from __future__ import annotations

import torch

from ..core.config import CubbyConfig
from ..core.context import FrozenSlotRouter
from ..core.device import move_model
from ..core.generation import BasisHyperGenerator, SnapshotHardener
from .assembly import CubbyModel
from .backbone import HybridBackbone
from .binding import BindingHead
from .memory import MemoryLayer
from .vocab import HybridEmbedding, TopKRetrievalHead

__wiring__ = "WIRED"


def build_model(meta: dict, dev, seed: int = 0, grad_checkpoint: bool = False) -> CubbyModel:
    """The model a checkpoint's ``meta`` describes, on ``dev``; ``grad_checkpoint``: the trainer's flag."""
    torch.manual_seed(seed)
    d, n_layers, v = meta["D"], meta["L"], meta["vocab"]
    cfg = CubbyConfig(d_model=d, n_layers=n_layers, ctx_dim=meta["ctx"], vocab_core=v)
    model = CubbyModel(
        config=cfg,
        context_source=FrozenSlotRouter(input_dim=d, n_slots=meta["slots"],
                                        ctx_dim=meta["ctx"]).freeze(),
        backbone=HybridBackbone(d, n_layers, attn_every=meta["attn_every"],
                                ffn_mult=meta.get("ffn_mult", 2),
                                window=meta["window"], heads=meta["heads"],
                                mem_every=int(meta.get("mem_every", 0) or 0),
                                mem_topk=int(meta.get("mem_topk", 8)), mem_key=int(meta.get("mem_key", 64)),
                                grad_checkpoint=grad_checkpoint),
        memory=MemoryLayer(BasisHyperGenerator(ctx_dim=meta["ctx"], d_model=d,
                                               n_layers=n_layers, n_basis=meta["gen_basis"],
                                               rank=meta["gen_rank"]),
                           SnapshotHardener(), d_model=d),
        binding=BindingHead(), embedding=HybridEmbedding(v, d),
        head=TopKRetrievalHead(torch.randn(v, d) * 0.02, learnable=True),
        retrieval_k=v,
        spatial=spatial_link(meta),
    )
    move_model(model, dev)
    return model


def spatial_link(meta: dict):
    """The spatial link the meta asks for, or None (``meta["spatial"] = {scale, seed}``)."""
    sp = meta.get("spatial")
    if not sp:
        return None
    from .spatial import SpatialLink
    return SpatialLink(meta["D"], scale=float(sp.get("scale", 2.0)), seed=int(sp.get("seed", 0)),
                       at=int(sp.get("at", meta["L"] // 2)), query=str(sp.get("query", "learned")),
                       bind=str(sp.get("bind", "placed")), copy=sp.get("copy"),   # copy: absent = no copy path
                       copy_tau=float(sp.get("copy_tau", 3.0)), copy_ring=int(sp.get("copy_ring", 256)),
                       copy_beta=sp.get("copy_beta"),                    # copy_beta: absent = the gate sees h alone
                       key=sp.get("key"), key_pieces=int(sp.get("key_pieces", 16)),   # key: absent = places (order 0006)
                       pointer_window=int(sp.get("pointer_window", 16)), pointer_dim=int(sp.get("pointer_dim", 64)),
                       copy_refuse=sp.get("copy_refuse"))                # copy_refuse: absent = the mix (order 0022)
