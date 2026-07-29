import inspect

import cubbyllm.model.backbone.base as b
import cubbyllm.model.backbone.mingru as m
from cubbyllm.core.protocols import Wiring
from cubbyllm.model.backbone import Backbone, MinGRUBackbone


def test_backbone_protocol_present():
    assert hasattr(b, "Backbone") and hasattr(b.Backbone, "forward")


def test_base_stays_protocol_only():
    """base.py holds ONLY the Backbone protocol — concrete backbones live in
    their own modules (mingru.py), so the interface never hard-codes a choice.
    """
    defined = [
        name for name, obj in inspect.getmembers(b, inspect.isclass)
        if obj.__module__ == b.__name__
    ]
    assert defined == ["Backbone"], f"base.py must stay protocol-only, found {defined}"


def test_mingru_backbone_conforms_and_runs():
    import torch

    torch.manual_seed(0)
    bb = MinGRUBackbone(d_model=16, n_layers=2)
    assert isinstance(bb, Backbone)
    x = torch.randn(2, 5, 16)
    out = bb.forward(x)
    assert out.shape == x.shape
    assert sum(p.numel() for p in bb.parameters()) > 0   # it is trainable


def test_wiring_declared():
    assert b.__wiring__ is Wiring.WIRED
    assert m.__wiring__ is Wiring.WIRED
