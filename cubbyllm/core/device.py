"""Device resolution for optional GPU acceleration (DirectML / CUDA / CPU).

Wired: STANDALONE — a torch-free-at-top-level helper (torch imported lazily).

On this machine (Windows + AMD Radeon RX 6750 XT) there is no CUDA, so torch
reaches the GPU through **DirectML** (``torch-directml``), which exposes a device
of type ``privateuseone``. ``torch-directml`` hard-pins ``torch==2.4.1``, so it
lives in a SEPARATE venv (``.venv-dml``) — the main test env stays on torch 2.10
CPU. See ``docs/DIRECTML.md``.

Nothing here is required for CPU use; the package runs unchanged without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .protocols import Wiring

if TYPE_CHECKING:
    from torch import device as TorchDevice


def resolve_device(prefer: str = "auto"):
    """Return the best available torch device: DirectML > CUDA > CPU.

    ``prefer``: 'auto' (default), 'dml'/'directml', 'cuda', or 'cpu'.
    """
    import torch

    if prefer in ("auto", "dml", "directml"):
        try:
            import torch_directml  # only present in the .venv-dml environment

            if torch_directml.is_available():
                return torch_directml.device()
        except Exception:
            if prefer in ("dml", "directml"):
                raise RuntimeError(
                    "DirectML requested but torch_directml is unavailable — "
                    "install it in the .venv-dml env (see docs/DIRECTML.md)."
                )
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_directml(device: "TorchDevice") -> bool:
    """True for a torch-directml device (torch exposes it as 'privateuseone')."""
    return getattr(device, "type", None) == "privateuseone"


def describe(device: "TorchDevice") -> str:
    """Human label for a resolved device."""
    if is_directml(device):
        try:
            import torch_directml

            return f"DirectML ({torch_directml.device_name(0)})"
        except Exception:
            return "DirectML"
    return str(device)


def _move_attrs(obj, device, seen: set, depth: int = 0):
    """Recursively move the nn.Modules / Parameters / Tensors an object holds.

    The CubbyModel's components (HybridEmbedding, TopKRetrievalHead, MemoryLayer,
    FrozenSlotRouter, ...) are PLAIN classes that *hold* torch state rather than
    subclass ``nn.Module``, so a bare ``.to()`` misses them. This walks each
    component's ``__dict__`` and moves the real state onto ``device``.
    """
    import torch
    import torch.nn as nn

    if obj is None or depth > 4 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, nn.Module):
        obj.to(device)                       # handles its own params + buffers
        return
    if not hasattr(obj, "__dict__"):
        return
    for name, val in list(vars(obj).items()):
        if isinstance(val, nn.Parameter):
            setattr(obj, name, nn.Parameter(
                val.detach().to(device), requires_grad=val.requires_grad))
        elif isinstance(val, nn.Module):
            val.to(device)
        elif isinstance(val, torch.Tensor):
            setattr(obj, name, val.to(device))
        elif hasattr(val, "__dict__") and not isinstance(val, type):
            _move_attrs(val, device, seen, depth + 1)


def move_model(model, device):
    """Move every torch component of an assembled ``CubbyModel`` onto ``device``.

    Call this BEFORE building the optimizer (params are reassigned on move). The
    grilly/numpy VSA (binding) is CPU algebra by design and is left alone.
    """
    seen: set = set()
    for name in ("embedding", "backbone", "memory", "head", "context_source"):
        _move_attrs(getattr(model, name, None), device, seen)
    return model


__wiring__ = Wiring.STANDALONE
