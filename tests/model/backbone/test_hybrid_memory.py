import pytest
torch = pytest.importorskip("torch")
from cubbyllm.model.backbone import HybridBackbone

D, L, B, S, W = 32, 6, 2, 24, 6


def test_memory_off_is_identical_to_a_plain_hybrid():
    torch.manual_seed(0)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4, mem_every=0)
    x = torch.randn(B, S, D)
    assert bb.forward(x).shape == x.shape
    # no MemoryRead modules created when mem_every=0
    from cubbyllm.model.recall import MemoryRead
    assert not any(isinstance(m, MemoryRead) for m in bb.modules())


def test_memory_layer_reads_the_beyond_window_past_in_forward():
    torch.manual_seed(1)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4, mem_every=2)
    from cubbyllm.model.recall import MemoryRead
    assert any(isinstance(m, MemoryRead) for m in bb.modules())
    x = torch.randn(1, S, D)
    with torch.no_grad():
        y0 = bb.forward(x)[0, -1]
        xf = x.clone(); xf[0, 0] = torch.randn(D)      # beyond every window
        yf = bb.forward(xf)[0, -1]
    assert not torch.allclose(y0, yf, atol=1e-6), "memory did not carry the far token"


def test_grad_checkpoint_with_memory_flows_grads_to_memory_params():
    """Under grad_checkpoint=True the per-layer body (including the memory branch)
    is recomputed in backward; MemoryRead's params must receive the SAME gradients
    as the non-checkpointed path, and non-zero ones. If `_layer` weren't the whole
    checkpointed unit — or if use_reentrant were True — the memory params would
    silently get zero/wrong grads. Guards the first real CB_MEM_EVERY>0 training run.
    """
    def build(gc):
        torch.manual_seed(0)
        return HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4,
                              grad_checkpoint=gc, mem_every=2, mem_topk=8)

    torch.manual_seed(1)
    x0 = torch.randn(B, S, D, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)   # same values, own leaf per model

    bb0 = build(False)                              # checkpointing OFF
    bb0.forward(x0).pow(2).sum().backward()
    bb1 = build(True)                               # same init, checkpointing ON
    bb1.forward(x1).pow(2).sum().backward()

    g0 = {n: p.grad for n, p in bb0.named_parameters()}
    g1 = {n: p.grad for n, p in bb1.named_parameters()}
    mem_params = [n for n in g0 if n.startswith("mem.")]   # MemoryRead q/k/v/o only
    assert mem_params, "no MemoryRead params found — mem_every wiring changed?"
    for n in mem_params:
        assert g0[n] is not None and g1[n] is not None, f"{n} received no gradient"
        assert g0[n].abs().sum() > 0, f"{n} gradient is all-zero (memory branch not exercised)"
        assert torch.allclose(g0[n], g1[n], atol=1e-4), \
            f"checkpointed grad diverges on {n}: max|diff| {(g0[n] - g1[n]).abs().max():.2e}"


def test_step_with_memory_matches_forward():
    """Incremental decode with the store must equal the parallel forward, so a
    benchmark measures the trained model. Single sequence (B=1): the v1 store is
    per-sequence, and eval decode (exp_needle_recall) is B=1. `step` retrieves by
    COSINE (matching forward's dense selection); Hamming is the O(1) option the
    Task-4 guard covers, not the decode-equivalence contract."""
    torch.manual_seed(0)
    bb = HybridBackbone(D, n_layers=L, attn_every=3, window=W, heads=4,
                        mem_every=2, mem_topk=8)
    x = torch.randn(1, S, D)
    with torch.no_grad():
        parallel = bb.forward(x)
        states, seq = None, []
        for t in range(S):
            y, states = bb.step(x[:, t], states)
            seq.append(y)
        inc = torch.stack(seq, dim=1)
    assert torch.allclose(parallel, inc, atol=1e-3), \
        f"decode diverged; max |diff| {(parallel-inc).abs().max():.2e}"
