"""old (expand+gather) vs new (flat index_select, no_grad selection): equal values and grads; time + peak."""
import time, sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from cubbyllm.model.recall import MemoryRead

def old_forward(m, h, window):
    B, S, d = h.shape
    q, k, v = m.qkv(h)
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    sim = torch.einsum("bik,bjk->bij", qn, kn)
    i = torch.arange(S)
    allowed = (i[:, None] - i[None, :]) > window
    sim = sim.masked_fill(~allowed[None], float("-inf"))
    kk = min(m.topk, S)
    topv, topi = sim.topk(kk, dim=-1)
    dk = k.shape[-1]
    k_set = torch.gather(k.unsqueeze(1).expand(B, S, S, dk), 2, topi.unsqueeze(-1).expand(B, S, kk, dk))
    v_set = torch.gather(v.unsqueeze(1).expand(B, S, S, d), 2, topi.unsqueeze(-1).expand(B, S, kk, d))
    valid = ~torch.isinf(topv)
    sim2 = (q.unsqueeze(2) * k_set).sum(-1) * m.scale
    sim2 = sim2.masked_fill(~valid, -1e9)
    w = torch.softmax(sim2, dim=-1)
    r = m.o((w.unsqueeze(-1) * v_set).sum(2))
    no_mem = ~valid.any(-1, keepdim=True)
    return torch.where(no_mem, torch.zeros_like(r), r)

torch.manual_seed(0)
for (B, S, d, W) in [(2, 96, 32, 16), (2, 512, 256, 128)]:
    m = MemoryRead(d, d_key=16, topk=8)
    h = torch.randn(B, S, d, requires_grad=True)
    outs, grads, times = [], [], []
    for f in (lambda: old_forward(m, h, W), lambda: m(h, W)):
        m.zero_grad(); h.grad = None
        t0 = time.perf_counter()
        y = f(); (y.pow(2).sum()).backward()
        times.append(time.perf_counter() - t0)
        outs.append(y.detach()); grads.append([h.grad.clone()] + [p.grad.clone() for p in m.parameters()])
    same = torch.allclose(outs[0], outs[1], atol=1e-5) and all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(*grads))
    print(f"B{B} S{S} d{d}: equal {same}; old {times[0]*1e3:.0f} ms, new {times[1]*1e3:.0f} ms fwd+bwd; "
          f"old backward scatter target {B*S*S*d*4/2**20:.0f} MB fp32 per layer")
