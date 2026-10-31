import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\exp_k10_switch.py"
s = open(p, encoding="utf-8").read()
i = s.index("@torch.no_grad()\ndef best_match(")
j = s.index("def gate_targets(")
new = '''@torch.no_grad()
def best_match(model, a, held, lay) -> dict:
    """The read's cosine to its best candidate (the gate's own `best`) and its margin over the second best, at the first
    answer token of present facts and of absent ones, and at every other empty asking position; and how well one
    threshold on the best match alone tells present from absent there (0021's kill line: if they overlap, no gate on
    these inputs can tell them apart)."""
    ids, key, plc, R, _gold, spans = held
    out, calls = {"present": [], "absent": [], "empty": []}, []

    def best(_m, inp, _o):
        r, table, idx, cid, ok = inp[2], inp[3], inp[4], inp[5], inp[6]
        e = F.normalize(table.float()[cid], dim=-1)
        rn = F.normalize(r.float(), dim=-1)
        sim = (rn @ e.T if cid.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)).masked_fill(~ok, float("-inf"))
        top = sim.topk(min(2, sim.shape[-1]), -1).values
        calls.append((idx, top[:, 0], top[:, 0] - (top[:, 1] if top.shape[-1] > 1 else 0)))
    hook = model.spatial.copy_path.register_forward_hook(best)
    try:
        for b in range(0, ids.shape[0], 16):
            sl, S = slice(b, b + 16), ids.shape[1] - 1
            calls.clear()
            fwd(model, ids[sl, :-1], ft.inputs(a, key[sl], plc[sl], R[sl])[:, :-1], R[sl, :-1])
            for ix, bst, mg in calls:
                for j, v, m in zip(ix.tolist(), bst.tolist(), mg.tolist()):
                    i, t = divmod(j, S)
                    here, o, info = spans[b + i]
                    k = ("absent" if info["absent"] else "present") if t == here else "empty"
                    if k != "empty" or not (here <= t < here + len(o)) or info["absent"]:
                        out[k].append((v, m if np.isfinite(m) else 0.0))
    finally:
        hook.remove()
    q = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 4) for p in (5, 25, 50, 75, 95)}   # noqa: E731
    rep = {k: {"n": len(v), "best": q([x[0] for x in v]), "margin": q([x[1] for x in v])} for k, v in out.items() if v}
    if out["present"] and out["absent"]:
        pb, ab = np.array([x[0] for x in out["present"]]), np.array([x[0] for x in out["absent"]])
        score, thr = max(((float((pb >= t).mean() + (ab < t).mean()) / 2, float(t)) for t in np.linspace(0, 0.6, 241)))
        rep["one_threshold"] = {"at": round(thr, 4), "balanced_accuracy": round(score, 4)}
    return rep


'''
s = s[:i] + new + s[j:]
s = s.replace('''    log(f"  frozen parameters unchanged: {rep['frozen_hash']['unchanged']}; best match present {rep['best_match']['present']}"
        f" empty {rep['best_match']['empty']}")''', '''    bm = rep["best_match"]
    log(f"  frozen parameters unchanged: {rep['frozen_hash']['unchanged']}; best match (p5/p50/p95) present "
        f"{[bm['present']['best'][k] for k in ('p5', 'p50', 'p95')]}, absent {[bm['absent']['best'][k] for k in ('p5', 'p50', 'p95')]};"
        f" one threshold {bm.get('one_threshold')}")''')
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
t = r"C:\Users\grill\Documents\GitHub\CubbyLLM\tests\test_k10_switch.py"
u = open(t, encoding="utf-8").read()
a_ = '''    assert set(bm) == {"present", "empty"} and bm["present"]["p5"] <= bm["present"]["p95"]'''
assert u.count(a_) == 1
u = u.replace(a_, '''    assert {"present", "absent"} <= set(bm) and bm["present"]["best"]["p5"] <= bm["present"]["best"]["p95"]
    assert 0.5 <= bm["one_threshold"]["balanced_accuracy"] <= 1.0


def test_the_gate_loss_is_off_by_default_and_targets_the_asked_pieces(joint):
    a, _rep = joint
    m1, b, lay, held = sw.load(a.save)
    m2, *_ = sw.load(a.save)
    torch.manual_seed(1)
    r1 = sw.copy_only(m1, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000)
    torch.manual_seed(1)
    r2 = sw.copy_only(m2, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000, gate_loss=0.0)
    assert r1["final"]["exact"] == r2["final"]["exact"] and all(torch.equal(x, y) for (_n, x), (_k, y) in zip(sw.tensors(m1), sw.tensors(m2)))
    m3, *_ = sw.load(a.save)
    r3 = sw.copy_only(m3, b, held, lay, steps=2, lr=1e-1, log=lambda m: None, eval_every=1000, gate_loss=1.0)
    assert r3["frozen_hash"]["unchanged"]
    spans = [(10, [5, 6], {"absent": False}), (10, [9], {"absent": True})]
    S = 20
    assert sw.gate_targets(torch.tensor([10, 11, 12, 30, 31]), spans, S).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]''')
open(t, "w", encoding="utf-8", newline="\n").write(u)
print("ok")
