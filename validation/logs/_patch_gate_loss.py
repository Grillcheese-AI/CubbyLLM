import ast
p = r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\exp_k10_switch.py"
s = open(p, encoding="utf-8").read()
reps = [
('''def copy_only(model, a, held, lay, steps: int, lr: float, log, eval_every: int = 100) -> dict:''',
 '''def gate_targets(idx, spans, S):
    """1 where the asked fact holds a piece at that asking position, 0 elsewhere (an absent fact, the closing EOS)."""
    out = []
    for j in idx.tolist():
        i, t = divmod(j, S)
        here, o, info = spans[i]
        out.append(float(not info["absent"] and here <= t < here + len(o)))
    return torch.tensor(out)


def copy_only(model, a, held, lay, steps: int, lr: float, log, eval_every: int = 100, gate_loss: float = 0.0) -> dict:
    """`gate_loss` (off by default: the order's recipe is the language loss alone): adds the gate's binary cross-entropy
    against `gate_targets`, one term an asking position -- a present fact's five pieces and an absent fact's one NONE
    weigh alike there, where in the language loss the pieces outvote it."""'''),
('''    before, t0, curve = frozen_hash(model), time.time(), []''',
 '''    lams = []
    hook = model.spatial.copy_path.register_forward_hook(lambda _m, inp, o: lams.append((inp[4], o[2][:, 0])) if gate_loss else None)
    before, t0, curve = frozen_hash(model), time.time(), []'''),
('''        ids, key, plc, R, _gold, _spans = batch(rng, lang, lay, a.batch, a)
        logits = fwd(model, ids[:, :-1], ft.inputs(a, key, plc, R)[:, :-1], R[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, a.vocab), ids[:, 1:].reshape(-1))''',
 '''        ids, key, plc, R, _gold, spans = batch(rng, lang, lay, a.batch, a)
        lams.clear()
        logits = fwd(model, ids[:, :-1], ft.inputs(a, key, plc, R)[:, :-1], R[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, a.vocab), ids[:, 1:].reshape(-1))
        for ix, lam in lams:                                             # the switch's own target, with --gate-loss
            loss = loss + gate_loss * F.binary_cross_entropy(lam.clamp(1e-6, 1 - 1e-6), gate_targets(ix, spans, ids.shape[1] - 1))'''),
('''    after = frozen_hash(model)''', '''    hook.remove()
    after = frozen_hash(model)'''),
('''    ap.add_argument("--lr", type=float, default=1e-2)''', '''    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--gate-loss", type=float, default=0.0, help="the gate's BCE on present/empty asking positions (off)")'''),
('''    rep.update(copy_only(model, a, held, lay, c.copy_only, c.lr, log, c.eval_every))''',
 '''    rep.update(copy_only(model, a, held, lay, c.copy_only, c.lr, log, c.eval_every, c.gate_loss))'''),
('''    name = f"exp_k10_switch_{a.arm}_s{a.seed}{'_t12' if getattr(a, 'templates', 6) == 12 else ''}{c.tag}_{c.date}"''',
 '''    name = (f"exp_k10_switch_{a.arm}_s{a.seed}{'_t12' if getattr(a, 'templates', 6) == 12 else ''}"
            f"{f'_gl{c.gate_loss:g}' if c.gate_loss else ''}{c.tag}_{c.date}")'''),
]
for a_, b_ in reps:
    assert s.count(a_) == 1, a_[:70]
    s = s.replace(a_, b_)
ast.parse(s)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("\n"))
