"""exp_refuse_450m -- order 0022 C: copy or refuse on the 450M's run 5, nothing retrained (CPU).

The probe: ``facts_far1k_v2``'s shape (facts ~800 tokens before their questions, readable names, the token before each
answer asking for its first piece) with ~20% of the questions about a fact the document does not hold
(``make_fact_shard.py --absent 0.2 --skip-facts <v2's facts>``: facts no earlier shard of the seed used). Its documents
split in two: the first half calibrates (theta, theta_end; ``k10_refuse.fit``, the same lines), the second is measured
once, through the model with ``copy_refuse`` set: the answer is the model's argmax from the answer run's first token
to the newline, refused when it is the refusal token (``</s>``). Beside it: run 5's own mix on the same questions, and
that the logits off the asking tokens are the same with the contract on and off (language unchanged).

    python validation/exp_refuse_450m.py --ckpt <spatial_link_5_final.pt> --probe <folder with the shard> [--docs 400]
-> validation/logs/exp_refuse_450m_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
from cubbyllm.model import spatial_refuse as sr  # noqa: E402

NAME, K, STEP = "facts_far1k_v2_absent", 16, (0, 0, 6)


def load_docs(probe: str, n: int):
    import exp_hypertoken_recall as hr
    hr.CORPUS = probe
    x, (places, roles) = hr.heldout_docs(NAME, n)
    absent = json.load(open(os.path.join(probe, f"{NAME}.absent.json"), encoding="utf-8"))[-len(x):]
    return x, places, roles, absent


def forward(model, x, places, roles):
    from recall_split import _forward
    return _forward(model, torch.from_numpy(x), torch.from_numpy(places), torch.from_numpy(roles))


def questions(tk, x, absent):
    """(doc, first answer token, last, absent) for every answer the probe renders."""
    from exp_hypertoken_recall import answers
    out = []
    for r, row in enumerate(x):
        spans = answers(tk, row)
        assert len(spans) == len(absent[r]), f"doc {r}: {len(spans)} answers, {len(absent[r])} flags"
        out += [(r, j, k, bool(a)) for (j, k), a in zip(spans, absent[r])]
    return out


@torch.no_grad()
def capture(model, tk, x, places, roles, absent, eos, batch) -> list:
    """Per question: (info, answer ids, reads ahead (K,), the copy's picks along the run) -- ``k10_refuse``'s rows."""
    cp = model.spatial.copy_path
    starts, bests, picks = [], [], {}
    real_runs, real_decide = sr.runs, sr.decide

    def runs(idx, S):
        st, off = real_runs(idx, S)
        starts.extend(idx[st.unique()].tolist())
        return st, off

    def decide(b, t, e):
        bests.append(b.clone())
        return real_decide(b, t, e)
    hook = cp.register_forward_hook(lambda _m, inp, o: picks.update(zip(inp[4].tolist(), o[1].argmax(-1).tolist())))
    cp.set_refuse(0.0, 0.0, eos, STEP, K)
    sr.runs, sr.decide, rows = runs, decide, []
    try:
        for b in range(0, len(x), batch):
            starts.clear(), bests.clear(), picks.clear()
            forward(model, x[b:b + batch], places[b:b + batch], roles[b:b + batch])
            S = x.shape[1] - 1
            by = dict(zip(starts, torch.cat(bests).numpy())) if bests else {}
            for r, j, k, ab in questions(tk, x[b:b + batch], absent[b:b + batch]):
                s0 = r * S + j - 1
                if s0 not in by:                              # the run did not start at the lead token
                    raise SystemExit(f"doc {b + r}: no answer run starts at {j - 1}")
                rows.append(({"absent": ab}, [int(t) for t in x[b + r, j:k + 1]], by[s0],
                             [picks.get(s0 + m) for m in range(K)]))
    finally:
        sr.runs, sr.decide = real_runs, real_decide
        hook.remove()
        cp.refuse = None
    return rows


@torch.no_grad()
def spoken(model, tk, x, places, roles, absent, eos, batch, refuse=None, end_id=None) -> dict:
    """The model's answers (argmax from the run's first token to the newline): with the contract, or run 5's mix."""
    nl = tk.encode("\n", add_special_tokens=False).ids[-1]
    if refuse is not None:
        model.spatial.copy_path.set_refuse(*refuse, eos, STEP, K, end_id=end_id)
    ex, ref, ab, lang = [], [], [], 0.0
    try:
        for b in range(0, len(x), batch):
            lg = forward(model, x[b:b + batch], places[b:b + batch], roles[b:b + batch])
            if refuse is not None and b == 0:                 # language unchanged: off the askers, the same logits
                model.spatial.copy_path.refuse, keep = None, model.spatial.copy_path.refuse
                off = forward(model, x[b:b + batch], places[b:b + batch], roles[b:b + batch])
                model.spatial.copy_path.refuse = keep
                asks = torch.from_numpy((roles[b:b + batch, :-1] & 2) != 0)
                lang = float((lg - off)[~asks].abs().max())
            pred = lg.argmax(-1).numpy()
            for r, j, k, a in questions(tk, x[b:b + batch], absent[b:b + batch]):
                said = []
                for t in pred[r, j - 1:j - 1 + K].tolist():
                    if t in (nl, eos):
                        said = said or [t]
                        break
                    said.append(t)
                if a:
                    ab.append(said[:1] not in ([eos], [nl]))
                else:
                    ex.append(said == [int(t) for t in x[b + r, j:k + 1]])
                    ref.append(said[:1] in ([eos], [nl]))
    finally:
        model.spatial.copy_path.refuse = None
    out = {"exact": float(np.mean(ex)), "present_refused": float(np.mean(ref)),
           "present_wrong": float(1 - np.mean(ex) - np.mean(ref)), "absent_answered": float(np.mean(ab)),
           "n_present": len(ex), "n_absent": len(ab)}
    if refuse is not None:
        out["max_logit_change_off_the_askers"] = lang
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tokenizer", default=os.environ.get("CUBBY_SPM") or os.path.join(
        os.environ.get("CUBBY_MODELS", ""), "base450m", "grilly", "tokenizer.json"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--theta", type=float, default=None, help="a fitted pair (with --theta-end): the test half only")
    ap.add_argument("--theta-end", type=float, default=None)
    ap.add_argument("--end", action="store_true", help="the contract says the answer's end (the newline) itself")
    c = ap.parse_args()
    torch.set_num_threads(c.threads)
    from tokenizers import Tokenizer
    import k10_refuse as kr
    import train_base as tb
    tk = Tokenizer.from_file(c.tokenizer)
    eos = tk.token_to_id("</s>")
    t0 = time.time()
    model = tb.load_base(c.ckpt, "cpu")                    # CubbyModel: no train mode to leave (no dropout)
    x, places, roles, absent = load_docs(c.probe, c.docs)
    h = len(x) // 2
    cal, test = (x[:h], places[:h], roles[:h], absent[:h]), (x[h:], places[h:], roles[h:], absent[h:])
    nl = tk.encode("\n", add_special_tokens=False).ids[-1]
    if c.theta is not None:                                # a fitted pair: the model's own answers on the test half only
        rep = {"checkpoint": os.path.basename(c.ckpt), "probe": NAME, "theta": c.theta, "theta_end": c.theta_end,
               "end_said_by_the_contract": c.end, "docs": {"test": len(x) - h},
               "test_model": spoken(model, tk, *test, eos, c.batch, (c.theta, c.theta_end), nl if c.end else None),
               "wall_s": round(time.time() - t0, 1)}
        out = os.path.join(HERE_DIR, "logs", f"exp_refuse_450m{'_end' if c.end else ''}_{c.date}.json")
        json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
        print(json.dumps(rep, indent=1))
        return
    f = kr.fit(capture(model, tk, *cal, eos, c.batch))
    rep = {"checkpoint": os.path.basename(c.ckpt), "probe": NAME, "docs": {"calibration": h, "test": len(x) - h},
           "lines": kr.LINES, "lookahead": K, **f, "test_mix": spoken(model, tk, *test, eos, c.batch)}
    if f["theta"] is not None:
        rep["test_simulated"] = kr.simulate(capture(model, tk, *test, eos, c.batch), f["theta"], f["theta_end"])
        rep["test_model"] = spoken(model, tk, *test, eos, c.batch, (f["theta"], f["theta_end"]))
    rep["wall_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE_DIR, "logs", f"exp_refuse_450m_{c.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
