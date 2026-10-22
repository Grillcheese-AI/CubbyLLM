"""refuse_diag_450m -- order 0029 step 1: where copy or refuse spends its refusals and its wrong answers (450M run 5,
0022's calibration documents only, nothing trained; CPU).

At each answer's first asking position the contract reads every piece ahead (``spatial_refuse``); this captures, per
piece ``k``: the best match among the document's writers, the runner-up (the margin), and which writer it is -- its
position, so its fact line. Along the run, the copy's gate lambda at each asking position. Then 0022's fitted pair
(theta, theta_end; ``exp_refuse_450m_2026-10-22.json``) decides every answer as the contract does, and the outcomes are
broken down:

- refusals by why (a true piece read below theta / the answer's end read above theta_end / no end within K / nothing
  read at all), by answer length, by the piece position that refused;
- wrong answers by kind: cut short or run on (the pieces right, the length not), a piece from another fact of the
  document (the same relation of another subject, or another relation), a piece of the right fact in the wrong place;
- every present answer by its weakest piece's match, its smallest margin, its piece count and its mean lambda (bins).

Each document's facts are its "Facts:" lines, ``<subject> <relation> <value>.``, the value's tokens the writers; a
question's own fact is the line whose value is its gold answer.

    python validation/refuse_diag_450m.py --ckpt <spatial_link_5_final.pt> --probe <0022's probe folder> [--docs 400]
    python validation/refuse_diag_450m.py --from-rows validation/logs/refuse_diag_450m_<date>.rows.json   (the summary again, no model)
-> validation/logs/refuse_diag_450m_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_DIR)
sys.path.insert(0, os.path.dirname(HERE_DIR))
import exp_refuse_450m as er  # noqa: E402
from cubbyllm.local_paths import value  # noqa: E402
from cubbyllm.model import spatial_refuse as sr  # noqa: E402

LOG0022 = os.path.join(HERE_DIR, "logs", "exp_refuse_450m_2026-10-22.json")


def fact_lines(tk, row: np.ndarray, roles: np.ndarray) -> dict:
    """writer position -> (fact line index, its relation words, its subject words, its value ids)."""
    out, line = {}, 0
    text_ids = row.tolist()
    starts = [0] + [i + 1 for i, t in enumerate(text_ids) if tk.decode([t]).endswith("\n")]
    for a, b in zip(starts, starts[1:] + [len(text_ids)]):
        w = [i for i in range(a, b) if roles[i] == 1]
        if not w:
            continue
        head = tk.decode(text_ids[a:w[0]]).strip().split()
        val = [text_ids[i] for i in w]
        for i in w:
            out[i] = (line, head, tuple(val))
        line += 1
    return out


@torch.no_grad()
def capture(model, tk, x, places, roles, absent, eos, batch):
    cp = model.spatial.copy_path
    rec = {"start": [], "best": [], "second": [], "arg": [], "lam": {}, "pick": {}}
    real_runs, real_bests = sr.runs, sr._bests

    def runs(idx, S):
        st, off = real_runs(idx, S)
        rec["start"].extend(idx[st.unique()].tolist())
        return st, off

    def bests(c, r, table, ids, ok):
        rn = F.normalize(r.float(), dim=-1)
        e = F.normalize(table.float()[ids], dim=-1)
        s = torch.einsum("nkd,cd->nkc", rn, e).masked_fill(~ok[:, None], float("-inf"))
        top = s.topk(2, dim=-1)
        rec["best"].append(top.values[..., 0].clone())
        rec["second"].append(top.values[..., 1].clone())
        rec["arg"].append(top.indices[..., 0].clone())
        return real_bests(c, r, table, ids, ok)

    def hook(_m, inp, o):
        rec["lam"].update(zip(inp[4].tolist(), o[2][:, 0].tolist()))
        rec["pick"].update(zip(inp[4].tolist(), o[1].argmax(-1).tolist()))   # what the copy says there (0022's rows)
    h = cp.register_forward_hook(hook)
    cp.set_refuse(0.0, 0.0, eos, er.STEP, er.K)
    sr.runs, sr._bests, rows = runs, bests, []
    try:
        for b in range(0, len(x), batch):
            for k in ("start", "best", "second", "arg"):
                rec[k].clear()
            rec["lam"].clear(), rec["pick"].clear()
            er.forward(model, x[b:b + batch], places[b:b + batch], roles[b:b + batch])
            S = x.shape[1] - 1
            writers = np.argwhere(roles[b:b + batch, :S] == 1)          # row-major: the ring's candidate order
            best, second, arg = (torch.cat(rec[k]).numpy() for k in ("best", "second", "arg"))
            at = {s: i for i, s in enumerate(rec["start"])}
            lines = [fact_lines(tk, x[b + r, :S], roles[b + r, :S]) for r in range(min(batch, len(x) - b))]
            for r, j, k, ab in er.questions(tk, x[b:b + batch], absent[b:b + batch]):
                s0 = r * S + j - 1
                i = at[s0]
                wpos = [tuple(writers[a]) for a in arg[i]]
                rows.append({"doc": b + r, "absent": ab, "gold": [int(t) for t in x[b + r, j:k + 1]],
                             "best": best[i].tolist(), "second": second[i].tolist(),
                             "piece": [int(x[b + wr, wc]) for wr, wc in wpos],
                             "line": [lines[wr].get(wc, (None,))[0] if wr == r else None for wr, wc in wpos],
                             "lines": {str(p): (v[0], v[1], list(v[2])) for p, v in lines[r].items()},
                             "lam": [rec["lam"].get(s0 + m) for m in range(er.K)],
                             "said": [rec["pick"].get(s0 + m) for m in range(er.K)]})
    finally:
        sr.runs, sr._bests = real_runs, real_bests
        h.remove()
        cp.refuse = None
    return rows


def outcome(row, theta, theta_end) -> dict:
    b = torch.tensor([row["best"]])
    L = int(sr.decide(b, theta, theta_end)[0])
    G = len(row["gold"])
    said = row["said"][:L]
    out = {"L": L, "G": G}
    if L == 0:
        bb = row["best"]
        end = next((k for k, v in enumerate(bb) if v < theta_end), None)
        if end == 0:
            why = "nothing read (the first read already ends)"
        elif end is None:
            why = "no end within K"
        else:
            unsure = next(k for k, v in enumerate(bb[:end]) if v < theta)
            why = "a true piece read below theta" if unsure < G else "the end read above theta_end"
            out["at"] = unsure
        out.update(kind="refused", why=why)
    elif row["absent"]:
        out["kind"] = "absent answered"
    elif said == row["gold"]:
        out["kind"] = "exact"
    else:
        gold_line = next((v[0] for v in row["lines"].values() if v[2] == row["gold"]), None)
        bad = [k for k in range(L) if k >= G or said[k] != row["gold"][k]]
        if all(said[k] == row["gold"][k] for k in range(min(L, G))):
            kind = "cut short" if L < G else "run on"
        else:
            k0 = next(k for k in bad if k < G)
            ln = row["line"][k0]
            if ln is None:
                kind = "a piece from no fact line"
            elif ln == gold_line:
                kind = "a piece of the right fact, in the wrong place"
            else:
                other = next((v for v in row["lines"].values() if v[0] == ln), None)
                mine = next((v for v in row["lines"].values() if v[0] == gold_line), None)
                same_rel = other is not None and mine is not None and other[1][-2:] == mine[1][-2:]
                kind = ("another fact: the same relation, another subject" if same_rel
                        else "another fact of the document")
        out["kind"] = "wrong: " + kind
    return out


def bins(vals, edges):
    return [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"], \
        [int(np.searchsorted(edges, v, side="right")) for v in vals]


def table(rows, outs, key_vals, edges):
    names, idx = bins(key_vals, edges)
    t = defaultdict(Counter)
    for i, o in zip(idx, outs):
        t[names[i]][o["kind"].split(":")[0]] += 1
    return {n: dict(t[n]) for n in names if n in t}


def followups(rows, outs, theta, theta_end) -> dict:
    """What the refused present answers' picks were (the copy's argmax over the gold's length: would the answer have been
    right?), the refusal rate by answer length, and where the absent questions' first reads sit against the pair."""
    ref = Counter()
    for r, o in zip(rows, outs):
        if not r["absent"] and o["kind"] == "refused":
            ref[f'{o["why"]} | {"picks = gold" if r["said"][:len(r["gold"])] == r["gold"] else "picks differ"}'] += 1
    by = defaultdict(Counter)
    for r, o in zip(rows, outs):
        if not r["absent"]:
            by[min(len(r["gold"]), 9)][o["kind"].split(":")[0]] += 1
    first = [r["best"][0] for r in rows if r["absent"]]
    return {"refused_present_picks": dict(ref),
            "refusal_rate_by_length": {("9+" if g == 9 else str(g)): {"n": sum(c.values()), "refused": round(c["refused"] / sum(c.values()), 3),
                                                                     "wrong": c["wrong"]} for g, c in sorted(by.items())},
            "absent_first_read": {"max": round(max(first), 4), "below_theta_end": round(sum(v < theta_end for v in first) / len(first), 3),
                                  "below_theta": round(sum(v < theta for v in first) / len(first), 3)},
            "wrong_length_off_by": dict(Counter(str(o["L"] - o["G"]) for r, o in zip(rows, outs)
                                                if not r["absent"] and o["kind"].startswith("wrong")))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--probe")
    ap.add_argument("--from-rows", help="recompute the summary from a saved .rows.json (no model; the capture's wall time kept)")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tokenizer", default=os.path.join(value("CUBBY_MODELS", ""), "base450m", "grilly", "tokenizer.json"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from tokenizers import Tokenizer
    fit = json.load(open(LOG0022, encoding="utf-8"))
    theta, theta_end = fit["theta"], fit["theta_end"]
    tk = Tokenizer.from_file(a.tokenizer)
    eos = tk.token_to_id("</s>")
    t0 = time.time()
    out = os.path.join(HERE_DIR, "logs", f"refuse_diag_450m_{a.date}.json")
    if a.from_rows:
        keys = ("doc", "absent", "gold", "best", "second", "piece", "line", "lines", "lam", "said")
        rows = [{k: r[k] for k in keys} for r in json.load(open(a.from_rows, encoding="utf-8"))]
        prev = json.load(open(a.from_rows.replace(".rows.json", ".json"), encoding="utf-8"))
        ckpt, h, wall = prev["checkpoint"], prev["docs_calibration"], prev["wall_s"]
    else:
        import train_base as tb
        model = tb.load_base(a.ckpt, "cpu")
        x, places, roles, absent = er.load_docs(a.probe, a.docs)
        h = len(x) // 2                                        # 0022's split: the first half calibrates; test unread
        rows = capture(model, tk, x[:h], places[:h], roles[:h], absent[:h], eos, a.batch)
        ckpt = os.path.basename(a.ckpt)
        wall = None
    outs = [outcome(r, theta, theta_end) for r in rows]
    pres = [(r, o) for r, o in zip(rows, outs) if not r["absent"]]
    pr, po = [r for r, _ in pres], [o for _, o in pres]
    weakest = [min(r["best"][:max(1, len(r["gold"]))]) for r in pr]
    margin = [min(b - s for b, s in zip(r["best"][:len(r["gold"])], r["second"][:len(r["gold"])])) for r in pr]
    lam = [float(np.mean([v for v in r["lam"][:len(r["gold"])] if v is not None] or [0.0])) for r in pr]
    n_ab = len(rows) - len(pr)
    agree = {"exact": sum(o["kind"] == "exact" for o in po) / len(po), "present_refused": sum(o["kind"] == "refused" for o in po) / len(po),
             "present_wrong": sum(o["kind"].startswith("wrong") for o in po) / len(po),
             "absent_answered": sum(o["kind"] == "absent answered" for o in outs) / max(1, n_ab)}
    agree = {k: round(v, 4) for k, v in agree.items()}
    rep = {"checkpoint": ckpt, "probe": er.NAME, "docs_calibration": h, "theta": theta,
           "reproduces_0022_calibration": {"here": agree, "0022": {k: round(fit["calibration"][k], 4) for k in agree}},
           "theta_end": theta_end, "n_present": len(pr), "n_absent": len(rows) - len(pr),
           "outcomes_present": dict(Counter(o["kind"] for o in po)),
           "outcomes_absent": dict(Counter(o["kind"] for r, o in zip(rows, outs) if r["absent"])),
           "refusals_by_why": dict(Counter(o["why"] for o in po if o["kind"] == "refused")),
           "refusal_at_piece": dict(Counter(str(o.get("at")) for o in po if o["kind"] == "refused" and "at" in o)),
           "by_length": {str(g): dict(Counter(o["kind"].split(":")[0] for o in po if o["G"] == g))
                         for g in sorted({o["G"] for o in po})},
           "by_weakest_match": table(pr, po, weakest, [0.085, 0.117, 0.15, 0.2, 0.3]),
           "by_smallest_margin": table(pr, po, margin, [0.0, 0.02, 0.05, 0.1, 0.2]),
           "by_mean_lambda": table(pr, po, lam, [0.25, 0.5, 0.75, 0.9]),
           "followups": followups(rows, outs, theta, theta_end),
           "wall_s": wall if wall is not None else round(time.time() - t0, 1)}
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    json.dump([{**r, **o} for r, o in zip(rows, outs)], open(out.replace(".json", ".rows.json"), "w", encoding="utf-8"))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("by_")}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
