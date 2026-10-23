"""diag_s12_copy -- order 0035, part B (no training): why run 5's copy did not supply a real village's names.

The copy mixes into the next-token distribution only at an ASKING token (``model/spatial_copy.py``). It reads the
scene there, finds the written token that best matches the read, and opens its gate lambda when the trunk's stream
(and the match: the read-aware gate) says so. Here, at every asking token:
- lambda;
- the best match's cosine;
- whether the token it points at is the token that comes next;
- for the questions, whether it is the answer's first token.

Each is read on:
- **village, as written**: 0034's villages (cut, the placed questions, ``$CUBBY_SCENES/questions.json``) rendered as
  0034 rendered them. The asking token is ``here``, and what follows it is "?" or a newline, not the answer;
- **village, lead**: the same documents, with the token right before each answer (the end of "Answer:") also asking
  at the question's cell. This is a diagnostic of the shape, not a fix: it says whether the copy would give the
  first token if it were asked where the answer is. Teacher-forced first-token rates by answer class come with it;
- **village, lead, written form**: as lead, with each answer in the token form its name was written in ("Answer:", a
  space, then the name encoded alone, as the far facts render it). The listing writes "coal" at a line's start; the
  answer after "Answer:" is " coal", another token, which no copy of a written token can give;
- **far facts**: ``facts_far1k_lead``'s held-out documents, the shape the copy was trained on (the token before an
  answer asks, and so does each answer piece);
- **shard villages**: ``spatial_here``'s held-out documents, the villages run 5 read at 0.953.

    CUBBY_SCENES=<folder> [CB_CORPUS=<token cache>] python validation/diag_s12_copy.py [--docs 40]
-> validation/logs/diag_s12_copy_<date>.json (counts and means only)

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cubbyllm.local_paths import expand  # noqa: E402
from cubbyllm.model.spatial import NO_PLACE  # noqa: E402
from exp_s12_scene_files import files, load_trunk, scenes_dir, sha, trunk_docs, views  # noqa: E402
from s12_questions import PLACED  # noqa: E402

__wiring__ = "STANDALONE"
ASK = 2


class Tap:
    """A forward hook on the copy: per asking position, (flat index, lambda, best cosine, the id it points at)."""

    def __init__(self, cp) -> None:
        self.rows = []
        cp.register_forward_hook(self)

    def __call__(self, mod, args, out) -> None:
        _logits, _h, r, table, idx, ids, ok = args
        rn = F.normalize(r.float(), dim=-1)
        e = F.normalize(table.float()[ids], dim=-1)
        s = rn @ e.T if ids.dim() == 1 else torch.einsum("nd,ncd->nc", rn, e)
        s = s.masked_fill(~ok, float("-inf"))
        best, at = s.max(-1)
        pick = (ids[at] if ids.dim() == 1 else ids.gather(1, at[:, None])[:, 0])
        lam = out[2][:, 0]
        for i in range(idx.numel()):
            self.rows.append((int(idx[i]), float(lam[i]), float(best[i]), int(pick[i])))


def run(model, tap, ids: np.ndarray, places: np.ndarray, roles: np.ndarray):
    """One document through the trunk; -> (the argmax at every position, {position: (lambda, best, pointed id)})."""
    x = torch.from_numpy(ids.astype(np.int64))[None]
    P = torch.from_numpy(places.astype(np.int64))[None]
    R = torch.from_numpy(roles.astype(np.int64))[None]
    on = P[..., 0] != NO_PLACE
    pos = torch.where(on.unsqueeze(-1), P, torch.zeros_like(P)).float()
    tap.rows.clear()
    with torch.no_grad():
        logits = model.forward(x[:, :-1], positions=pos[:, :-1], placed=on[:, :-1],
                               writes=(on & ((R & 1) != 0))[:, :-1], asks=(on & ((R & 2) != 0))[:, :-1])
    return logits.argmax(-1)[0].numpy(), {i: (lam, best, pick) for i, lam, best, pick in tap.rows}


def summary(rows: list[dict]) -> dict:
    """{class: n, mean lambda, median best, rates}: the counts the public log carries."""
    out = {}
    for c in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == c]
        out[c] = {"n": len(rs), "lambda_mean": round(float(np.mean([r["lam"] for r in rs])), 4),
                  "lambda_over_half": sum(r["lam"] > 0.5 for r in rs),
                  "best_median": round(float(np.median([r["best"] for r in rs])), 4),
                  "points_at_next_token": sum(r["next"] for r in rs)}
        if "answer" in rs[0]:
            out[c]["points_at_answer_first_token"] = sum(r["answer"] for r in rs)
        if "first" in rs[0]:
            out[c]["first_token_right"] = sum(r["first"] for r in rs)
        if "where" in rs[0]:
            out[c]["points_at"] = dict(collections.Counter(r["where"] for r in rs))
    return out


def _where(tk, sc, q, pick: int, answer) -> str:
    """What the copy pointed at: the answer's first piece, another piece of it, a piece of a thing within two steps of
    the asked cell (a neighbour's crosstalk), a piece of a thing further off, or no thing's token."""
    if len(answer) and pick == int(answer[0]):
        return "the answer's first piece"
    said = tk.decode([pick]).strip()
    if len(answer) and said and said == tk.decode([int(answer[0])]).strip():
        return "the answer's first piece, in its written form"   # "coal" written, " coal" answered: another token
    if pick in {int(t) for t in answer[1:]} or said in {tk.decode([int(t)]).strip() for t in answer[1:]}:
        return "another piece of the answer"
    cell = tuple(q["address"])
    near, far = set(), set()
    for name, c in sc.things.items():
        (near if sum(abs(c[i] - cell[i]) for i in range(3)) <= 2 else far).update(
            tk.encode(" " + name, add_special_tokens=False).ids + tk.encode(name, add_special_tokens=False).ids)
    return "a neighbour's piece" if pick in near else "a farther thing's piece" if pick in far else "no thing's token"


def written_form(tk, ids, places, roles, spans):
    """Each answer in the token form its name was written in: "Answer:" then a space token, then the name encoded alone
    (as the far facts render it), so a written token can be the answer's. -> the new arrays and spans."""
    from exp_hypertoken_recall import answers
    space = tk.encode(" ", add_special_tokens=False).ids
    ids, places, roles = list(ids), list(places), list(roles)
    for j, k in reversed(spans):                             # from the last: an earlier span does not move
        new = space + tk.encode(tk.decode([int(t) for t in ids[j:k + 1]]).strip(), add_special_tokens=False).ids
        ids[j:k + 1], places[j:k + 1], roles[j:k + 1] = new, [places[j]] * len(new), [0] * len(new)
    ids = np.asarray(ids)
    return ids, np.asarray(places), np.asarray(roles), answers(tk, ids)      # the spans read again, where they now are


def villages(model, tap, tk, eos: int, lead: bool, limit=None, form: bool = False) -> list[dict]:
    from exp_hypertoken_recall import answers
    d = scenes_dir()
    fixed = json.load(open(os.path.join(d, "questions.json"), encoding="utf-8"))
    rows = []
    for f in (files(d)[:limit] if limit else files(d)):
        sc = views(os.path.join(d, f))["cut"]
        qs = [q for q in fixed["villages"][f]["cut"] if q["shape"] in PLACED]
        for ids, places, roles, chunk in trunk_docs(tk, sc, qs, True, eos):
            spans, asks = answers(tk, ids), [i for i in range(len(ids)) if roles[i] & ASK]
            places, roles = places.copy(), roles.copy()
            if form:                                         # the answers in the written token form
                ids, places, roles, spans = written_form(tk, ids, places, roles, spans)
            if lead:                                         # the token before each answer also asks, at its cell
                for q, (j, _k) in zip(chunk, spans):
                    roles[j - 1] |= ASK
                    places[j - 1] = q["address"]
                asks = [j - 1 for j, _k in spans]
            pred, seen = run(model, tap, ids, places, roles)
            for q, i, (j, k) in zip(chunk, asks, spans):
                if i in seen:
                    lam, best, pick = seen[i]
                    rows.append({"cls": q["cls"], "lam": lam, "best": best, "next": pick == int(ids[i + 1]),
                                 "answer": pick == int(ids[j]), "where": _where(tk, sc, q, pick, ids[j:k + 1]),
                                 **({"first": bool(pred[j - 1] == ids[j])} if lead else {})})
    return rows


def shard(model, tap, name: str, docs: int) -> list[dict]:
    """The asking positions of a shard's held-out documents: does the copy point at the token that comes next?"""
    import exp_hypertoken_recall as hr
    hr.CORPUS = os.environ.get("CB_CORPUS") or expand("$CUBBY_TRAINING_DATA/token_cache_cont")
    x, places = hr.heldout_docs(name, docs)
    places, roles = places
    rows = []
    for b in range(len(x)):
        _pred, seen = run(model, tap, x[b], places[b], roles[b])
        for i, (lam, best, pick) in seen.items():
            rows.append({"cls": "all", "lam": lam, "best": best, "next": pick == int(x[b][i + 1])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", type=int, default=40, help="held-out documents per shard")
    ap.add_argument("--limit", type=int, default=None, help="scene files (default: all)")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    model, tk, meta = load_trunk()
    tap = Tap(model.spatial.copy_path)
    eos = tk.token_to_id("</s>")
    arms = {"village_as_written": villages(model, tap, tk, eos, False, a.limit),
            "village_lead": villages(model, tap, tk, eos, True, a.limit),
            "village_lead_written_form": villages(model, tap, tk, eos, True, a.limit, form=True),
            "far_facts_lead": shard(model, tap, "facts_far1k_lead", a.docs),
            "shard_villages": shard(model, tap, "spatial_here", a.docs)}
    out = {"what": "order 0035 B: run 5's copy at every asking token (lambda, best match, what it points at), on 0034's "
                   "villages as written and with the answer's lead token asking, beside the far facts and the shard's villages",
           "date": a.date, "copy": meta.get("spatial"), "questions_sha256": sha(os.path.join(scenes_dir(), "questions.json")),
           "arms": {k: summary(v) for k, v in arms.items()}}
    path = os.path.join(HERE, "logs", f"diag_s12_copy_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["arms"], indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
