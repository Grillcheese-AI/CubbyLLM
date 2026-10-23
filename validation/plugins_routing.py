"""plugins_routing -- order 0028's routing line: a plugin's ``examples`` wake it (>= 90%) and its ``not`` lines do not
(<= 10%), with ``tau_plugin`` tuned on them -- the lines written before any tuning (10 + 10 at first, 30 + 30 after
the review's addition 4, both languages).

Two readings of the plugins' own lines:
- **as the order has it**: the router indexes each plugin's description and examples, so an example scores against
  itself (the in-sample line, the one ``tau`` is tuned on);
- **leave one out**: each example scored with its own line taken out of the index -- how a new phrasing of the same
  need would fare (the honest number for generalisation).
For every ``tau`` on a grid: per plugin, examples that wake it, ``not`` lines that wake it, and the other plugins' lines
that wake it (the confusion table). The chosen ``tau``: the one with the most leave-one-out examples woken while no
plugin's ``not`` lines wake it more than 10% of the time, also read leave one out; ties go to the higher ``tau``. (Until
the review's addition 4 it was chosen on the in-sample line, which every ``tau`` up to 1 meets -- an example finds
itself -- so the choice drifted to whatever silenced the not-lines and new phrasings stopped waking anything:
``plugins_routing_n30_2026-10-22.json``.)

``--router logistic``: ``plugins_route.LogisticRouter`` (per-plugin logistic heads on the encoder; the not-lines are
training negatives, so they are scored leave one out too). ``--encoder minilm``: ``question_encoder.QuestionEncoder``.

``--heldout FILE``: Cowork's held-out set (``## <plugin>``, a numbered "should wake" list, then a numbered list of near
misses), scored ONCE at the chosen ``tau``, after the tuning above and never fed back into it. The log keeps the
counts and which other plugin woke, not the phrasings (the set stays held out). ``--second-check``: a wake counts only
when the woken plugin then claims the question (``cubbyllm.plugins.check.claims``: a source's or world's own ``covers``,
the thalamus for an adapter), as the server's loader counts it (order 0028, review 2); the router's count is kept too.
The plugins are approved for it in a temporary lock, never the real one, and nothing is loaded.

    python validation/plugins_routing.py [--dirs plugins] [--encoder overlap|fastword|potion] [--heldout FILE]
-> validation/logs/plugins_routing[_<encoder>]_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "standin")]

GRID = [round(0.20 + 0.025 * i, 3) for i in range(25)]          # 0.20 .. 0.80
EX_LINE, NOT_LINE = 0.9, 0.1                                    # the order's 9/10 and 1/10, as shares


def router(index, kind: str, enc, leave_out=None):
    from plugins_route import LogisticRouter, Router
    return LogisticRouter(index, enc, leave_out=leave_out) if kind == "logistic" else Router(index, enc=enc, leave_out=leave_out)


def score_all(index, kind: str, loo: bool, enc=None) -> dict:
    """{plugin: {"examples": [scores...], "not": [scores...]}}: every plugin's score for each of its lines (thresholds
    applied later). Leave one out: a line scored by a router built without it -- examples always, not-lines when they
    are training data (the logistic router)."""
    shared = router(index, kind, enc)
    out = {}
    for name in index.names():
        p = index[name]
        out[name] = {"examples": [], "not": []}
        for part, lines in (("examples", p.examples), ("not", p.not_)):
            for line in lines:
                own = loo and (part == "examples" or kind == "logistic")
                r = router(index, kind, enc, leave_out=" ".join(line.split())) if own else shared
                out[name][part].append(r.scores(line))
    return out


def table(scores: dict, tau: float) -> dict:
    out = {}
    for name, s in scores.items():
        row = {"examples": 0, "of": len(s["examples"]), "not": 0, "not_of": len(s["not"]), "others_woken_by_examples": {},
               "not_woke": []}
        for sc in s["examples"]:
            woke = [n for n, v in sc.items() if v >= tau]
            row["examples"] += name in woke
            for n in woke:
                if n != name:
                    row["others_woken_by_examples"][n] = row["others_woken_by_examples"].get(n, 0) + 1
        for i, sc in enumerate(s["not"]):
            if sc.get(name, 0.0) >= tau:
                row["not"] += 1
                row["not_woke"].append(i)
        out[name] = row
    return out


def meets(t: dict) -> bool:
    return all(r["examples"] >= EX_LINE * r["of"] and r["not"] <= NOT_LINE * r["not_of"] for r in t.values())


def quiet(t: dict) -> bool:
    return all(r["not"] <= NOT_LINE * r["not_of"] for r in t.values())


def read_heldout(path: str, plugin: str | None = None) -> dict:
    """{plugin: {"wake": [...], "near": [...]}}: numbered lines under ``## <plugin>``; a heading line mentioning "not"
    (or "near") after the first list starts the near misses. ``plugin``: a set written for one plugin, with no ``##``
    heading (v3, alerts only): its lists are that plugin's."""
    out, cur, part = ({plugin: {"wake": [], "near": []}} if plugin else {}), plugin, "wake"
    for line in open(path, encoding="utf-8"):
        line = line.rstrip()
        if line.startswith("## "):
            cur, part = line[3:].strip(), "wake"
            out[cur] = {"wake": [], "near": []}
        elif cur and re.match(r"^\d+\.\s", line):
            out[cur][part].append(re.sub(r"^\d+\.\s+", "", line))
        elif cur and line and re.search(r"\bnot\b|near", line, re.I) and out[cur]["wake"]:
            part = "near"
    return out


def heldout(index, sets: dict, tau: float, kind: str, enc=None, check=None) -> dict:
    r, out = router(index, kind, enc), {}
    r.tau = tau
    for name, s in sets.items():
        row = {"wake": 0, "wake_of": len(s["wake"]), "near": 0, "near_of": len(s["near"]), "others_woken": {}}
        if check is not None:
            row.update(wake_checked=0, near_checked=0, declined={})
        for q, key in [(q, "wake") for q in s["wake"]] + [(q, "near") for q in s["near"]]:
            woke = [n for n, _s in r.wake(q)]
            row[key] += name in woke
            if check is not None and name in woke:
                ok, why = check(name, q)
                if ok:
                    row[key + "_checked"] += 1
                else:
                    why = f"{key}: " + why.split(" (")[0]          # the reason's kind, no question text
                    row["declined"][why] = row["declined"].get(why, 0) + 1
            for n in woke:
                if n != name:
                    row["others_woken"][n] = row["others_woken"].get(n, 0) + 1
        out[name] = row
    return out


def second_check(idx):
    """``(plugin, question) -> (claims it, why)`` as the server's loader asks it: every plugin approved in a temporary
    lock (never the real one), the brain's thalamus (``needs_facts``) for an adapter; builds, loads nothing."""
    import tempfile
    tmp = tempfile.mkdtemp()
    os.environ["CUBBY_PLUGINS_LOCK"] = os.path.join(tmp, "lock.json")
    from cubbyllm.plugins import guard
    from cubbyllm.plugins.check import claims
    from cubbyllm.plugins.load import Loader
    for n in idx.names():
        guard.approve(idx[n], by="plugins_routing --second-check (a temporary lock)")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests", "brain"))     # test_serve's fixtures (order 0031)
    import serve as sv
    from test_serve import STORE, ChainEmitter, overlap_retriever
    brain = sv.CubbyServe(ChainEmitter(), overlap_retriever, STORE, route_tau=0.0)
    loader = Loader(idx, data_root=tmp, log=lambda *_a, **_k: None, router="nearest", thalamus=brain.needs_facts)
    return lambda name, q: claims(loader, name, q)[:2]


def encoder(kind: str):
    if kind == "fastword":
        from exp_m3_cot_pipeline import V4_TABLE
        from exp_m3_domain_routing import _load_semantic_words
        return _load_semantic_words().FastWordEncoder.from_npz(str(V4_TABLE))
    if kind == "minilm":
        from question_encoder import QuestionEncoder
        qe = QuestionEncoder.find()
        if qe is None:
            sys.exit("paraphrase-MiniLM-L6-v2 is not in the Hugging Face cache")
        return qe
    if kind == "potion":
        from ear_lexicon import Lexicon
        lex = Lexicon.find("minishlab/potion-retrieval-32M")
        if lex is None:
            sys.exit("potion-retrieval-32M is not in the Hugging Face cache")
        return lex
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dirs", default=os.path.join(os.path.dirname(HERE), "plugins"))
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--encoder", choices=["overlap", "fastword", "potion", "minilm"], default="overlap",
                    help="overlap: IDF word overlap; fastword: the serve stack's v4 table; potion: potion-retrieval-32M; "
                         "minilm: paraphrase-MiniLM-L6-v2")
    ap.add_argument("--router", choices=["nearest", "logistic"], default="nearest")
    ap.add_argument("--heldout", help="the held-out set, scored once at the chosen tau (counts only in the log)")
    ap.add_argument("--heldout-plugin", help="the plugin a one-plugin held-out set (no ## heading) is written for")
    ap.add_argument("--tag", default="", help="a suffix for the log's name")
    ap.add_argument("--tau", type=float, help="score the held-out set at this tau instead of the chosen one (the server as committed)")
    ap.add_argument("--second-check", action="store_true", help="count a held-out wake only when the plugin claims the question")
    a = ap.parse_args()
    enc = encoder(a.encoder)
    from plugins_index import discover
    idx = discover(a.dirs)
    rep = {"plugins": idx.names(), "lines": {n: [len(idx[n].examples), len(idx[n].not_)] for n in idx.names()},
           "encoder": a.encoder, "router": a.router, "line": {"examples": EX_LINE, "not": NOT_LINE}, "grid": {}}
    if a.router == "logistic" and enc is None:
        sys.exit("the logistic router needs an encoder")
    best = None
    t0 = time.perf_counter()
    s_in, s_loo = score_all(idx, a.router, False, enc), score_all(idx, a.router, True, enc)
    grid = GRID if a.router == "nearest" else [round(0.30 + 0.025 * i, 3) for i in range(28)]
    for tau in grid:
        ins, loo = table(s_in, tau), table(s_loo, tau)
        loo_hits = sum(loo[n]["examples"] for n in idx.names())
        rep["grid"][str(tau)] = {"meets": meets(ins), "loo_meets": meets(loo), "loo_quiet": quiet(loo), "in_sample": ins,
                                 "leave_one_out": loo}
        if quiet(loo) and (best is None or loo_hits >= best[1]):
            best = (tau, loo_hits)
    rep["grid_s"] = round(time.perf_counter() - t0, 1)
    rep["start_tau"] = 0.35
    rep["chosen_tau"] = best[0] if best else None
    rep["chosen"] = rep["grid"][str(best[0])] if best else None
    if a.heldout and (best or a.tau is not None):
        sets, t_h = read_heldout(a.heldout, a.heldout_plugin), a.tau if a.tau is not None else best[0]
        if not any(s["wake"] or s["near"] for s in sets.values()):
            raise SystemExit(f"{a.heldout}: no question read (a one-plugin set needs --heldout-plugin); nothing scored")
        rep["heldout"] = {"tau": t_h, "second_check": a.second_check,
                          "plugins": heldout(idx, sets, t_h, a.router, enc, second_check(idx) if a.second_check else None)}
    tag = ("" if a.encoder == "overlap" else "_" + a.encoder) + ("" if a.router == "nearest" else "_" + a.router)
    tag += a.tag and "_" + a.tag
    out = os.path.join(HERE, "logs", f"plugins_routing{tag}_{a.date}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for tau in ("0.35" if a.router == "nearest" else None, str(best[0]) if best else None):
        if tau is None:
            continue
        g = rep["grid"][tau]
        print(f"tau {tau}: meets {g['meets']}")
        for n in idx.names():
            i, l = g["in_sample"][n], g["leave_one_out"][n]
            print(f"  {n:18s} examples {i['examples']}/{i['of']} (leave-one-out {l['examples']}/{l['of']})  "
                  f"not {i['not']}/{i['not_of']} (leave-one-out {l['not']}/{l['not_of']})  others (loo) "
                  f"{l['others_woken_by_examples']}")
    if not best:
        print("no tau keeps every plugin's not-lines at or under 10% leave one out")
    for n, r in (rep.get("heldout") or {}).get("plugins", {}).items():
        print(f"  held out {n:18s} wakes {r['wake']}/{r['wake_of']}  near misses woken {r['near']}/{r['near_of']}  "
              f"others {r['others_woken']}" + (f"  | after the check: wakes {r['wake_checked']}/{r['wake_of']}, near "
                                               f"{r['near_checked']}/{r['near_of']}, declined {r['declined']}" if "declined" in r else ""))
    print("->", out)


if __name__ == "__main__":
    main()
