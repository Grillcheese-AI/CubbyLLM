"""exp_s12_scene_files -- order 0034 (H-S12 gate 3): villages from scene files, read by the host and by the trunk.

The scene files (``cubby.scene/1``, ``cubbyllm.spatial.scene_file``) are in a folder named by ``CUBBY_SCENES``, outside
this repo: a world's own exporter writes them. Every village is read twice:
- **cut**: cropped round its centre to the spatial shard's size (at most 32 things, a 24-cell square, 3 levels);
- **whole**: moved to the origin and nothing left out, reported apart as a capacity check.

    python validation/exp_s12_scene_files.py fix   [--seed 0]   questions for every file, written next to them
    python validation/exp_s12_scene_files.py run   [--no-trunk] [--limit N]

``fix`` draws each village's 30 questions (``s12_questions``) and writes them, with the hash of every scene file, to
``$CUBBY_SCENES/questions.json``. ``run`` reads that file and refuses when it is missing or a scene changed since:
the questions are fixed before any run.

**Host:** a fresh ``AskLoop(scene=)`` per village. The grammar is one template per relation the scene states
(``s12_questions.forms``); nothing is fetched but the scene; the CubeLang VM verifies. A question is *routed* when the
loop sent it to the scene. Each is spoken right, spoken wrong (the kill line) or refused.

**Trunk:** run 5's checkpoint (``CB_CKPT``) through torch, with the link and the copy. Each village is rendered as
the shard renders one (``make_spatial_shard.render``): the things along the Hilbert curve, their cells on their name
tokens, and a placed question's ``here`` carrying the host's cell. Answers are scored teacher-forced as
``exp_hypertoken_recall`` scores them, by answer class (``name``, ``kind``, ``nothing``, ...).
- The three placed shapes (neighbour, offset, at) are what run 5 was trained on.
- The other four are asked unplaced and reported apart: run 5 never saw them.

Logs: the questions and every answer, with the villages' names and cells, go to ``$CUBBY_SCENES``. This repo gets
only the counts (``validation/logs/exp_s12_scene_files_<date>.json``): no name, no cell, no kind word.

Wired: STANDALONE (a measurement).
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.local_paths import expand  # noqa: E402
from cubbyllm.spatial.scene import Scene  # noqa: E402
from cubbyllm.spatial.scene_file import crop, load  # noqa: E402
from s12_questions import PLACED, forms, questions  # noqa: E402

__wiring__ = "STANDALONE"

CKPT = os.environ.get("CB_CKPT") or expand("$CUBBY_DRIVE/cubbyllm/runs/spatial_link_5/spatial_link_5_final.pt")
SPM = os.environ.get("CUBBY_SPM") or expand("$CUBBY_MODELS/base450m/grilly/tokenizer.json")


def scenes_dir() -> str:
    d = os.environ.get("CUBBY_SCENES")
    if not d or not os.path.isdir(d):
        raise SystemExit("set CUBBY_SCENES to the folder of scene files (outside this repo)")
    return d


def files(d: str) -> list[str]:
    return sorted(f for f in os.listdir(d) if f.endswith(".json") and not f.startswith(("questions", "exp_s12")))


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def views(path: str) -> dict:
    """{"cut": the cropped scene, "whole": the whole one}, both moved to the origin. The centre is the file's
    ``meta.centre`` (the exporter's: the village's own centre), else the middle of its things."""
    d = load(path)
    sc = Scene.from_json(d)
    cs = list(sc.cells)
    lo = [min(c[i] for c in cs) for i in range(3)]
    hi = [max(c[i] for c in cs) for i in range(3)]
    centre = (d.get("meta") or {}).get("centre") or [(lo[i] + hi[i]) // 2 for i in range(2)] + [lo[2]]
    reach = max(abs(c[i] - centre[i]) for c in cs for i in range(2))
    whole = crop(sc, (centre[0], centre[1], lo[2]), max_things=10**9, side=2 * reach + 1, height=hi[2] - lo[2] + 1)
    return {"cut": crop(sc, centre), "whole": whole}


def fix(seed: int) -> None:
    d = scenes_dir()
    out = {"seed": seed, "fixed": time.strftime("%Y-%m-%dT%H:%M:%S"), "villages": {}}
    for i, f in enumerate(files(d)):
        v = views(os.path.join(d, f))
        out["villages"][f] = {"sha256": sha(os.path.join(d, f)),
                              **{k: questions(sc, seed * 1_000_003 + i * 2 + (k == "whole")) for k, sc in v.items()}}
    path = os.path.join(d, "questions.json")
    if os.path.exists(path):
        os.replace(path, path + f".old-{time.strftime('%Y%m%d%H%M%S')}")       # moved aside, never overwritten
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"{len(out['villages'])} villages, {sum(len(v['cut']) + len(v['whole']) for v in out['villages'].values())} "
          f"questions fixed -> {path} (sha256 {sha(path)[:16]})")


class NoSource:
    name = "none"

    def facts(self, entity, **kw):
        return []


def host(sc: Scene, qs: list[dict], vm) -> list[dict]:
    from cubbyllm.reasoning.ask import AskLoop
    from cubbyllm.reasoning.planner import normalize
    from cubbyllm.reasoning.question_forms import Forms, TemplateEmitter
    from cubbyllm.worlds.base import FactStore
    loop = AskLoop(TemplateEmitter(Forms(forms(sc))), world=FactStore([], name="scene facts"), source=NoSource(),
                   lexicon=False, scene=sc, run_fn=vm)
    out = []
    for q in qs:
        rec = loop.ask(q["question"])
        spoken = bool(rec.get("verified"))
        ok = spoken and normalize(rec.get("answer") or "") == normalize(q["answer"])
        out.append({"routed": rec.get("world") == "scene", "spoken": spoken, "right": ok, "wrong": spoken and not ok,
                    "said": rec.get("answer"), "reason": rec.get("reason")})
    return out


def trunk_docs(tk, sc: Scene, qs: list[dict], placed: bool, eos: int, lead: bool | str = False):
    """The documents a village's questions are asked in: the village, then as many questions as fit a 512-token
    document (a village longer than that gets the smallest doubling that holds it); none is dropped."""
    from make_spatial_shard import render
    qa = [(("What is at here?" if q["shape"] == "at" else q["question"]) if placed else q["question"], q["answer"],
           tuple(q["address"]) if placed and q["address"] is not None else None) for q in qs]
    docs, i = [], 0
    while i < len(qa):
        for doc in (512, 1024, 2048, 4096, 8192):
            try:
                n = len(qa) - i
                while n > 0:
                    ids, places, roles = render(tk, sc, qa[i:i + n], "places", eos, doc, lead)   # lead: 0035 C
                    if _count_answers(tk, ids) == n:
                        break
                    n -= 1
                if n > 0:
                    break
            except ValueError:                           # the village alone does not fit this length
                continue
        docs.append((ids, places, roles, qs[i:i + n]))
        i += n
    return docs


def _count_answers(tk, ids) -> int:
    from exp_hypertoken_recall import answers
    return len(answers(tk, ids))


def trunk(model, tk, sc: Scene, qs: list[dict], placed: bool, eos: int) -> list[dict]:
    import numpy as np
    import torch
    from exp_hypertoken_recall import answers
    from cubbyllm.model.spatial import NO_PLACE
    out = []
    for ids, places, roles, chunk in trunk_docs(tk, sc, qs, placed, eos):
        x = torch.from_numpy(ids.astype(np.int64))[None]
        P = torch.from_numpy(places.astype(np.int64))[None]
        R = torch.from_numpy(roles.astype(np.int64))[None]
        on = P[..., 0] != NO_PLACE
        pos = torch.where(on.unsqueeze(-1), P, torch.zeros_like(P)).float()
        with torch.no_grad():
            logits = model.forward(x[:, :-1], positions=pos[:, :-1], placed=on[:, :-1],
                                   writes=(on & ((R & 1) != 0))[:, :-1], asks=(on & ((R & 2) != 0))[:, :-1])
        pred = logits.argmax(-1)[0].numpy()
        for q, (j, k) in zip(chunk, answers(tk, ids)):
            ok = [pred[t - 1] == ids[t] for t in range(j, k + 1)]
            out.append({"first": bool(ok[0]), "exact": bool(all(ok)), "said": tk.decode([int(pred[t - 1]) for t in range(j, k + 1)])})
    return out


def load_trunk():
    import torch
    from train_base import build_model
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(dict(ck["meta"]), torch.device("cpu"))
    with torch.no_grad():
        for p, s in zip(model.parameters(), ck["params"]):
            p.copy_(s)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(SPM)
    return model, tk, dict(ck["meta"])


def tally(rows: list[dict], key: str) -> dict:
    """{shape or class: {n, and each flag's count}} over ``rows`` (the counts the public log carries)."""
    out: dict = {}
    for r in rows:
        t = out.setdefault(r[key], collections.Counter())
        t["n"] += 1
        for f in ("routed", "spoken", "right", "wrong", "exact", "first"):
            if r.get(f):
                t[f] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def _gates(trunk_rows: list[dict]) -> dict:
    """The order's lines on the placed shapes (what run 5 was trained to read): kinds >= 0.90, names >= 0.80,
    "nothing" apart; the kill under 0.75 on kinds."""
    def rate(c):
        rs = [r for r in trunk_rows if r["placed"] and r["cls"] == c]
        return (round(sum(r["exact"] for r in rs) / len(rs), 4) if rs else None), len(rs)
    (kind, nk), (name, nn), (nothing, nz) = rate("kind"), rate("name"), rate("nothing")
    return {"kind": kind, "kind_n": nk, "kind_line": 0.90, "name": name, "name_n": nn, "name_line": 0.80,
            "nothing": nothing, "nothing_n": nz, "killed": kind is not None and kind < 0.75}


def run(no_trunk: bool, limit: int | None, date: str) -> None:
    d = scenes_dir()
    qpath = os.path.join(d, "questions.json")
    if not os.path.exists(qpath):
        raise SystemExit("no questions.json: run `fix` first (the questions are fixed before any run)")
    fixed = json.load(open(qpath, encoding="utf-8"))
    names = files(d)[:limit] if limit else files(d)
    changed = [f for f in names if fixed["villages"].get(f, {}).get("sha256") != sha(os.path.join(d, f))]
    if changed:
        raise SystemExit(f"{len(changed)} scene files are new or changed since the questions were fixed: run `fix` again")
    t0 = time.time()
    from cubbyllm.bridges import cubelang_client as cc
    session = cc.CubelangSession()
    vm = lambda source, fn: session.run(source, fn=fn)  # noqa: E731 -- one VM session for every village
    model = tk = eos = None
    if not no_trunk:
        model, tk, _meta = load_trunk()
        eos = tk.token_to_id("</s>")
    rows, sizes = [], {"cut": [], "whole": []}
    for f in names:
        for view, sc in views(os.path.join(d, f)).items():
            qs = fixed["villages"][f][view]
            h = host(sc, qs, vm)
            t_in = iter(trunk(model, tk, sc, [q for q in qs if q["shape"] in PLACED], True, eos) if model else [])
            t_out = iter(trunk(model, tk, sc, [q for q in qs if q["shape"] not in PLACED], False, eos) if model else [])
            for q, hr in zip(qs, h):
                tr = (next(t_in) if q["shape"] in PLACED else next(t_out)) if model else None
                rows.append({"village": f, "view": view, **q, "host": hr, "trunk": tr})
            crop_m = sc.meta.get("crop") or {}
            sizes[view].append({"things": len(sc.things), "left_out": crop_m.get("left_out", 0)})
        print(f"  {f}: {len(rows)} answers so far [{time.time() - t0:.0f}s]", flush=True)
    detail = os.path.join(d, f"exp_s12_scene_files_{date}_detail.json")
    json.dump({"questions_sha256": sha(qpath), "rows": rows}, open(detail, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pub = {"what": "order 0034 (H-S12 gate 3): villages from scene files, the host's AskLoop(scene=) and run 5's trunk; "
                   "counts only (the villages, their names and cells stay in the private folder)",
           "date": date, "ckpt": None if no_trunk else os.path.basename(CKPT), "questions_sha256": sha(qpath),
           "scenes_sha256": hashlib.sha256("".join(fixed["villages"][f]["sha256"] for f in names).encode()).hexdigest(),
           "villages": len(names), "wall_s": round(time.time() - t0, 1), "views": {}}
    for view in ("cut", "whole"):
        rs = [r for r in rows if r["view"] == view]
        th = sorted(x["things"] for x in sizes[view])
        hrows = [{"shape": r["shape"], **r["host"]} for r in rs]
        trows = [{"cls": r["cls"], "shape": r["shape"], "placed": r["shape"] in PLACED, **r["trunk"]} for r in rs if r["trunk"]]
        v = {"questions": len(rs), "things": {"min": th[0], "median": th[len(th) // 2], "max": th[-1]} if th else None,
             "left_out_median": sorted(x["left_out"] for x in sizes[view])[len(th) // 2] if th else None,
             "host": {"by_shape": tally(hrows, "shape"), "routed": sum(r["routed"] for r in hrows),
                      "spoken": sum(r["spoken"] for r in hrows), "right": sum(r["right"] for r in hrows),
                      "wrong": sum(r["wrong"] for r in hrows),
                      "spoken_routed": sum(r["spoken"] for r in hrows if r["routed"]),
                      "routed_spoken_right": sum(r["right"] for r in hrows if r["routed"])}}
        if trows:
            v["trunk"] = {"placed_by_class": tally([r for r in trows if r["placed"]], "cls"),
                          "placed_by_shape": tally([r for r in trows if r["placed"]], "shape"),
                          "unplaced_by_class": tally([r for r in trows if not r["placed"]], "cls"),
                          "unplaced_by_shape": tally([r for r in trows if not r["placed"]], "shape"), "gates": _gates(trows)}
        pub["views"][view] = v
    if os.path.exists(os.path.join(d, "act_log.jsonl")):          # the world's act arm, counted (s12_act)
        from s12_act import counts
        pub["act"] = counts(os.path.join(d, "act_log.jsonl"))
    path = os.path.join(HERE, "logs", f"exp_s12_scene_files_{date}.json")
    json.dump(pub, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: pub[k] for k in ("villages", "wall_s")}), "->", path, "| detail ->", detail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("fix", "run"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-trunk", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    fix(a.seed) if a.what == "fix" else run(a.no_trunk, a.limit, a.date)


if __name__ == "__main__":
    main()
