"""diag_s12_names -- order 0034's diagnosis, reported apart from its gate: is the trunk's miss on real villages the
WORDS (names it never saw as village things) or the PLACES (layouts unlike the synthetic ones)?

The same scene files, the same cells and the same fixed questions (``$CUBBY_SCENES/questions.json``, the placed
shapes), asked twice through run 5's trunk:
- as exported;
- with every thing renamed to a word of the spatial shard's for its kind (``make_spatial_shard.KINDS``: a building
  becomes a hut, a forge, a barn ...; a villager becomes a road, a gate ...; numbered as the shard numbers).
The answers follow the names; the addresses do not move. If the renamed villages read near the shard's 0.95, the
places transfer and the words do not.

    CUBBY_SCENES=<folder> python validation/diag_s12_names.py [--limit N]
-> validation/logs/diag_s12_names_<date>.json (counts only)

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

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cubbyllm.spatial.scene import Scene  # noqa: E402
from exp_s12_scene_files import files, load_trunk, scenes_dir, sha, trunk, views  # noqa: E402
from make_spatial_shard import KINDS  # noqa: E402
from s12_questions import PLACED  # noqa: E402

__wiring__ = "STANDALONE"
AS = {"building": "building", "light": "light", "ore": "stone", "tree": "tree", "field": "way", "villager": "way"}


def renamed(sc: Scene) -> tuple[Scene, dict]:
    """The scene with every thing under a shard word of its kind, numbered as the shard numbers; and old -> new."""
    out, used, rename = Scene(sc.ssp), collections.Counter(), {}
    for c, n in sorted(sc.cells.items()):
        words = KINDS[AS.get(sc.kinds.get(n) or "", "way")]
        base = words[int(hashlib.blake2b(n.encode(), digest_size=4).hexdigest(), 16) % len(words)]
        used[base] += 1
        rename[n] = base if used[base] == 1 else f"{base} {used[base]}"
        out.place(rename[n], c, sc.kinds.get(n))
    for r, (lo, hi) in sc.regions.items():
        out.region(r, lo, hi)
    return out, rename


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    a = ap.parse_args()
    d = scenes_dir()
    fixed = json.load(open(os.path.join(d, "questions.json"), encoding="utf-8"))
    names = files(d)[:a.limit] if a.limit else files(d)
    model, tk, _meta = load_trunk()
    eos = tk.token_to_id("</s>")
    tally = {arm: collections.defaultdict(collections.Counter) for arm in ("as_exported", "renamed")}
    for f in names:
        sc = views(os.path.join(d, f))["cut"]
        qs = [q for q in fixed["villages"][f]["cut"] if q["shape"] in PLACED]
        new, rename = renamed(sc)
        qs2 = []
        for q in qs:
            ans = new.at(tuple(q["address"]))
            subj = rename.get(q["subject"], q["subject"]) if q["subject"] else None
            text = q["question"].replace(f" of {q['subject']}?", f" of {subj}?") if subj else q["question"]
            qs2.append({**q, "question": text, "answer": ans or "nothing"})
        for arm, scene, qq in (("as_exported", sc, qs), ("renamed", new, qs2)):
            for q, r in zip(qq, trunk(model, tk, scene, qq, True, eos)):
                t = tally[arm]["nothing" if q["answer"] == "nothing" else "thing"]
                t["n"] += 1
                t["exact"] += int(r["exact"])
                t["first"] += int(r["first"])
        print(f"  {f}", {arm: {k: dict(v) for k, v in t.items()} for arm, t in tally.items()}, flush=True)
    out = {"what": "order 0034's diagnosis: the same villages' placed questions through run 5, as exported and with every thing "
                   "renamed to a spatial-shard word of its kind (same cells, same addresses); counts only",
           "date": a.date, "villages": len(names), "view": "cut", "questions_sha256": sha(os.path.join(d, "questions.json")),
           "arms": {arm: {k: {**dict(v), "rate": round(v["exact"] / v["n"], 4) if v["n"] else None} for k, v in t.items()}
                    for arm, t in tally.items()}}
    path = os.path.join(HERE, "logs", f"diag_s12_names_{a.date}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["arms"]), "->", path)


if __name__ == "__main__":
    main()
