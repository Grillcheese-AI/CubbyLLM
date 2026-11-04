"""names_segment -- order 0046, R1b: the VM's names segment, built from the names extract (``names_extract``, CC0, the
wikidata-pq tables of Wikidata's dump of 2026-09-28) and the EN and FR Wikipedia redirects (``wiki_redirects``).

Every name becomes a key of CubeLang's layered store (``docs/STORE.md`` in the CubeLang repo); each fact under it is an
item the name names, its source the name's kinds:

  key     the name as written (the VM normalizes it: lowercase, punctuation to spaces)
  text    the item (Q-id); properties and lexemes are left out
  source  its kinds, e.g. ``label:en,label:mul`` or ``alias:fr`` or ``title:en`` or ``redirect:en`` -- one name and one
          item make one fact, whatever languages and kinds carry it

Redirects name an item through its article: ``from`` is a key for the item whose enwiki / frwiki title is ``to``. A
redirect to a section (a fragment) names part of an article, not the item: left out (written before scoring).

The store is built as shards, by a hash of the folded name, each small enough for ``cubelang knowledge build`` to hold
in memory; CubeLang opens the directory as one store, each shard a layer read through its own index.

    python validation/retrieval_sets/names_segment.py [--src $CUBBY_DRIVE/cubbyllm/r1b_names] [--shards 32]
-> $CUBBY_DATA_ROOT/retrieval/r1b_segment/shard_NN.cubeseg + manifest.json (the work files beside it, in _build/)

Wired: STANDALONE (a data builder).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

__wiring__ = "STANDALONE"

_NOT_ALNUM = re.compile(r"[\W_]+")


def fold(name: str) -> str:
    """The VM's key normalization, as near as Python has it (lowercase; what is not a letter or digit to one space):
    used only to route a name to its shard and to merge a name's kinds."""
    return _NOT_ALNUM.sub(" ", name.lower()).strip()


def shard_of(key: str, shards: int) -> int:
    return zlib.crc32(key.encode("utf-8")) % shards


def cubelang_exe() -> str:
    """The CubeLang binary with sharded stores and ``knowledge batch``: ``$CUBELANG_EXE``, else the cowork checkout's."""
    from cubbyllm.local_paths import value
    root = os.path.dirname(os.path.dirname(HERE))
    exe = value("CUBELANG_EXE") or os.path.join(os.path.dirname(root), "cubelang-cowork", "target", "release",
                                                 "cubelang.exe" if os.name == "nt" else "cubelang")
    if not os.path.isfile(exe):
        raise SystemExit("no CubeLang binary with sharded stores: set CUBELANG_EXE")
    return exe


def route(src: str, work: str, shards: int) -> dict:
    """Pass 1: every (name, item, kind) to its shard's work file. Returns counts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    outs = [open(os.path.join(work, f"shard_{i:02d}.tsv"), "w", encoding="utf-8") for i in range(shards)]
    c = {"names": 0, "kept": 0, "redirects": 0, "redirects_kept": 0, "redirect_fragment": 0, "redirect_no_item": 0}
    titles: dict[str, dict[str, str]] = {"en": {}, "fr": {}}

    def put(name: str, q: str, tag: str) -> None:
        k = fold(name)
        if k:
            outs[shard_of(k, shards)].write(f"{name.lower()}\t{q}\t{tag}\n")
            c["kept"] += 1

    for f in sorted(glob.glob(os.path.join(src, "names_*", "*.parquet"))):
        pf, t0 = pq.ParquetFile(f), time.time()
        for b in pf.iter_batches(batch_size=1 << 20, columns=["id", "name", "kind", "lang"]):
            b = b.filter(pc.starts_with(b.column("id"), "Q"))
            ids, names = b.column("id").to_pylist(), b.column("name").to_pylist()
            kinds, langs = b.column("kind").to_pylist(), b.column("lang").to_pylist()
            for q, n, k, lg in zip(ids, names, kinds, langs):
                c["names"] += 1
                if not n:
                    continue
                n = n.replace("\t", " ").replace("\n", " ")
                put(n, q, f"{k}:{lg}")
                if k == "title" and lg in titles:
                    titles[lg][n] = q
        print(f"{os.path.basename(f)}: {pf.metadata.num_rows:,} rows, {time.time() - t0:.0f} s", flush=True)
    for lg in ("en", "fr"):
        f = os.path.join(src, f"redirects_{lg}wiki.parquet")
        t0 = time.time()
        for b in pq.ParquetFile(f).iter_batches(batch_size=1 << 20):
            for fr, to, frag in zip(b.column("from").to_pylist(), b.column("to").to_pylist(), b.column("fragment").to_pylist()):
                c["redirects"] += 1
                if frag:
                    c["redirect_fragment"] += 1
                    continue
                q = titles[lg].get(to)
                if not q:
                    c["redirect_no_item"] += 1
                    continue
                put(fr.replace("\t", " "), q, f"redirect:{lg}")
                c["redirects_kept"] += 1
        print(f"redirects {lg}: {time.time() - t0:.0f} s", flush=True)
    for o in outs:
        o.close()
    return c


def build_shard(tsv: str, jsonl: str, seg: str, exe: str) -> dict:
    """Pass 2: one shard's names merged by (name, item), written as facts, built into a segment."""
    merged: dict[tuple[str, str], set] = {}
    for line in open(tsv, encoding="utf-8"):
        name, q, tag = line.rstrip("\n").split("\t")
        merged.setdefault((name, q), set()).add(tag)
    with open(jsonl, "w", encoding="utf-8") as f:
        for (name, q), tags in sorted(merged.items()):
            f.write(json.dumps({"key": name, "text": q, "source": ",".join(sorted(tags))}, ensure_ascii=False) + "\n")
    r = subprocess.run([exe, "knowledge", "build", jsonl, "-o", seg], capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"{seg}: {r.stderr.strip()}")
    return {"facts": len(merged), "built": r.stdout.strip(), "bytes": os.path.getsize(seg)}


def main() -> None:
    from cubbyllm.local_paths import expand
    from retrieval_sets.common import data_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="the extract's folder (default $CUBBY_DRIVE/cubbyllm/r1b_names)")
    ap.add_argument("--shards", type=int, default=32)
    a = ap.parse_args()
    src = a.src or expand("$CUBBY_DRIVE/cubbyllm/r1b_names")
    out = os.path.join(data_root(), "r1b_segment")
    work = os.path.join(out, "_build")
    os.makedirs(work, exist_ok=True)
    if glob.glob(os.path.join(out, "*.cubeseg")):
        raise SystemExit(f"{out} already holds segments: move them aside first")
    exe, t0 = cubelang_exe(), time.time()
    counts = route(src, work, a.shards)
    print(f"routed in {time.time() - t0:.0f} s: {counts}", flush=True)
    shards = []
    for i in range(a.shards):
        s = build_shard(os.path.join(work, f"shard_{i:02d}.tsv"), os.path.join(work, f"shard_{i:02d}.jsonl"),
                        os.path.join(out, f"shard_{i:02d}.cubeseg"), exe)
        shards.append(s)
        print(f"shard {i:02d}: {s['facts']:,} facts, {s['bytes'] / 1e6:.0f} MB ({time.time() - t0:.0f} s)", flush=True)
    names_man = json.load(open(glob.glob(os.path.join(src, "names_*", "manifest.json"))[0], encoding="utf-8"))
    red_man = json.load(open(os.path.join(src, "redirects_manifest.json"), encoding="utf-8"))
    man = {"what": "order 0046 R1b: names, aliases, titles and redirects to items, as CubeLang knowledge keys",
           "names_rev": names_man.get("rev"), "redirect_dumps": {d["wiki"]: {t: d.get(t, {}).get("modified") for t in ("page", "redirect")}
                                                                 for d in red_man},
           "shards": a.shards, "counts": counts, "facts": sum(s["facts"] for s in shards),
           "bytes": sum(s["bytes"] for s in shards), "built_s": round(time.time() - t0), "per_shard": shards,
           "rule": "a fact is (name, item); its source the name's kinds; redirects to a section left out"}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"{man['facts']:,} facts, {man['bytes'] / 1e9:.2f} GB in {a.shards} shards, {man['built_s']} s -> {out}")


if __name__ == "__main__":
    main()
