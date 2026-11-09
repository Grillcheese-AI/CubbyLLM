"""validation/fact_store/dump_count.py -- order 0055 step 1: one pass over the dump's parts, never unpacked to disk.

The parts (``dump_fetch.py``) are read in order and decompressed by ``lbzip2`` on every core (Python's ``bz2`` when it
is absent: slow, for tests); blocks of whole lines go to worker processes (at most two a worker in flight, so memory
stays bounded). Each entity is parsed once and counted (``count_entity``); the 1% hash sample (``in_sample``: the
sha1 of its id, never the first Q-ids) is written as raw JSON lines, gzipped. Written to ``--out``:
  counts.json      entities by type; items' statements by rank and snak type, main values by datatype; qualifier
                   snaks (time qualifiers apart) and references (blocks, snaks, distinct by hash, estimated); EN, FR
                   and ``mul`` labels, descriptions and aliases; enwiki/frwiki sitelinks; scholarly articles (P31
                   Q13442814) and their statements; statements per property; bytes by section on the sample, x100
  sample_1pct.jsonl.gz   the sample's entities, as in the dump
  progress.json    every few minutes: bytes read, entities, rate
  resume.json      every 15 minutes: a checkpoint; a rerun resumes from it (``dump_resume.py``), then it is renamed
                   resume.done.json. The sample is written in closed segments meanwhile, joined at the end.
The pass also hashes the compressed parts (``hash_parts``, checked against ``manifest.json``'s sha1 or md5 when the
download wrote one), stops if lbzip2 fails, and records whether the stream ended at the dump's closing ``]``.

    python validation/fact_store/dump_count.py --parts <dir> --out <dir> [--workers 8] [--stage <local dir>]

Wired: STANDALONE (order 0055's build; Colab, Nick's run).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import time

try:
    from . import dump_resume as R
    from .dump_stream import blocks, part_list, stream
except ImportError:                                           # run as a script from its folder (Colab)
    import dump_resume as R
    from dump_stream import blocks, part_list, stream

__wiring__ = "STANDALONE"

SAMPLE_MOD = 100
TIME_Q = {"P580", "P582", "P585"}
SCHOLARLY = "Q13442814"
LANGS = ("en", "fr", "mul")
SECTIONS = ("labels", "descriptions", "aliases", "claims", "sitelinks")
HLL_BITS = 14


def in_sample(eid: str) -> bool:
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16) % SAMPLE_MOD == 0


def _hll_add(reg: bytearray, hexhash: str) -> None:
    x = int(hexhash[:16], 16)
    i, rest = x >> (64 - HLL_BITS), (x << HLL_BITS) & ((1 << 64) - 1)
    rank = 1
    while rank <= 64 - HLL_BITS and not rest & (1 << 63):
        rank, rest = rank + 1, (rest << 1) & ((1 << 64) - 1)
    reg[i] = max(reg[i], rank)


def hll_estimate(reg: bytes) -> int:
    m = len(reg)
    z = sum(2.0 ** -r for r in reg)
    est = 0.7213 / (1 + 1.079 / m) * m * m / z
    zeros = reg.count(0)
    if est <= 2.5 * m and zeros:
        import math
        est = m * math.log(m / zeros)
    return int(est)


def _item(st: dict) -> str | None:
    v = (((st.get("mainsnak") or {}).get("datavalue")) or {}).get("value")
    return v.get("id") if isinstance(v, dict) else None


def count_entity(e: dict, c: collections.Counter, props: collections.Counter, reg: bytearray) -> None:
    t = e.get("type", "?")
    c[f"entities:{t}"] += 1
    for sec in ("labels", "descriptions"):
        for lg in LANGS:
            if lg in (e.get(sec) or {}):
                c[f"{sec}:{lg}"] += 1
    for lg in LANGS:
        c[f"aliases:{lg}"] += len((e.get("aliases") or {}).get(lg) or [])
    for site in ("enwiki", "frwiki"):
        if site in (e.get("sitelinks") or {}):
            c[f"sitelinks:{site}"] += 1
    claims = e.get("claims") or {}
    scholarly = SCHOLARLY in {_item(s) for s in claims.get("P31") or []}
    if scholarly:
        c["items:scholarly"] += 1
    for pid, sts in claims.items():
        props[pid] += len(sts)
        for st in sts:
            ms = st.get("mainsnak") or {}
            c[f"statements:{t}"] += 1
            c[f"rank:{st.get('rank')}"] += 1
            c[f"snak:{ms.get('snaktype')}"] += 1
            c[f"datatype:{ms.get('datatype')}"] += 1
            if scholarly:
                c["statements:scholarly"] += 1
            for qp, qs in (st.get("qualifiers") or {}).items():
                c["qualifier_snaks"] += len(qs)
                if qp in TIME_Q:
                    c["qualifier_snaks:time"] += len(qs)
            for ref in st.get("references") or []:
                c["reference_blocks"] += 1
                c["reference_snaks"] += sum(len(v) for v in (ref.get("snaks") or {}).values())
                if ref.get("hash"):
                    _hll_add(reg, ref["hash"])


def work(block: bytes) -> tuple[dict, dict, bytes, list[bytes], list[bytes]]:
    try:
        import orjson
    except ImportError:                                       # tests here; Colab installs orjson
        import json as _j

        class orjson:                                         # noqa: N801 -- the two calls used
            loads = staticmethod(_j.loads)
            dumps = staticmethod(lambda o: _j.dumps(o, separators=(",", ":")).encode())
    c, props, reg, sample = collections.Counter(), collections.Counter(), bytearray(1 << HLL_BITS), []
    for line in block.split(b"\n"):
        line = line.strip()
        if line.endswith(b","):
            line = line[:-1]
        if not line or line in (b"[", b"]"):
            continue
        e = orjson.loads(line)
        count_entity(e, c, props, reg)
        c["bytes"] += len(line)
        if e.get("id") and in_sample(e["id"]):
            sample.append(line)
            for sec in SECTIONS:
                c[f"sample_bytes:{sec}"] += len(orjson.dumps(e.get(sec))) if e.get(sec) else 0
            c["sample_entities"] += 1
            c["sample_bytes"] += len(line)
    return dict(c), dict(props), bytes(reg), sample, []


class Fmt:
    """What differs between the dump formats: the worker, the cut between blocks (whole entities), the sample's and
    the side file's names, the dump's last character, the sections sized on the sample, and a last step on the
    result. JSON (Wikimedia's dump) here; Turtle (the WikiKGQA dump) in ``ttl_count.py``."""
    work, sep, sample, side, end, sections = staticmethod(work), b"\n", "sample_1pct.jsonl.gz", None, b"]", SECTIONS

    @staticmethod
    def finish(res: dict, props, out: str) -> None:
        pass


def _source_hash(parts_dir: str) -> tuple[str, str | None]:
    """The download's own hash, from its manifest: Zenodo gives an md5, Wikimedia a sha1."""
    man = os.path.join(parts_dir, "manifest.json")
    m = json.load(open(man)) if os.path.exists(man) else {}
    return ("md5", m["md5"]) if m.get("md5") else ("sha1", m.get("sha1"))


def run(parts_dir: str, out: str, workers: int = 8, fmt=Fmt, stage: str | None = None, every: float | None = None) -> dict:
    """One pass; resumes from ``out``'s checkpoint when there is one (``dump_resume``), checkpointing every ``every``
    seconds. ``stage``: a local disk where each segment is written before it is copied whole to ``out``."""
    parts = part_list(parts_dir)
    algo, want = _source_hash(parts_dir)
    h = hashlib.new(algo)
    os.makedirs(out, exist_ok=True)
    names = [fmt.sample] + ([fmt.side] if fmt.side else [])
    ck_key = R.key(parts, names, want)
    st = R.load(out, ck_key) or {"offset": 0, "counts": {}, "props": {}, "reg": "00" * (1 << HLL_BITS), "tail": "",
                                  "s": 0.0, "segments": [], "resumed_from": []}
    tot, props, reg = collections.Counter(st["counts"]), collections.Counter(st["props"]), bytearray.fromhex(st["reg"])
    off, tail, before, resumed = st["offset"], bytes.fromhex(st["tail"]), st["s"], list(st["resumed_from"])
    every = R.EVERY if every is None else every
    segs = R.Segments(out, names, stage, st["segments"])
    src = stream(parts, workers, h)
    skip_s = 0.0
    if off:
        resumed.append(off)
        skip_s = R.skip(src, off)
    t0 = last = last_ck = time.time()
    bytes0 = tot["bytes"]
    with cf.ProcessPoolExecutor(workers) as ex:
        inflight = collections.deque()
        done = {"off": off, "tail": tail}

        def drain(one: bool) -> None:
            nonlocal last, last_ck
            while inflight and (len(inflight) >= 2 * workers or not one):
                fut, b_off, b_tail = inflight.popleft()
                c, p, r, sample, side = fut.result()
                tot.update(c)
                props.update(p)
                for i, v in enumerate(r):
                    if v > reg[i]:
                        reg[i] = v
                for line in sample:
                    segs.write(0, line)
                for line in side:
                    segs.write(1, line)
                done["off"], done["tail"] = b_off, b_tail
                now = time.time()
                if now - last > 300:
                    last = now
                    rate = (tot["bytes"] - bytes0) / max(now - t0, 1e-9) / 1e6
                    json.dump({"bytes": tot["bytes"], "entities": sum(v for k, v in tot.items() if k.startswith("entities:")),
                               "s": round(before + now - t0), "MB_s": round(rate, 1)},
                              open(os.path.join(out, "progress.json"), "w"))
                    print(f"  {tot['bytes'] / 1e9:.0f} GB, {tot['entities:item']:,} items, {rate:.0f} MB/s", flush=True)
                if one and now - last_ck >= every:              # never in the last drain: the last block's offset is
                    last_ck = now                               # one past the end
                    segs.close()
                    R.save(out, {"key": ck_key, "offset": done["off"], "counts": dict(tot), "props": dict(props),
                                 "reg": bytes(reg).hex(), "tail": done["tail"].hex(), "s": before + now - t0,
                                 "segments": segs.sizes, "resumed_from": resumed})
                if one:
                    break
        for b in blocks(src, fmt.sep):
            off += len(b) + 1                                   # the cut drops the separator's first byte
            tail = (tail + b)[-64:]
            inflight.append((ex.submit(fmt.work, b), off, tail))
            drain(True)
        drain(False)
    segs.join()
    k = tot["sample_entities"] or 1
    n = sum(v for key, v in tot.items() if key.startswith("entities:"))
    res = {"counts": dict(sorted(tot.items())), "references_distinct_est": hll_estimate(bytes(reg)),
           "bytes_by_section_est": {s: tot[f"sample_bytes:{s}"] * n // k for s in fmt.sections},
           "props_top": props.most_common(200), "props_n": len(props), "parts": len(parts),
           "hash_algo": algo, "hash_parts": h.hexdigest(), "hash_source": want,
           "hash_ok": (h.hexdigest() == want) if want else None, "ended_at_close": tail.rstrip().endswith(fmt.end),
           "wall_s": round(before + time.time() - t0), "resumed_from": resumed, "skip_s": round(skip_s)}
    fmt.finish(res, props, out)
    json.dump(res, open(os.path.join(out, "counts.json"), "w"), indent=1)
    R.finish(out)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--stage", default=None, help="a local disk: each segment is written there, then copied whole")
    ap.add_argument("--every", type=float, default=None, help="seconds between checkpoints (default 900)")
    a = ap.parse_args(argv)
    res = run(a.parts, a.out, a.workers, stage=a.stage, every=a.every)
    print(json.dumps({k: v for k, v in res.items() if k != "props_top"}, indent=1))


if __name__ == "__main__":
    main()
