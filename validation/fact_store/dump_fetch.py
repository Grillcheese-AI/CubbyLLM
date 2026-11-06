"""validation/fact_store/dump_fetch.py -- order 0055 step 1: the Wikidata JSON dump, downloaded once, kept as parts.

The newest dated dump that has ``wikidata-<date>-all.json.bz2`` (Wikimedia's own sha1 beside it), fetched in parts of
``PART`` bytes with HTTP ranges, at most ``WORKERS`` at a time (Wikimedia asks for no more than two connections). A
part is written to ``part_NNNNN.part`` and renamed when whole, so a stopped run resumes at the first missing part; the
parts are the source copy (a parser bug means a reparse, not a re-download). When every part is in, the sha1 of their
concatenation is checked against Wikimedia's, and ``manifest.json`` records the url, date, size, sha1 and parts.

    python validation/fact_store/dump_fetch.py --out <dir on the Drive>   [--date 20261026]
    python validation/fact_store/dump_fetch.py --out <dir> --zenodo 22661413:wikikgqa-wikidata-all-dump.ttl.bz2

A Zenodo file (``--zenodo``) is checked against the record's md5 instead.

Wired: STANDALONE (order 0055's build; Colab, Nick's run).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request

__wiring__ = "STANDALONE"

BASE = "https://dumps.wikimedia.org/wikidatawiki/entities"
UA = "CubbyLLM-fact-store/0.1 (order 0055; research measurement)"
KIND = "all.json.bz2"
PART, WORKERS, BUF = 1 << 30, 2, 8 << 20


def _open(url: str, headers: dict | None = None, timeout: float = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})}),
                                  timeout=timeout)


def latest(date: str | None = None, opener=_open) -> dict:
    """{date, url, size, sha1} of the newest dated dump with the full JSON (or ``date``'s)."""
    idx = opener(f"{BASE}/").read().decode()
    dates = [date] if date else sorted(set(re.findall(r'href="(\d{8})/"', idx)), reverse=True)
    for d in dates:
        page = opener(f"{BASE}/{d}/").read().decode()
        name = f"wikidata-{d}-{KIND}"
        m = re.search(rf'href="{re.escape(name)}">[^<]+</a>\s+\S+ \S+\s+(\d+)', page)
        if not m:
            continue
        sums = opener(f"{BASE}/{d}/wikidata-{d}-sha1sums.txt").read().decode()
        sha = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(name)), None)
        return {"date": d, "name": name, "url": f"{BASE}/{d}/{name}", "size": int(m.group(1)), "sha1": sha}
    raise SystemExit(f"no {KIND} in {dates[:5]}")


def zenodo(record: str, key: str, opener=_open) -> dict:
    """{name, url, size, md5} of one file of a Zenodo record (0055's source dump is the WikiKGQA 2026 record's)."""
    r = json.loads(opener(f"https://zenodo.org/api/records/{record}").read().decode())
    f = next((f for f in r.get("files", []) if f["key"] == key), None)
    if not f:
        raise SystemExit(f"{key} is not in Zenodo record {record}")
    algo, _, digest = f["checksum"].partition(":")
    return {"date": r.get("metadata", {}).get("publication_date"), "name": key, "url": f["links"]["self"],
            "size": int(f["size"]), algo: digest, "zenodo": record,
            "licence": (r.get("metadata", {}).get("license") or {}).get("id")}


def part_path(out: str, i: int) -> str:
    return os.path.join(out, f"part_{i:05d}")


def spans(size: int, part: int = PART) -> list[tuple[int, int]]:
    return [(s, min(s + part, size) - 1) for s in range(0, size, part)]


def _landed(path: str, want: int, waits: int = 10, pause: float = 30.0) -> None:
    """The copy on the Drive shows its full size (the mount can lag behind a closed file), or the part is refetched."""
    for k in range(waits + 1):
        if os.path.exists(path) and os.path.getsize(path) == want:
            return
        if k < waits:
            time.sleep(pause)
    raise IOError(f"{os.path.basename(path)} not on the Drive at {want} bytes after the copy")


def fetch_part(url: str, out: str, i: int, span: tuple[int, int], opener=_open, tries: int = 6,
               stage: str | None = None) -> int:
    """One part, whole or not at all; returns its bytes (0 when already there). With ``stage`` (a local disk), the part
    is written and size-checked there, then copied whole to ``out``: the Drive's mount lost freshly written ``.part``
    files on 2026-11-06 (each loss cost a 1 GB refetch). A copy cut short leaves a short file, which a rerun refetches."""
    path, want = part_path(out, i), span[1] - span[0] + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return 0
    tmp = os.path.join(stage, f"part_{i:05d}.part") if stage else path + ".part"
    for k in range(tries):
        try:
            with opener(url, {"Range": f"bytes={span[0]}-{span[1]}"}) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(BUF)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(tmp) != want:
                raise IOError(f"part {i}: {os.path.getsize(tmp)} of {want} bytes")
            if stage:
                shutil.copyfile(tmp, path)
                _landed(path, want)
                os.remove(tmp)
            else:
                os.replace(tmp, path)
            return want
        except Exception as e:                                # noqa: BLE001 -- retried with a pause, then raised
            print(f"  part {i}: {e}; retry {k + 1}", flush=True)
            time.sleep(min(300, 15 * 2 ** k))
    raise SystemExit(f"part {i} failed {tries} times")


def verify(out: str, n: int, waits: int = 10, pause: float = 30.0, algo: str = "sha1") -> str:
    """The hash (``algo``) of the parts in order. The Drive's mount can show a part late after its rename (part 46 of 97 on
    2026-11-06): a missing part is waited for before the check gives up."""
    for k in range(waits + 1):
        missing = [i for i in range(n) if not os.path.exists(part_path(out, i))]
        if not missing:
            break
        if k == waits:
            raise SystemExit(f"parts missing after the download: {missing[:10]}; run again (it fetches only those)")
        print(f"  waiting for parts {missing[:5]} (the Drive's lag?)", flush=True)
        time.sleep(pause)
    h = hashlib.new(algo)
    for i in range(n):
        with open(part_path(out, i), "rb") as f:
            for b in iter(lambda: f.read(BUF), b""):
                h.update(b)
    return h.hexdigest()


def run(out: str, info: dict, workers: int = WORKERS, opener=_open, stage: str | None = None) -> dict:
    os.makedirs(out, exist_ok=True)
    if stage:
        os.makedirs(stage, exist_ok=True)
    sp = spans(info["size"])
    t0, got = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_part, info["url"], out, i, s, opener, stage=stage): i for i, s in enumerate(sp)}
        for k, fu in enumerate(cf.as_completed(futs)):
            got += fu.result()
            if (k + 1) % 5 == 0 or k + 1 == len(sp):
                dt = max(1e-6, time.time() - t0)
                print(f"  {k + 1}/{len(sp)} parts; {got / 1e9:.1f} GB this run, {got / dt / 1e6:.1f} MB/s", flush=True)
    algo = "md5" if info.get("md5") else "sha1"
    got = verify(out, len(sp), algo=algo)
    man = {**info, "parts": len(sp), "part_bytes": PART, "hash_algo": algo, "hash_parts": got,
           "hash_ok": got == info.get(algo), "fetched": time.strftime("%Y-%m-%d")}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    if not man["hash_ok"]:
        raise SystemExit(f"{algo} {got} is not the source's {info.get(algo)}: the parts are kept; check, then refetch")
    return man


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=None, help="a dated dump (default: the newest with the full JSON)")
    ap.add_argument("--zenodo", default=None, help="RECORD:FILE, a Zenodo file instead of Wikimedia's dump")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--stage", default=None, help="a local folder each part is written to before its copy to --out")
    a = ap.parse_args(argv)
    info = zenodo(*a.zenodo.split(":", 1)) if a.zenodo else latest(a.date)
    print(json.dumps(info), flush=True)
    print(json.dumps(run(a.out, info, a.workers, stage=a.stage), indent=1))


if __name__ == "__main__":
    main()
