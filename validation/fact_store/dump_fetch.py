"""validation/fact_store/dump_fetch.py -- order 0055 step 1: the Wikidata JSON dump, downloaded once, kept as parts.

The newest dated dump that has ``wikidata-<date>-all.json.bz2`` (Wikimedia's own sha1 beside it), fetched in parts of
``PART`` bytes with HTTP ranges, at most ``WORKERS`` at a time (Wikimedia asks for no more than two connections). A
part is written to ``part_NNNNN.part`` and renamed when whole, so a stopped run resumes at the first missing part; the
parts are the source copy (a parser bug means a reparse, not a re-download). When every part is in, the sha1 of their
concatenation is checked against Wikimedia's, and ``manifest.json`` records the url, date, size, sha1 and parts.

    python validation/fact_store/dump_fetch.py --out <dir on the Drive>   [--date 20261026]

Wired: STANDALONE (order 0055's build; Colab, Nick's run).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
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


def part_path(out: str, i: int) -> str:
    return os.path.join(out, f"part_{i:05d}")


def spans(size: int, part: int = PART) -> list[tuple[int, int]]:
    return [(s, min(s + part, size) - 1) for s in range(0, size, part)]


def fetch_part(url: str, out: str, i: int, span: tuple[int, int], opener=_open, tries: int = 6) -> int:
    """One part, whole or not at all; returns its bytes (0 when already there)."""
    path, want = part_path(out, i), span[1] - span[0] + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return 0
    for k in range(tries):
        try:
            with opener(url, {"Range": f"bytes={span[0]}-{span[1]}"}) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(BUF)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(path + ".part") != want:
                raise IOError(f"part {i}: {os.path.getsize(path + '.part')} of {want} bytes")
            os.replace(path + ".part", path)
            return want
        except Exception as e:                                # noqa: BLE001 -- retried with a pause, then raised
            print(f"  part {i}: {e}; retry {k + 1}", flush=True)
            time.sleep(min(300, 15 * 2 ** k))
    raise SystemExit(f"part {i} failed {tries} times")


def verify(out: str, n: int, waits: int = 10, pause: float = 30.0) -> str:
    """The sha1 of the parts in order. The Drive's mount can show a part late after its rename (part 46 of 97 on
    2026-11-06): a missing part is waited for before the check gives up."""
    for k in range(waits + 1):
        missing = [i for i in range(n) if not os.path.exists(part_path(out, i))]
        if not missing:
            break
        if k == waits:
            raise SystemExit(f"parts missing after the download: {missing[:10]}; run again (it fetches only those)")
        print(f"  waiting for parts {missing[:5]} (the Drive's lag?)", flush=True)
        time.sleep(pause)
    h = hashlib.sha1()
    for i in range(n):
        with open(part_path(out, i), "rb") as f:
            for b in iter(lambda: f.read(BUF), b""):
                h.update(b)
    return h.hexdigest()


def run(out: str, info: dict, workers: int = WORKERS, opener=_open) -> dict:
    os.makedirs(out, exist_ok=True)
    sp = spans(info["size"])
    t0, got = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_part, info["url"], out, i, s, opener): i for i, s in enumerate(sp)}
        for k, fu in enumerate(cf.as_completed(futs)):
            got += fu.result()
            if (k + 1) % 5 == 0 or k + 1 == len(sp):
                dt = max(1e-6, time.time() - t0)
                print(f"  {k + 1}/{len(sp)} parts; {got / 1e9:.1f} GB this run, {got / dt / 1e6:.1f} MB/s", flush=True)
    sha = verify(out, len(sp))
    man = {**info, "parts": len(sp), "part_bytes": PART, "sha1_parts": sha, "sha1_ok": sha == info.get("sha1"),
           "fetched": time.strftime("%Y-%m-%d")}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    if not man["sha1_ok"]:
        raise SystemExit(f"sha1 {sha} is not Wikimedia's {info.get('sha1')}: the parts are kept; check, then refetch")
    return man


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=None, help="a dated dump (default: the newest with the full JSON)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args(argv)
    info = latest(a.date)
    print(json.dumps(info), flush=True)
    print(json.dumps(run(a.out, info, a.workers), indent=1))


if __name__ == "__main__":
    main()
