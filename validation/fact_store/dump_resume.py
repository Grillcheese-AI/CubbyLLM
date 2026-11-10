"""validation/fact_store/dump_resume.py -- order 0055 step 1: the count resumes where a dropped runtime left it.

The challenge dump's first count (2026-11-06) stopped at 961 GB of about 1 TB after 3 h 20 min, and nothing of it
was kept: the 1% sample, written into one file left open for the whole pass, never reached the Drive (Colab's mount
uploads a file once it is closed). So ``dump_count.run`` now checkpoints:

- **The sample and side files in segments** (``<name>.seg00000``, ...): each segment is a gzip member closed at a
  checkpoint, written on the local ``stage`` when there is one, copied whole to ``out`` and its size checked there.
- **``resume.json``**, written after the segments it names have landed: the decompressed bytes counted so far (always
  at a cut between blocks, so whole entities), the counters, the references' HLL registers, the dump's last bytes,
  the time spent, and each segment's sizes.
- **On a rerun with the same parts**, the stream is read again from the start (the parts' hash needs every compressed
  byte), decompressed without parsing up to the checkpoint's offset, and the count goes on from there.
- **At the end** the segments are joined into the one file the order names (gzip members concatenate), the segments
  removed, and the checkpoint renamed ``resume.done.json``.

A checkpoint from other parts, or one whose segments are not on the Drive, is set aside (``resume.stale.json``) and
the pass starts over.

Wired: STANDALONE (imported by dump_count.py).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import time

try:
    from . import dump_stream as S
except ImportError:                                           # run as a script from its folder (Colab)
    import dump_stream as S

__wiring__ = "STANDALONE"

EVERY = 900.0                                                 # seconds between checkpoints: at most 15 min lost
CKPT, DONE, STALE = "resume.json", "resume.done.json", "resume.stale.json"


def seg_path(out: str, name: str, k: int) -> str:
    return os.path.join(out, f"{name}.seg{k:05d}")


def _land(tmp: str, path: str, staged: bool) -> int:
    """``tmp`` moved or copied to ``path``; its size, checked on the Drive (the mount can lag behind a closed file)."""
    want = os.path.getsize(tmp)
    if not staged:
        os.replace(tmp, path)
        return want
    shutil.copyfile(tmp, path)
    for k in range(S.WAITS + 1):
        if os.path.exists(path) and os.path.getsize(path) == want:
            os.remove(tmp)
            return want
        if k < S.WAITS:
            time.sleep(S.PAUSE)
    raise SystemExit(f"{os.path.basename(path)} not on the Drive at {want} bytes after the copy")


class Segments:
    """The sample and side files, written a segment at a time; ``names`` holds the files' final names."""

    def __init__(self, out: str, names: list[str], stage: str | None, sizes: list[list[int]]):
        self.out, self.names, self.stage, self.sizes = out, names, stage, [list(s) for s in sizes]
        if stage:
            os.makedirs(stage, exist_ok=True)
        self._open()

    def _tmp(self, name: str) -> str:
        k = len(self.sizes)
        return os.path.join(self.stage, f"{name}.seg{k:05d}") if self.stage else seg_path(self.out, name, k) + ".part"

    def _open(self) -> None:
        self.files = [gzip.open(self._tmp(n), "wb", compresslevel=6) for n in self.names]

    def write(self, i: int, line: bytes) -> None:
        self.files[i].write(line + b"\n")

    def close(self, reopen: bool = True) -> None:
        """Closes this segment and lands it; the next one opens (unless it is the last)."""
        for f in self.files:
            f.close()
        k = len(self.sizes)
        self.sizes.append([_land(self._tmp(n), seg_path(self.out, n, k), bool(self.stage)) for n in self.names])
        if reopen:
            self._open()

    def join(self) -> None:
        """Every segment landed into its final file, whole, then the segments removed."""
        self.close(reopen=False)
        for i, name in enumerate(self.names):
            tmp = os.path.join(self.stage or self.out, name + ".part")
            with open(tmp, "wb") as w:
                for k in range(len(self.sizes)):
                    with open(seg_path(self.out, name, k), "rb") as r:
                        shutil.copyfileobj(r, w, 8 << 20)
            if os.path.getsize(tmp) != sum(s[i] for s in self.sizes):
                raise SystemExit(f"{name}: the joined segments do not add up")
            _land(tmp, os.path.join(self.out, name), bool(self.stage))
            for p in glob.glob(os.path.join(self.out, f"{glob.escape(name)}.seg[0-9][0-9][0-9][0-9][0-9]")):
                os.remove(p)


def key(parts: list[str], names: list[str], want: str | None) -> dict:
    """What a checkpoint must match to be resumed: the same parts, the same output files, the same source hash."""
    return {"parts": len(parts), "parts_bytes": sum(os.path.getsize(p) for p in parts), "names": names,
            "hash_source": want}


def save(out: str, state: dict) -> None:
    tmp = os.path.join(out, CKPT + ".part")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join(out, CKPT))


def load(out: str, want_key: dict) -> dict | None:
    """The checkpoint to resume from, or None (none there; or set aside: other parts, or a segment not on the Drive)."""
    path = os.path.join(out, CKPT)
    if not os.path.exists(path):
        return None
    st = json.load(open(path))
    why = None
    if st.get("key") != want_key:
        why = "it was written for other parts"
    else:
        for k in range(S.WAITS + 1):
            gone = [(n, j) for j, sz in enumerate(st["segments"]) for i, n in enumerate(want_key["names"])
                    if not (os.path.exists(seg_path(out, n, j)) and os.path.getsize(seg_path(out, n, j)) == sz[i])]
            if not gone:
                break
            if k < S.WAITS:
                print(f"  waiting for segments {gone[:3]} (the Drive's lag?)", flush=True)
                time.sleep(S.PAUSE)
        else:
            why = f"segment {gone[0][1]} of {gone[0][0]} is not on the Drive at its size"
    if why:
        os.replace(path, os.path.join(out, STALE))
        print(f"  checkpoint set aside ({STALE}): {why}; the pass starts over", flush=True)
        return None
    print(f"  resuming at {st['offset'] / 1e9:.0f} GB ({sum(v for k, v in st['counts'].items() if k.startswith('entities:')):,}"
          f" entities counted, {len(st['segments'])} segments)", flush=True)
    return st


def skip(f, n: int, every: float = 300.0, live: dict | None = None) -> float:
    """Reads and drops the stream's first ``n`` decompressed bytes (never past them); the seconds it took."""
    t0 = last = time.time()
    done = 0
    while done < n:
        b = f.read(min(S.BLOCK, n - done))
        if not b:
            raise SystemExit(f"the stream ended at {done:,} bytes, before the checkpoint's {n:,}")
        done += len(b)
        if live is not None:
            live["at"] = done
        if time.time() - last > every:
            last = time.time()
            print(f"  skipping: {done / 1e9:.0f} of {n / 1e9:.0f} GB, {done / (last - t0) / 1e6:.0f} MB/s", flush=True)
    return time.time() - t0


def finish(out: str) -> None:
    path = os.path.join(out, CKPT)
    if os.path.exists(path):
        os.replace(path, os.path.join(out, DONE))
