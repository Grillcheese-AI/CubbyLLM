"""validation/fact_store/dump_stream.py -- order 0055 step 1: the dump's parts read back as one stream, for
``dump_count.py``.

``part_list``: the parts, none missing (a gap the Drive's mount shows late is waited out). ``stream``: the decompressed
dump (``lbzip2`` on every core, Python's ``bz2`` for tests), hashing the compressed bytes on the way; a failed lbzip2
stops the pass instead of passing for the dump's end. ``blocks``: whole lines, in blocks of ``BLOCK`` bytes (cut at a fallback where the separator is long absent).

Wired: STANDALONE (imported by dump_count.py).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time

__wiring__ = "STANDALONE"

BLOCK = 32 << 20
FALLBACK_BLOCKS, MAX_BLOCKS = 4, 64                           # no separator in 128 MB: the fallback cut; no cut in 2 GB: stop
WAITS, PAUSE = 10, 30.0                                       # the Drive's lag: up to 5 minutes


class _Checked:
    """lbzip2's output; at its end, lbzip2 and the feeder must have finished clean, or the pass stops (a truncated or
    corrupt part must not pass for the end of the dump)."""

    def __init__(self, dec, feeder, err):
        self.dec, self.feeder, self.err = dec, feeder, err

    def read(self, n):
        b = self.dec.stdout.read(n)
        if not b:
            self.feeder.join()
            if self.err:
                raise SystemExit(f"reading the parts failed: {self.err[0]!r}")
            if self.dec.wait() != 0:
                raise SystemExit(f"lbzip2 exited {self.dec.returncode}: a part is corrupt or missing")
        return b


def stream(parts: list[str], workers: int, h=None):
    """The decompressed dump, as a binary file; ``h`` (a hashlib object) takes the compressed bytes on the way, so the
    pass also checks the parts against Wikimedia's sha1."""
    def chunks():
        for p in parts:
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(8 << 20), b""):
                    if h is not None:
                        h.update(b)
                    yield b
    if shutil.which("lbzip2"):
        import threading
        dec = subprocess.Popen(["lbzip2", "-dc", "-n", str(workers)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               bufsize=BLOCK)
        err = []

        def feed():
            try:
                for b in chunks():
                    dec.stdin.write(b)
            except Exception as e:                            # noqa: BLE001 -- handed to the reader, which stops
                err.append(e)
            finally:
                try:
                    dec.stdin.close()
                except OSError:
                    pass
        t = threading.Thread(target=feed, daemon=True)
        t.start()
        return _Checked(dec, t, err)
    import bz2
    import io

    class Cat(io.RawIOBase):
        def __init__(self):
            self.it, self.rest = chunks(), b""

        def readable(self):
            return True

        def readinto(self, b):
            if not self.rest:
                self.rest = next(self.it, b"")
            n = min(len(b), len(self.rest))
            b[:n], self.rest = self.rest[:n], self.rest[n:]
            return n
    return bz2.BZ2File(io.BufferedReader(Cat()))                # raises EOFError on a truncated stream


def part_list(parts_dir: str, waits: int | None = None, pause: float | None = None) -> list[str]:
    """part_00000 .. part_<n-1>, none missing (``manifest.json``'s count when the download wrote one). The Drive's
    mount can show a part late after its rename: a gap is waited out, then stops the pass."""
    waits, pause = WAITS if waits is None else waits, PAUSE if pause is None else pause
    want = None
    man = os.path.join(parts_dir, "manifest.json")
    if os.path.exists(man):
        want = json.load(open(man)).get("parts")
    for k in range(waits + 1):
        got = sorted(glob.glob(os.path.join(parts_dir, "part_[0-9][0-9][0-9][0-9][0-9]")))
        n = want or (int(got[-1][-5:]) + 1 if got else 0)
        missing = [i for i in range(n) if os.path.join(parts_dir, f"part_{i:05d}") not in got]
        if got and not missing:
            return got[:n]
        if k < waits:
            print(f"  waiting for parts {missing[:5]} (the Drive's lag?)", flush=True)
            time.sleep(pause)
    raise SystemExit(f"parts missing: {missing[:10]}")


def blocks(f, sep: bytes = b"\n", fallback: bytes | None = None, start: int = 0, note=None):
    """Blocks cut just before the last ``sep`` in each read (a newline: whole lines; ``\\ndata:``: whole Turtle
    entities, the next block starting at ``data:``).

    Where ``sep`` has not appeared in ``FALLBACK_BLOCKS`` reads, the block is cut at the last ``fallback`` instead
    (Turtle: a blank line, the end of a subject's triples): a guard, since with no cut the buffer would grow without
    end, each read copying all of it (B2's stall was not this but the pool's pipe; ``dump_count.pool``). ``MAX_BLOCKS`` reads with no cut at all
    stop the pass, never hang it. ``note(kind, at, data)`` hears of each fallback cut ("fallback", the buffer's first
    MB), of a stop ("stopped", the same) and of every 5 minutes without a cut ("waiting", the buffer's size); ``at`` is
    the buffer's offset in the decompressed dump (``start``: where this read began)."""
    rest, pos, last = b"", start, time.time()
    while True:
        b = f.read(BLOCK)
        if not b:
            if rest:
                yield rest
            return
        pos += len(b)
        b = rest + b
        cut = b.rfind(sep)
        if cut < 0 and fallback and len(b) >= FALLBACK_BLOCKS * BLOCK:
            cut = b.rfind(fallback)
            if cut >= 0 and note:
                note("fallback", pos - len(b), b[:1 << 20])
        if cut < 0:
            if len(b) >= MAX_BLOCKS * BLOCK:
                if note:
                    note("stopped", pos - len(b), b[:1 << 20])
                raise SystemExit(f"no cut in {len(b) >> 20} MB from byte {pos - len(b):,} of the dump (its first MB "
                                 "is kept beside the counts)")
            rest = b
            if note and time.time() - last > 300:
                last = time.time()
                note("waiting", pos - len(b), len(b))
            continue
        rest, last = b[cut + 1:], time.time()
        yield b[:cut]
