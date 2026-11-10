"""The dump count's worker pool (order 0055): its workers never hold lbzip2's input pipe open.

B2 stalled at the dump's last byte four times (2026-11-06 to 11-09): fork-started workers held a copy of the write end
of lbzip2's input pipe, so lbzip2 never saw the end of its input. The pipe test needs fork (Linux, Colab); on Windows
only the start-method check runs.
"""
import multiprocessing as mp
import select
import subprocess
import sys
import threading

import pytest

from validation.fact_store import dump_count as C


def test_the_pool_is_never_started_by_fork():
    ex = C.pool(1)
    try:
        assert ex._mp_context.get_start_method() != "fork"
    finally:
        ex.shutdown()


def _ends(ex, wait: float) -> bool:
    """A feeder writes into ``cat``'s stdin and closes it while the pool's workers run: does cat's output end?"""
    dec = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():
        for _ in range(20):
            dec.stdin.write(b"x" * 65536)
        dec.stdin.close()
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        ex.submit(abs, 1).result()                          # the workers start with the pipe open
        while True:
            if not select.select([dec.stdout], [], [], wait)[0]:
                return False
            if not dec.stdout.read1(1 << 16):
                return True
    finally:
        dec.kill()
        ex.shutdown()


@pytest.mark.skipif(sys.platform == "win32" or "fork" not in mp.get_all_start_methods(), reason="needs fork (Linux)")
def test_lbzip2s_input_ends_with_the_count_pool():
    assert _ends(C.pool(2), wait=10.0)


@pytest.mark.skipif(sys.platform == "win32" or "fork" not in mp.get_all_start_methods(), reason="needs fork (Linux)")
def test_a_forked_pool_keeps_the_pipe_open():
    """The cause, pinned: with fork the output never ends (what the pool above avoids)."""
    import concurrent.futures as cf
    assert not _ends(cf.ProcessPoolExecutor(2, mp_context=mp.get_context("fork")), wait=3.0)
