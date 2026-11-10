"""cubby_serve/board/store_client.py -- the hot Wikidata store (order 0055's `wdstore`), asked over a pipe.

`StoreClient` starts `wdstore serve <store dir>` once, on first use, and sends it one JSON request a line on its
stdin, reading one JSON answer a line from its stdout (a pipe to a child process: no socket, no network). One
request at a time (a lock). A child that has died is started again once; a second failure is an error answer.

Where things are, never in a committed file: `CUBBY_WD_STORE` (the store's folder) and `CUBBY_WDSTORE_EXE` (the
binary), from the environment or `paths.local.json` (`cubbyllm.local_paths`). With either unset, `from_env()` gives
None and the board's Knowledge section says the store is not configured.

Wired: WIRED (`cubby_serve/board/store.py`, the board's Knowledge section).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading

from cubbyllm.local_paths import value

__wiring__ = "WIRED"


class StoreClient:
    def __init__(self, exe: str, store: str) -> None:
        self.exe, self.store = exe, store
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen([self.exe, "serve", self.store], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, creationflags=flags)

    def _once(self, line: str) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline()
        if not out:
            raise OSError("the store closed its pipe")
        return json.loads(out)

    def ask(self, request: dict) -> dict:
        """One request (`{"op": ...}`, as `wdstore serve` takes them) -> its answer; `{"error": ...}` on a failure."""
        line = json.dumps(request, ensure_ascii=False)
        with self._lock:
            for attempt in (1, 2):
                try:
                    return self._once(line)
                except (OSError, ValueError) as e:
                    self.close()
                    if attempt == 2:
                        return {"error": f"the store did not answer: {e}"}
        return {"error": "unreachable"}

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()


_client: StoreClient | None = None
_client_lock = threading.Lock()


def from_env() -> StoreClient | None:
    """The one client for this server, when the store and its binary are named and exist; else None."""
    global _client
    store, exe = value("CUBBY_WD_STORE"), value("CUBBY_WDSTORE_EXE")
    if not store or not exe or not os.path.isdir(store) or not os.path.isfile(exe):
        return None
    with _client_lock:
        if _client is None or (_client.store, _client.exe) != (store, exe):
            _client = StoreClient(exe, store)
        return _client
