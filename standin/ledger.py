"""[stand-in] the certification ledger — every VM decision hashed, signed and stored in a database.

Wired: STANDALONE (the forge and the game's program library write to it; nothing in cubbyllm/ imports it).

Owner (2026-09-04): "are VM certifications signed?" — they were not: a certified program was a boolean verdict in
a JSON file the library trusted on load. "We should store the decision hash in a database" — this is that database.

A decision is one VM run judged against an expectation: the program text, the function run, the input, the
expectation, what the VM returned, the verdict, and the VM build that produced it. Its **hash** is the SHA-256 of the
canonical JSON of those fields; its **signature** is HMAC-SHA256 over the hash with the host key (a local secret,
`CB_LEDGER_KEY` or a key file generated once beside the database). Both go into a SQLite table with the fields
themselves. A program library entry keeps only the hash; on load the entry's program text must hash to a row that is
signed, certified and made by the current VM build — otherwise the entry loads RETIRED (never deleted) with the reason,
and re-enters only through a fresh certification. A new VM build therefore retires every certificate at once, and the
stored vectors are what re-certification replays.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import secrets
import sqlite3
import threading
import time

__wiring__ = "STANDALONE"

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "out" / "ledger.sqlite"
KEY_ENV = "CB_LEDGER_KEY"
FIELDS = ("program_sha256", "name", "kind", "fn", "input", "expected", "got", "ok", "vm_build")


def program_sha256(program: str) -> str:
    return hashlib.sha256(" ".join(str(program or "").split()).encode("utf-8")).hexdigest()


def vm_build_id(exe: str | os.PathLike | None = None) -> str:
    """The identity of the VM that certifies: the SHA-256 of cubelang.exe (the first 16 hex), 'no-vm' without one."""
    try:
        from cubbyllm.bridges import cubelang_client as cc
        path = pathlib.Path(exe) if exe else cc.find_cubelang_exe()
    except Exception:
        return "no-vm"
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "no-vm"


def decision_hash(rec: dict) -> str:
    """SHA-256 of the canonical JSON of the decision's fields (sorted keys, no whitespace)."""
    body = {k: rec.get(k) for k in FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class Ledger:
    """SQLite-backed, append-only; `record` returns the decision hash, `verify` checks a hash against the row."""

    def __init__(self, path: str | os.PathLike | None = None, key: bytes | None = None, vm_build: str | None = None) -> None:
        self.path = pathlib.Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key or self._load_key()
        self.vm_build = vm_build or vm_build_id()
        # serve_api is a ThreadingHTTPServer: the game's steps (and their certifications) run on request threads, while
        # the ledger is opened at startup on the main thread — one connection, same-thread check off, every access
        # under one lock (live_error "SQLite objects created in a thread can only be used in that same thread", 2026-09-04)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS certifications (
            hash TEXT PRIMARY KEY, program_sha256 TEXT NOT NULL, name TEXT, kind TEXT, fn TEXT,
            input TEXT, expected TEXT, got TEXT, ok INTEGER NOT NULL, vm_build TEXT NOT NULL,
            verdict TEXT, program TEXT NOT NULL, created_utc TEXT NOT NULL, signature TEXT NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_cert_program ON certifications(program_sha256)")
        self.db.commit()

    # ── the key ──────────────────────────────────────────────────────────────
    def _load_key(self) -> bytes:
        env = os.environ.get(KEY_ENV)
        if env:
            return env.encode("utf-8")
        import vault                                     # one host key protects the vault and signs the ledger, via derived subkeys
        return vault.subkey("ledger")

    def sign(self, h: str) -> str:
        return hmac.new(self.key, h.encode("ascii"), hashlib.sha256).hexdigest()

    # ── decisions ────────────────────────────────────────────────────────────
    def record(self, program: str, name: str, kind: str, fn: str | None, input: str | None, expected, got, ok: bool,
               verdict: str = "") -> str:
        """Hash, sign and store one decision; returns the decision hash (idempotent on the same decision)."""
        rec = {"program_sha256": program_sha256(program), "name": name, "kind": kind, "fn": fn, "input": input,
               "expected": None if expected is None else str(expected), "got": None if got is None else str(got),
               "ok": bool(ok), "vm_build": self.vm_build}
        h = decision_hash(rec)
        import vault                                     # the input may be a user's turn (a factory request): never in the clear
        with self._lock:
            self.db.execute("INSERT OR IGNORE INTO certifications VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (h, rec["program_sha256"], name, kind, fn, None if input is None else vault.encrypt_str(str(input)),
                             rec["expected"], rec["got"], int(rec["ok"]), rec["vm_build"],
                             verdict, program, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), self.sign(h)))
            self.db.commit()
        return h

    def get(self, h: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT hash, program_sha256, name, kind, fn, input, expected, got, ok, vm_build, verdict, program, created_utc, signature "
                                  "FROM certifications WHERE hash = ?", (h,)).fetchone()
        if row is None:
            return None
        keys = ("hash", "program_sha256", "name", "kind", "fn", "input", "expected", "got", "ok", "vm_build", "verdict", "program", "created_utc", "signature")
        d = dict(zip(keys, row))
        d["ok"] = bool(d["ok"])
        if d.get("input"):
            import vault
            try:
                d["input"] = vault.decrypt_str(d["input"])
            except (ValueError, TypeError):
                pass                                     # a pre-vault row
        return d

    def verify(self, h: str, program: str | None = None, require_current_vm: bool = True) -> tuple[bool, str]:
        """Is `h` a signed, certified decision — for THIS program text, by the current VM build? -> (ok, reason)."""
        d = self.get(h)
        if d is None:
            return False, "no such decision"
        if not hmac.compare_digest(d["signature"], self.sign(h)):
            return False, "bad signature"
        if decision_hash(d) != h:
            return False, "row does not hash to its key"
        if program is not None and program_sha256(program) != d["program_sha256"]:
            return False, "program text changed"
        if not d["ok"]:
            return False, "decision was a rejection"
        if require_current_vm and d["vm_build"] != self.vm_build:
            return False, f"certified by another VM build ({d['vm_build']} != {self.vm_build})"
        return True, "certified"

    def for_program(self, program: str) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT hash FROM certifications WHERE program_sha256 = ? ORDER BY created_utc", (program_sha256(program),)).fetchall()
        return [self.get(r[0]) for r in rows]

    def count(self, ok: bool | None = None) -> int:
        with self._lock:
            if ok is None:
                return self.db.execute("SELECT COUNT(*) FROM certifications").fetchone()[0]
            return self.db.execute("SELECT COUNT(*) FROM certifications WHERE ok = ?", (int(ok),)).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self.db.close()
