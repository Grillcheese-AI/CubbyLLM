"""Pins for the vault (user chats never stored unencrypted) and the certification ledger (every VM decision hashed,
signed, stored; a library entry loads RETIRED when its certificate does not hold). No model, no VM: the VM build id is
faked. Run: python -m pytest standin/tests -q"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "standin"), str(ROOT / "standin" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

import vault  # noqa: E402
from ledger import Ledger, decision_hash, program_sha256  # noqa: E402


def test_vault_round_trips_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    msg = "tu porte quoi ce soir ? — a user turn, never in the clear".encode("utf-8")
    blob = vault.encrypt_bytes(msg)
    assert blob.startswith(vault.MAGIC) and msg not in blob and vault.decrypt_bytes(blob) == msg
    assert vault.encrypt_bytes(msg) != blob, "a fresh nonce per message"
    bad = bytearray(blob); bad[len(vault.MAGIC) + vault.NONCE + 3] ^= 1
    with pytest.raises(ValueError):
        vault.decrypt_bytes(bytes(bad))
    monkeypatch.setenv(vault.KEY_ENV, "another-key")
    with pytest.raises(ValueError):
        vault.decrypt_bytes(blob)
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    f = tmp_path / "turns.jsonl.enc"
    vault.write_lines(f, [json.dumps({"text": "hi there cubby"}), json.dumps({"text": "salut"})])
    assert b"cubby" not in f.read_bytes() and [json.loads(l)["text"] for l in vault.read_lines(f)] == ["hi there cubby", "salut"]
    assert vault.decrypt_str(vault.encrypt_str("x")) == "x" and vault.subkey("vault") != vault.subkey("ledger")


def test_ledger_records_signs_verifies_and_retires_tampered_entries(tmp_path, monkeypatch):
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    led = Ledger(tmp_path / "ledger.sqlite", vm_build="vm-abc")
    prog = "program Dec0 implements ISolver { }"
    h = led.record(prog, "FLEE#1", "decision", "solve", "A sensor reads 2 units", "flag (>= 50)", "90", True, "certified")
    assert h == decision_hash({"program_sha256": program_sha256(prog), "name": "FLEE#1", "kind": "decision", "fn": "solve",
                               "input": "A sensor reads 2 units", "expected": "flag (>= 50)", "got": "90", "ok": True, "vm_build": "vm-abc"})
    assert led.verify(h, prog) == (True, "certified") and led.count() == 1 and led.count(ok=True) == 1
    assert led.record(prog, "FLEE#1", "decision", "solve", "A sensor reads 2 units", "flag (>= 50)", "90", True) == h and led.count() == 1, "idempotent"
    row = led.get(h)
    assert row["input"] == "A sensor reads 2 units", "the input reads back in the clear through the vault"
    raw = led.db.execute("SELECT input FROM certifications WHERE hash = ?", (h,)).fetchone()[0]
    assert "sensor" not in raw, "and is stored encrypted"
    assert led.verify(h, prog + " // edited")[1] == "program text changed"
    assert led.verify("0" * 64)[1] == "no such decision"
    rej = led.record(prog + "2", "FLEE#2", "decision", "solve", "x", "flag", "30", False, "REJECTED")
    assert led.verify(rej, prog + "2")[1] == "decision was a rejection"
    led.db.execute("UPDATE certifications SET signature = ? WHERE hash = ?", ("00" * 32, h)); led.db.commit()
    assert led.verify(h, prog)[1] == "bad signature"
    other = Ledger(tmp_path / "ledger.sqlite", vm_build="vm-new")
    led2 = Ledger(tmp_path / "ledger2.sqlite", vm_build="vm-abc"); h2 = led2.record(prog, "X", "decision", "solve", None, "a", "a", True)
    assert Ledger(tmp_path / "ledger2.sqlite", vm_build="vm-new").verify(h2, prog)[1].startswith("certified by another VM build")
    # the program library: an entry whose program was edited loads RETIRED, never deleted; a pre-ledger entry is flagged
    from pacman import ProgramLibrary
    lib = ProgramLibrary(tmp_path / "programs.json", ledger=led2)
    lib.add("FLEE#1", None, prog, "tool", 0, {"why": "test"}, cert=h2)
    lib.add("OLD", None, "program Old implements ISolve { }", "tool", 0, {"why": "pre-ledger"})
    data = json.loads((tmp_path / "programs.json").read_text(encoding="utf-8"))
    data["entries"]["FLEE#1"]["program"] = prog + " // tampered on disk"
    (tmp_path / "programs.json").write_text(json.dumps(data), encoding="utf-8")
    again = ProgramLibrary(tmp_path / "programs.json", ledger=led2)
    assert again.entries["FLEE#1"]["retired"] and again.entries["FLEE#1"]["retired_reason"] == "certificate: program text changed"
    assert "FLEE#1" in again.entries and again.entries["OLD"].get("uncertified") and not again.entries["OLD"].get("retired")



def test_ledger_records_from_request_threads(tmp_path, monkeypatch):
    """serve_api is threaded: certifications arrive from request threads while the ledger was opened on the main
    thread (live_error 2026-09-04). One shared connection under a lock takes them all."""
    import threading
    monkeypatch.setenv(vault.KEY_ENV, "test-host-key")
    led = Ledger(tmp_path / "ledger.sqlite", vm_build="vm-abc")
    errors, hashes = [], []

    def worker(k):
        try:
            hashes.append(led.record(f"program P{k} implements ISolve {{ }}", f"J#{k}", "join", "solve", None, "a", "a", True))
        except Exception as e:                           # noqa: BLE001 - the failure under test
            errors.append(repr(e))
    ts = [threading.Thread(target=worker, args=(k,)) for k in range(12)]
    for th in ts: th.start()
    for th in ts: th.join()
    assert errors == [] and len(set(hashes)) == 12 and led.count(ok=True) == 12
    assert all(led.verify(h)[0] for h in hashes)
