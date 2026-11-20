"""Order 0076 (cubby_serve/records/keystore.py), Cowork's review of 2026-11-19: the loader reads a key set from the OS
key store into `Keys` that seal and verify. The test writes its own credentials under a TEST prefix (keys made in
memory for it), reads them through the loader, seals a record and verifies it, then removes every credential it
wrote. No real target (``GRL/CheezML/...``) is read or touched. Skipped off Windows."""
from __future__ import annotations

import json
import sys
import uuid

import pytest

if sys.platform != "win32":
    pytest.skip("the OS key store read here is Windows Credential Manager", allow_module_level=True)
pytest.importorskip("joserfc")
pytest.importorskip("pyld")
from joserfc.jwk import OctKey, OKPKey  # noqa: E402

from cubby_serve.records import keystore as K  # noqa: E402
from cubby_serve.records.envelope import seal  # noqa: E402

TEST = f"GRL-TEST/CheezML-keystore-{uuid.uuid4().hex[:8]}"


def _cred():
    import ctypes
    from ctypes import wintypes as w
    adv = ctypes.WinDLL("advapi32", use_last_error=True)

    class FILETIME(ctypes.Structure):
        _fields_ = [("lo", w.DWORD), ("hi", w.DWORD)]

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [("Flags", w.DWORD), ("Type", w.DWORD), ("TargetName", w.LPWSTR), ("Comment", w.LPWSTR),
                    ("LastWritten", FILETIME), ("CredentialBlobSize", w.DWORD),
                    ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", w.DWORD),
                    ("AttributeCount", w.DWORD), ("Attributes", ctypes.c_void_p), ("TargetAlias", w.LPWSTR),
                    ("UserName", w.LPWSTR)]
    adv.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), w.DWORD]
    adv.CredDeleteW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD]

    def write(target: str, blob: bytes) -> None:
        assert target.startswith("GRL-TEST/")
        buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        c = CREDENTIALW()
        c.Type, c.TargetName, c.UserName, c.Persist = 1, target, "test", 2       # generic, this logon session up
        c.CredentialBlobSize, c.CredentialBlob = len(blob), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
        assert adv.CredWriteW(ctypes.byref(c), 0), ctypes.get_last_error()

    def delete(target: str) -> None:
        assert target.startswith("GRL-TEST/")
        adv.CredDeleteW(target, 1, 0)
    return write, delete


@pytest.fixture
def stored():
    write, delete = _cred()
    kids = {r: f"test-{r}-{uuid.uuid4().hex[:6]}" for r in K.ROLES}
    jwk = {"sign": OKPKey.generate_key("Ed25519").as_dict(private=True),
           "conversation": OctKey.generate_key(256).as_dict(private=True),
           "provenance": OctKey.generate_key(256).as_dict(private=True)}
    targets = [f"{TEST}/{r}/{kids[r]}" for r in K.ROLES] + [f"{TEST}/current"]
    try:
        for r in K.ROLES:
            write(f"{TEST}/{r}/{kids[r]}", json.dumps(jwk[r]).encode())
        write(f"{TEST}/current", json.dumps(kids).encode())
        yield kids, jwk
    finally:
        for t in targets:
            delete(t)
        assert all(K._read(t) is None for t in targets)                       # removed by the test


def test_the_loader_reads_a_key_set_that_seals_and_verifies(stored):
    kids, jwk = stored
    assert K.current(TEST) == kids
    keys = K.keys(prefix=TEST)
    assert (keys.sign_id, keys.conversation_id, keys.provenance_id) == (kids["sign"], kids["conversation"], kids["provenance"])
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "validation"))
    import verify_cheez as V
    from test_cheez_writer import REC, RUN
    from cubbyllm.records import cheez_writer as W
    sealed = seal(W.write(REC, RUN)[0], keys)
    pub = OKPKey.import_key({k: v for k, v in jwk["sign"].items() if k != "d"})
    assert V.signed(sealed, pub) == "signed"
    assert V.read(sealed, pub, keys.conversation, keys.provenance)["says"] == "same"


def test_a_missing_key_set_is_an_error_not_an_empty_key():
    with pytest.raises(LookupError):
        K.current(f"{TEST}-absent")
