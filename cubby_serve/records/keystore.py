"""cubby_serve/records/keystore.py -- the keys CheezML records are sealed with, read from the OS key store (order 0076).

Nick makes them (2026-11-19) with his own script, straight into Windows Credential Manager (generic credentials of
this user on this machine, encrypted by Windows with the user's logon):
- ``GRL/CheezML/current``: which key ids are in use (not secret): {"sign", "conversation", "provenance", "made"};
- ``GRL/CheezML/<role>/<kid>``: each key as a JWK (the Ed25519 signing key; the AES-256 keys of the two parts).

This module only reads, with ``CredReadW``. It writes, prints and logs no key, takes none from a file, an argument or
the environment, and keeps none past the `Keys` it hands back. On another system, or with no keys made, it raises.

Wiring: STANDALONE (validation/system_eval/seal_check.py ``--keys os``; nothing on the serve path yet).
"""
from __future__ import annotations

import json

from .envelope import Keys

__wiring__ = "STANDALONE"

PREFIX = "GRL/CheezML"
ROLES = ("sign", "conversation", "provenance")


def _read(target: str) -> bytes | None:
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
    adv.CredReadW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    adv.CredReadW.restype = w.BOOL
    adv.CredFree.argtypes = [ctypes.c_void_p]
    p = ctypes.POINTER(CREDENTIALW)()
    if not adv.CredReadW(target, 1, 0, ctypes.byref(p)):                  # 1: a generic credential
        if ctypes.get_last_error() == 1168:                                # ERROR_NOT_FOUND
            return None
        raise OSError(f"the key store refused a read (error {ctypes.get_last_error()})")
    try:
        return bytes(p.contents.CredentialBlob[:p.contents.CredentialBlobSize])
    finally:
        adv.CredFree(p)


def current(prefix: str = PREFIX) -> dict:
    """The key ids in use (not secret). ``prefix``: the test's own, never a real target."""
    blob = _read(f"{prefix}/current")
    if blob is None:
        raise LookupError("no CheezML keys in the key store: Nick makes them")
    return json.loads(blob)


def keys(kids: dict | None = None, prefix: str = PREFIX) -> Keys:
    """The keys in use (or those named by ``kids``, for an older record), as `envelope.Keys`."""
    from joserfc.jwk import OctKey, OKPKey
    kids = kids or current(prefix)
    got = {}
    for role in ROLES:
        blob = _read(f"{prefix}/{role}/{kids[role]}")
        if blob is None:
            raise LookupError(f"the {role} key {kids[role]} is not in the key store")
        got[role] = (OKPKey if role == "sign" else OctKey).import_key(json.loads(blob))
        del blob
    return Keys(got["sign"], kids["sign"], got["conversation"], kids["conversation"], got["provenance"], kids["provenance"])


__all__ = ["PREFIX", "ROLES", "current", "keys"]
