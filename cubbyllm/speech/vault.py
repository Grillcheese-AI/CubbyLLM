"""cubbyllm/speech/vault.py -- the vault: user chats are never stored unencrypted (owner, 2026-09-04; moved from
standin/vault.py in order 0031: keeping a user's chat encrypted is Cubby's rule).

Wired: WIRED (the server's ledger and harness wiring; the context graph's saves; the builders and probes).

Stdlib only (no `cryptography` on this machine): authenticated encryption built from HMAC-SHA256 —
  keystream  = HMAC(k_enc, nonce || counter) blocks, XORed over the plaintext (a CTR-style stream cipher over a
               16-byte random nonce; a keystream block is never reused because the nonce is fresh per message)
  tag        = HMAC(k_mac, nonce || ciphertext)                      (encrypt-then-MAC; constant-time compare)
  message    = MAGIC || nonce || ciphertext || tag
The host key lives where the server names it (`use_key_file`; the serve stack's is `standin/data/out/host.key`,
generated once, 64 hex chars, `cubby_serve.keys`) or comes from `CB_HOST_KEY`; the vault and the
ledger use keys derived from it (`subkey("vault")`, `subkey("ledger")`), so one file protects both and neither can be
used for the other. What goes through the vault: the real-user-turn eval file, the talk probe's rows, the ledger's
`input` column (a factory request is a user turn), and any chat log the serve stack ever persists. The training
sets are the owner's decision and are not covered here (they are plaintext by necessity where they train).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import secrets

__wiring__ = "WIRED"

KEY_FILE: pathlib.Path | None = None   # the server names it (use_key_file); the package names no place (order 0031)
KEY_ENV = "CB_HOST_KEY"
MAGIC = b"CBV1"
NONCE = 16
TAG = 32
BLOCK = 32


def host_key(path: pathlib.Path | None = None) -> bytes:
    """The host key: the env var, else the key file the server named (created once with 256 bits of entropy).
    Neither: a RuntimeError, never a file in a place of the package's own choosing."""
    env = os.environ.get(KEY_ENV)
    if env:
        return hashlib.sha256(env.encode("utf-8")).digest()
    kf = path or KEY_FILE
    if kf is None:
        raise RuntimeError(f"no host key: the server names its file (vault.use_key_file) or {KEY_ENV} is set")
    if kf.exists():
        return bytes.fromhex(kf.read_text(encoding="ascii").strip())
    kf.parent.mkdir(parents=True, exist_ok=True)
    k = secrets.token_bytes(32)
    kf.write_text(k.hex(), encoding="ascii")
    try:
        os.chmod(kf, 0o600)
    except OSError:
        pass
    return k


def use_key_file(path) -> None:
    """Where the host key lives (the server's choice: ``cubby_serve.keys``)."""
    global KEY_FILE
    KEY_FILE = pathlib.Path(path)


def subkey(purpose: str, key: bytes | None = None) -> bytes:
    return hmac.new(key or host_key(), b"cubby:" + purpose.encode("utf-8"), hashlib.sha256).digest()


def _keystream(k_enc: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hmac.new(k_enc, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:n])


def encrypt_bytes(data: bytes, key: bytes | None = None) -> bytes:
    k = key or subkey("vault")
    k_enc, k_mac = subkey("enc", k), subkey("mac", k)
    nonce = secrets.token_bytes(NONCE)
    ct = bytes(a ^ b for a, b in zip(data, _keystream(k_enc, nonce, len(data))))
    tag = hmac.new(k_mac, nonce + ct, hashlib.sha256).digest()
    return MAGIC + nonce + ct + tag


def decrypt_bytes(blob: bytes, key: bytes | None = None) -> bytes:
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + NONCE + TAG:
        raise ValueError("not a vault message")
    k = key or subkey("vault")
    k_enc, k_mac = subkey("enc", k), subkey("mac", k)
    nonce, ct, tag = blob[len(MAGIC):len(MAGIC) + NONCE], blob[len(MAGIC) + NONCE:-TAG], blob[-TAG:]
    if not hmac.compare_digest(tag, hmac.new(k_mac, nonce + ct, hashlib.sha256).digest()):
        raise ValueError("vault tag mismatch: wrong key or tampered")
    return bytes(a ^ b for a, b in zip(ct, _keystream(k_enc, nonce, len(ct))))


def encrypt_str(text: str, key: bytes | None = None) -> str:
    """A string field for a database or JSON: base64-free hex of the vault message."""
    return encrypt_bytes(text.encode("utf-8"), key).hex()


def decrypt_str(hexblob: str, key: bytes | None = None) -> str:
    return decrypt_bytes(bytes.fromhex(hexblob), key).decode("utf-8")


def write_json(path: str | os.PathLike, obj, key: bytes | None = None) -> None:
    pathlib.Path(path).write_bytes(encrypt_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), key))


def read_json(path: str | os.PathLike, key: bytes | None = None):
    return json.loads(decrypt_bytes(pathlib.Path(path).read_bytes(), key).decode("utf-8"))


def write_lines(path: str | os.PathLike, lines: list[str], key: bytes | None = None) -> None:
    pathlib.Path(path).write_bytes(encrypt_bytes("\n".join(lines).encode("utf-8"), key))


def read_lines(path: str | os.PathLike, key: bytes | None = None) -> list[str]:
    text = decrypt_bytes(pathlib.Path(path).read_bytes(), key).decode("utf-8")
    return [l for l in text.split("\n") if l]
