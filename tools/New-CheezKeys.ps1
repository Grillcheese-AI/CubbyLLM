<#
.SYNOPSIS
  Makes the keys CheezML records are sealed with (order 0076), straight into Windows Credential Manager.

.DESCRIPTION
  Three keys, as cubby_serve/records/envelope.py takes them (its Keys class):
    sign          Ed25519, signs each record's public envelope (anyone checks it with the public key)
    conversation  AES-256-GCM, opens the conversation part (the tenant's members)
    provenance    AES-256-GCM, opens the provenance part (the audit view only)

  Each key is made in memory by python3's `cryptography` and written directly to Credential Manager
  (generic credential, this Windows user, this machine only; encrypted by Windows with the user's logon).
  No key is ever printed, logged, put in a file, passed as an argument or sent anywhere.
  Only the PUBLIC signing key is written out, so verifiers can check signatures.

  Where the keys live (Control Panel > Credential Manager > Windows Credentials):
    GRL/CheezML/sign/<kid>           the Ed25519 private key, as a JWK
    GRL/CheezML/conversation/<kid>   the AES-256 key, as a JWK
    GRL/CheezML/provenance/<kid>     the AES-256 key, as a JWK
    GRL/CheezML/current              which kids are in use (not secret)

  Nothing is ever deleted or overwritten. A new set (-Rotate) is added beside the old one, and "current" points to
  it, so records sealed with older keys still open.

.PARAMETER Check
  Reads the current keys and proves they work (sign and verify, encrypt and decrypt) without showing them.

.PARAMETER Rotate
  Makes a new set even though one exists. The old keys stay.

.PARAMETER SelfTest
  Checks that this script can write and read Credential Manager, with a dummy entry that holds no key and is removed
  right after. Makes no key.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\New-CheezKeys.ps1
  powershell -ExecutionPolicy Bypass -File tools\New-CheezKeys.ps1 -Check
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Rotate,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

$mode = 'make'
if ($SelfTest) { $mode = 'selftest' }
elseif ($Check) { $mode = 'check' }
elseif ($Rotate) { $mode = 'rotate' }

if ($mode -eq 'rotate') {
    Write-Host 'This adds a NEW set of keys and makes it current. The old keys stay in Credential Manager.'
    $answer = Read-Host 'Type ROTATE to go on'
    if ($answer -ne 'ROTATE') { Write-Host 'Nothing made.'; exit 1 }
}

if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) {
    Write-Error 'python3 was not found. The keys are made by python3 with the cryptography package.'
}

$publicDir = Join-Path $HOME '.grl-keys'

# The Python below makes and stores the keys itself, so no key ever passes through PowerShell.
# It is ASCII only (Windows PowerShell pipes it to python3 as ASCII).
$py = @'
import base64, ctypes, datetime, json, os, secrets, sys
from ctypes import wintypes as w

PREFIX = "GRL/CheezML"
ROLES = ("sign", "conversation", "provenance")
CRED_TYPE_GENERIC, CRED_PERSIST_LOCAL_MACHINE, ERROR_NOT_FOUND = 1, 2, 1168


class FILETIME(ctypes.Structure):
    _fields_ = [("lo", w.DWORD), ("hi", w.DWORD)]


class CREDENTIALW(ctypes.Structure):
    _fields_ = [("Flags", w.DWORD), ("Type", w.DWORD), ("TargetName", w.LPWSTR), ("Comment", w.LPWSTR),
                ("LastWritten", FILETIME), ("CredentialBlobSize", w.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", w.DWORD),
                ("AttributeCount", w.DWORD), ("Attributes", ctypes.c_void_p), ("TargetAlias", w.LPWSTR),
                ("UserName", w.LPWSTR)]


adv = ctypes.WinDLL("advapi32", use_last_error=True)
adv.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), w.DWORD]
adv.CredWriteW.restype = w.BOOL
adv.CredReadW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
adv.CredReadW.restype = w.BOOL
adv.CredDeleteW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD]
adv.CredDeleteW.restype = w.BOOL
adv.CredFree.argtypes = [ctypes.c_void_p]


def write(target, user, blob, comment):
    buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    c = CREDENTIALW()
    c.Type, c.TargetName, c.Comment, c.UserName = CRED_TYPE_GENERIC, target, comment, user
    c.CredentialBlobSize, c.CredentialBlob = len(blob), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    c.Persist = CRED_PERSIST_LOCAL_MACHINE
    if not adv.CredWriteW(ctypes.byref(c), 0):
        raise OSError("Credential Manager refused the write (error %d)" % ctypes.get_last_error())


def read(target):
    p = ctypes.POINTER(CREDENTIALW)()
    if not adv.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(p)):
        if ctypes.get_last_error() == ERROR_NOT_FOUND:
            return None
        raise OSError("Credential Manager refused the read (error %d)" % ctypes.get_last_error())
    try:
        return bytes(p.contents.CredentialBlob[:p.contents.CredentialBlobSize])
    finally:
        adv.CredFree(p)


def b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make(public_dir, rotate):
    from cryptography.hazmat.primitives import serialization as s
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    old = read(PREFIX + "/current")
    if old is not None and not rotate:
        cur = json.loads(old)
        print("Keys already exist (made %s): nothing made, nothing changed." % cur.get("made"))
        for r in ROLES:
            print("  %-13s %s" % (r, cur[r]))
        print("To add a new set beside them, run with -Rotate.")
        return 0
    day = datetime.date.today().strftime("%Y%m%d")
    kids = {r: "grl-%s-%s-%s" % (r, day, secrets.token_hex(4)) for r in ROLES}
    for r in ROLES:
        if read("%s/%s/%s" % (PREFIX, r, kids[r])) is not None:
            raise RuntimeError("a key with that id already exists; run again")
    raw = Ed25519PrivateKey.generate()
    pub = raw.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw)
    priv = raw.private_bytes(s.Encoding.Raw, s.PrivateFormat.Raw, s.NoEncryption())
    jwk = {"sign": {"kty": "OKP", "crv": "Ed25519", "x": b64(pub), "d": b64(priv), "kid": kids["sign"],
                    "use": "sig", "alg": "Ed25519"}}
    for r in ("conversation", "provenance"):
        jwk[r] = {"kty": "oct", "k": b64(secrets.token_bytes(32)), "kid": kids[r], "use": "enc", "alg": "A256GCM"}
    made = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    what = {"sign": "Ed25519 signing key", "conversation": "AES-256 key, conversation part",
            "provenance": "AES-256 key, provenance part"}
    for r in ROLES:
        write("%s/%s/%s" % (PREFIX, r, kids[r]), kids[r], json.dumps(jwk[r]).encode("utf-8"),
              "CheezML %s (order 0076), made %s" % (what[r], made))
    for r in ROLES:  # read each back before pointing "current" at them
        if json.loads(read("%s/%s/%s" % (PREFIX, r, kids[r]))) != jwk[r]:
            raise RuntimeError("a key did not read back the same; current not changed")
    current = dict(kids, made=made)  # older sets stay under their own kids: a record's kid finds its key
    write(PREFIX + "/current", "current", json.dumps(current).encode("utf-8"), "CheezML keys in use (not secret)")
    os.makedirs(public_dir, exist_ok=True)
    public = {"kty": "OKP", "crv": "Ed25519", "x": b64(pub), "kid": kids["sign"], "use": "sig", "alg": "Ed25519"}
    path = os.path.join(public_dir, "cheezml-%s.public.jwk.json" % kids["sign"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(public, f, indent=1)
    del jwk, priv, raw
    print("Made and stored in Credential Manager (this user, this machine):")
    for r in ROLES:
        print("  %-13s %s" % (r, kids[r]))
    print("Public signing key (safe to share): " + path)
    print(json.dumps(public))
    return 0


def check():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    cur = read(PREFIX + "/current")
    if cur is None:
        print("No keys yet: run the script without -Check to make them.")
        return 1
    cur = json.loads(cur)
    keys = {}
    for r in ROLES:
        blob = read("%s/%s/%s" % (PREFIX, r, cur[r]))
        if blob is None:
            print("MISSING: %s key %s is not in Credential Manager." % (r, cur[r]))
            return 1
        keys[r] = json.loads(blob)
    sk = Ed25519PrivateKey.from_private_bytes(unb64(keys["sign"]["d"]))
    if b64(sk.public_key().public_bytes_raw()) != keys["sign"]["x"]:
        print("BAD: the signing key's public half does not match.")
        return 1
    msg = secrets.token_bytes(64)
    sk.public_key().verify(sk.sign(msg), msg)
    for r in ("conversation", "provenance"):
        k = unb64(keys[r]["k"])
        if len(k) != 32:
            print("BAD: the %s key is not 256 bits." % r)
            return 1
        nonce = secrets.token_bytes(12)
        if AESGCM(k).decrypt(nonce, AESGCM(k).encrypt(nonce, msg, None), None) != msg:
            print("BAD: the %s key did not decrypt what it encrypted." % r)
            return 1
    print("OK: the keys made %s sign, verify, encrypt and decrypt." % cur.get("made"))
    for r in ROLES:
        print("  %-13s %s" % (r, cur[r]))
    print("Public signing key: " + json.dumps({"kty": "OKP", "crv": "Ed25519", "x": keys["sign"]["x"],
                                               "kid": cur["sign"]}))
    return 0


def selftest():
    target, dummy = PREFIX + "-selftest/delete-me", b"not a key: a self-test of this script"
    write(target, "selftest", dummy, "self-test of New-CheezKeys.ps1, removed right after")
    ok = read(target) == dummy
    adv.CredDeleteW(target, CRED_TYPE_GENERIC, 0)
    gone = read(target) is None
    print("Credential Manager: write %s, read back %s, dummy removed %s." %
          ("ok", "ok" if ok else "DIFFERENT", "ok" if gone else "NO"))
    return 0 if ok and gone else 1


if __name__ == "__main__":
    mode, public_dir = sys.argv[1], sys.argv[2]
    try:
        if mode == "selftest":
            sys.exit(selftest())
        if mode == "check":
            sys.exit(check())
        sys.exit(make(public_dir, rotate=(mode == "rotate")))
    except Exception as e:  # never print a traceback: its locals could hold a key
        print("FAILED: %s" % type(e).__name__ + (": " + str(e) if isinstance(e, (OSError, RuntimeError)) else ""))
        sys.exit(1)
'@

$OutputEncoding = [System.Text.Encoding]::ASCII
$py | & python3 - $mode $publicDir
exit $LASTEXITCODE
