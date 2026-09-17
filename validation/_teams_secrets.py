"""Credential detection for the chat corpus - the one place the default is REMOVE.

Everywhere else in this pipeline a doubtful match is REPORTED rather than rewritten,
because silently editing text you cannot inspect is worse than leaving it alone. Here
that trade inverts. A false positive drops one message out of ~10,000 and costs
nothing. A false negative puts a live credential into a training corpus, where it is
unrecoverable - you cannot un-train a password, and a model that has seen one will
happily complete it.

So this module is deliberately greedy, and it QUARANTINES THE WHOLE MESSAGE rather
than masking the secret inside it. Masking is not enough: the sentence around a
credential is what says which account it belongs to ("le mdp du compte admin de X
c'est ..."), and half a leak is still a leak.

Three detectors, in decreasing confidence:

  1. KNOWN PREFIXES   Shopify (shpat_/shpss_/shpca_/shppa_), OpenAI sk-, GitHub ghp_,
                      AWS AKIA, Google AIza, Slack xox, Stripe sk_live/pk_live, JWTs,
                      PEM blocks, and URLs or connection strings with an inline
                      password. These are unambiguous - a hit is a secret.
  2. LABEL + VALUE    a credential word (mdp, mot de passe, password, user, login,
                      identifiant, clé, token, secret, acces) within ~60 characters of
                      something that looks like a value. This is the one that catches
                      how people actually share them in chat, in French and English.
  3. ENTROPY          a bare high-entropy token: 8+ chars, at least two character
                      classes, and a flat character distribution. Catches a password
                      pasted with no label at all.

The quarantine report names WHICH accounts appeared and WHEN, with the secret itself
masked - enough to drive a rotation list, without re-printing the passwords into a
second file. Rotation is the actual fix: scrubbing the corpus stops the credential
spreading further, it does not un-share it. It was already sent over the wire and it
is already sitting in the chat history on the server.
"""
from __future__ import annotations

import math
import re

# --- 1. unambiguous ---------------------------------------------------------------
PREFIXES = re.compile(
    r"\b("
    r"shpat_\w+|shpss_\w+|shpca_\w+|shppa_\w+"          # Shopify admin / shared secret
    r"|sk-[A-Za-z0-9_\-]{16,}|sk_live_\w+|pk_live_\w+|rk_live_\w+"   # OpenAI / Stripe
    r"|ghp_\w{20,}|gho_\w{20,}|github_pat_\w{20,}"      # GitHub
    r"|AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}"             # AWS
    r"|AIza[0-9A-Za-z_\-]{30,}"                         # Google
    r"|xox[abposr]-\w{8,}"                              # Slack
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\."   # JWT
    r"|glpat-\w{15,}|dop_v1_\w{20,}|SG\.\w{15,}"        # GitLab / DigitalOcean / SendGrid
    r")", re.I)

PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
# scheme://user:password@host  and  Server=..;Password=..;
URL_CRED = re.compile(r"[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]+@", re.I)
CONN_STR = re.compile(r"\b(pwd|password|pass|uid|user id)\s*=\s*[^\s;]{3,}", re.I)

# --- 1b. user:pass, the way they were actually shared -----------------------------
# Owner, on how credentials appear in this corpus: labelled "password" / "mdp" /
# "mot de passe", and pasted in `user:pass` form. So that shape gets its own
# high-confidence detector rather than relying on the entropy fallback.
#
# The guards are what keep it from eating the rest of the chat: no whitespace around
# the colon, both sides >= 3 chars, and a right-hand side that actually looks like a
# secret rather than a word. Excluded up front: URLs (scheme://), clock times (14:30),
# and `key: value` written with a space, which is how people write prose.
USER_PASS = re.compile(
    r"(?<![\w./:@])"
    r"([A-Za-z0-9._%+\-]{3,64})"          # user, or an email
    r":"
    r"([^\s:;,\"']{3,64})"                # pass - no spaces, so prose can't match
    r"(?![\w.])")
TIME_LIKE = re.compile(r"^\d{1,2}:\d{2}$")

# --- 2. label near a value --------------------------------------------------------
# French first, because that is the register: mdp, mot de passe, usager, identifiant.
LABEL = re.compile(
    r"\b(mdp|mots?\s*de\s*passe|passwords?|passwd|pwd|pass|usager|utilisateur|username|"
    r"user|login|identifiant|cl[eé]s?\s*(api|d.acc[eè]s)?|api[\s_-]?key|secret|token|"
    r"acc[eè]s|credential|compte\s+admin|admin\s+account|pin)\b", re.I)
# Something that could BE a value: no spaces, 6+ chars, not a bare French word.
VALUE = re.compile(r"(?<![\w@.])[A-Za-z0-9!@#$%^&*()_+\-=\[\]{};:,.?~]{6,}(?![\w])")
NEAR = 60


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _classes(s: str) -> int:
    return sum(bool(re.search(p, s)) for p in
               (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))


# Things that look high-entropy but are not secrets: hex colours, SKUs the shop uses,
# file hashes in a code discussion, ISO timestamps, version strings. Dropping these
# anyway is harmless, but naming them keeps the quarantine report readable.
BENIGN = re.compile(r"^(#[0-9a-f]{3,8}|\d{4}-\d{2}-\d{2}.*|v?\d+\.\d+[\d.]*|"
                    r"[A-Z]{2,4}\d{2,6}(-[A-Z]{1,3})?)$", re.I)


def looks_like_value(tok: str) -> bool:
    if len(tok) < 8 or BENIGN.match(tok):
        return False
    if _classes(tok) < 2:
        return False
    return _entropy(tok) > 2.6


# Known usernames and mail domains, read from an off-repo file so a client's domain is
# not committed to the repo. One per line; `#` comments and blanks ignored. A line
# containing any of these next to a colon is treated as a credential with no further
# test - the owner says the usernames in this corpus are mostly one short handle and
# addresses at one domain, and that is a far stronger signal than entropy.
HINTS_FILE = "secrets_hints.txt"
_HINTS: list[str] = []


def load_hints(path) -> list[str]:
    global _HINTS
    try:
        _HINTS = [l.strip().lower() for l in open(path, encoding="utf-8").read().splitlines()
                  if l.strip() and not l.startswith("#")]
    except OSError:
        _HINTS = []
    return _HINTS


def _user_pass_hits(text: str) -> list[str]:
    out = []
    for m in USER_PASS.finditer(text):
        whole, user, pwd = m.group(0), m.group(1), m.group(2)
        if TIME_LIKE.match(whole) or "//" in whole:
            continue
        low = user.lower()
        if any(h in low for h in _HINTS):          # known handle or domain -> certain
            out.append("user:pass (known username)")
            continue
        if "@" in user and "." in user:            # an email on the left is conclusive
            out.append("user:pass (email)")
            continue
        if len(pwd) >= 6 and _classes(pwd) >= 2 and not BENIGN.match(pwd):
            out.append("user:pass")
    return out


def scan(text: str) -> list[str]:
    """Return the reasons this message is a credential risk; empty list means clean."""
    why = []
    if PEM.search(text):
        why.append("private-key-block")
    if PREFIXES.search(text):
        why.append("known-key-prefix")
    if URL_CRED.search(text):
        why.append("url-with-password")
    if CONN_STR.search(text):
        why.append("connection-string")
    why.extend(_user_pass_hits(text))
    # A base64 blob starting `ey` that never got its second dot - a half-copied JWT, a
    # bearer token pasted without its tail. The strict three-part pattern above misses
    # all of those, and the owner reported one may still be in the corpus.
    if re.search(r"(?<![\w])ey[A-Za-z0-9_\-]{14,}", text):
        why.append("loose-token (ey...)")
    # A known username anywhere in a message that also carries a credential label is
    # enough on its own - the password may be on the next line, or in an attachment.
    low = text.lower()
    if _HINTS and any(h in low for h in _HINTS) and LABEL.search(text):
        why.append("known username + credential label")

    for m in LABEL.finditer(text):
        window = text[m.end():m.end() + NEAR]
        # a value on the same line, or on the next one (people paste them underneath)
        for tok in VALUE.findall(window):
            if looks_like_value(tok) or (len(tok) >= 6 and _classes(tok) >= 3):
                why.append(f"label:{m.group(0).lower().strip()}")
                break
        else:
            continue
        break

    if not why:
        for tok in VALUE.findall(text):
            if looks_like_value(tok) and _classes(tok) >= 3:
                why.append("high-entropy-token")
                break
    return why


def mask(text: str) -> str:
    """For the quarantine report: keep the sentence, blank the values.

    The point of the report is 'which account, and when' so a rotation list can be
    written. Re-printing the secret into a second file would just move the problem."""
    def blank(m):
        t = m.group(0)
        return t if not (looks_like_value(t) or _classes(t) >= 3 and len(t) >= 6) else \
            f"<{len(t)} chars redacted>"
    out = PREFIXES.sub("<key redacted>", text)
    out = URL_CRED.sub("<url-credential redacted>//", out)
    out = PEM.sub("<private key redacted>", out)
    out = VALUE.sub(blank, out)
    return out
