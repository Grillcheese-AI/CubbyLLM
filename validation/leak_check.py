"""Grep the finished corpus for everything that should already be gone.

The `reply_to` leak is the reason this file exists. The scrub was written to clean
message TEXT, and when the structured export added nested metadata - a reply's original
author, carried as a full name - it went into the corpus untouched. No audit section
covered it, because the audit reports on what the scrub did, and the scrub never looked
there.

So this checks the OUTPUT rather than the process: read the shipped file back as raw
JSON lines and look for anything that should not survive, wherever it sits - inside a
value, a key, a nested object, a list. A scrub that reports success is not evidence; a
corpus that greps clean is.

Run it after every harvest. It costs a second and it catches the class of bug where the
pipeline is correct and the data is not.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
REDACT = ROOT / "validation" / "redact.txt"
HINTS = ROOT / "validation" / "secrets_hints.txt"

PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "url": re.compile(r"https?://\S+"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "money": re.compile(r"\$\s?\d[\d ,.]*|\d[\d ,.]*\s?\$"),
    "loose-token": re.compile(r"(?<![\w])ey[A-Za-z0-9_\-]{14,}"),
    "key-prefix": re.compile(r"\b(shpat_|shpss_|sk-|sk_live_|ghp_|AKIA|AIza|xox[abp]-)\w+"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def names_to_check() -> list[str]:
    """Only what the redaction map was told to REMOVE.

    `secrets_hints.txt` is a detection aid - a username or a domain that makes a nearby
    password obvious - not a list of things that must vanish. Checking against it made
    the first run report `prextra` x276 as a leak when nobody had asked for the vendor
    to be redacted at all. A leak check that cries wolf is one nobody reads."""
    out = []
    for path in (REDACT,):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("=", 1)[0].strip() if "=" in line else line
            if len(name) >= 3 and not name.startswith("["):
                out.append(name)
    return sorted(set(out), key=len, reverse=True)


def main() -> int:
    if not CORPUS.exists():
        raise SystemExit(f"no corpus at {CORPUS}")
    raw = CORPUS.read_text(encoding="utf-8")
    lines = raw.splitlines()

    findings = []
    for label, pat in PATTERNS.items():
        hits = pat.findall(raw)
        if hits:
            findings.append((label, len(hits), hits[:3]))

    for name in names_to_check():
        pat = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])", re.I)
        n = len(pat.findall(raw))
        if n:
            findings.append((f"name:{name}", n, []))

    # Anything that still looks like a person's full name, wherever it sits.
    fullname = re.compile(r'"[A-ZÀ-Þ][a-zà-ÿ]+(?:-[A-ZÀ-Þ][a-zà-ÿ]+)? [A-ZÀ-Þ][a-zà-ÿ]{2,}"')
    fn = fullname.findall(raw)
    if fn:
        findings.append(("full-name-shaped string", len(fn), sorted(set(fn))[:5]))

    print(f"{len(lines)} rows checked in {CORPUS.name}\n")
    if not findings:
        print("CLEAN - nothing matched.")
        return 0
    print("LEAKS:")
    for label, n, sample in findings:
        s = ("  e.g. " + ", ".join(str(x)[:60] for x in sample)) if sample else ""
        print(f"  {label:<34} x{n}{s}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
