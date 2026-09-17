"""teams_harvest - real Quebec work chat into a scrubbed, reviewable corpus.

Wired: STANDALONE (data-building only). THIS SCRIPT NEVER CALLS A NETWORK. Parsing and
scrubbing only, on purpose: the labelling pass that assigns Plutchik petal and intensity
sends text to a third party, and it must run on the SCRUBBED file, after a human has read
the audit. Two scripts, in that order, so nothing unreviewed can leave the machine.

WHY THIS CORPUS EXISTS. Four frontier models were asked to write Quebec French affect
(`validation/fr_affect_pilot.py`). All four produced a performance of Quebecness -
sacres in a third of all lines, "coudonc", "ayoye", zero anglicisms, intensity by lexical
escalation. Real Quebec work chat does almost none of that: sacres are rare and spike-
bound, `lol` carries most of the discharge, intensity runs by UNDERSTATEMENT ("c pas
pire", "un ti peu" while the ETA just went 16h to 100h), and English loanwords are
constant and unmarked. A description of a register produces caricature. The register
itself does not. So: harvest, do not generate.

That is the same lesson three times now - translated GoEmotions gave us the worst-scoring
family in every stand-in read, a described register gave us caricature, and formal French
(689 files in the training tree clear a French-word threshold and every one scores ~0 on
informal markers) is not the thing either.

THE SCRUB IS A REVIEW LOOP, NOT A GUESS. Three tiers, deliberately different:

  REPLACED   things with an unambiguous shape - money, urls, emails, phone numbers, and
             whatever names are listed in the redaction file.
  DROPPED    pasted machine output (dashboards, tracebacks, invoices). Not register, and
             it is where numbers and identifiers hide in bulk.
  REPORTED   capitalised tokens that are not in the safe list. NOT auto-replaced: a
             heuristic that silently rewrites proper nouns also mangles Shopify, Claude
             and Toronto, and a corpus you cannot trust the edits in is worse than one
             you have not edited. These go in the audit for a human to sort.

The redaction list lives OFF-REPO (`validation/redact.txt`, gitignored) because a list of
a client's names committed to a repo is itself the leak it was meant to prevent. One entry
per line, `name = REPLACEMENT`, blank lines and # comments ignored.

    python validation/teams_harvest.py <export.txt>            # parse, scrub, audit
    python validation/teams_harvest.py <export.txt> --report-only   # audit, write nothing
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "standin" / "data" / "out" / "teams_corpus.jsonl"
AUDIT = ROOT / "standin" / "data" / "out" / "teams_scrub_audit.md"
REDACT_FILE = ROOT / "validation" / "redact.txt"          # gitignored

# The Teams copy-out has TWO shapes and its own interface chrome mixed in; `_teams_parse`
# holds the reader and explains both. Keeping it separate because the parse is the part
# most likely to need changing when a different export lands.
from _teams_parse import parse  # noqa: E402
# The STRUCTURED export is the better input by a wide margin: it carries timestamps,
# reactions and reply threading that the copy-paste has no way to express, and those
# are affect signal the clock and the other person recorded for free.
from _teams_json import parse_json  # noqa: E402
# Credentials are the ONE place this pipeline auto-removes instead of reporting: a
# false positive costs one message in ten thousand, a false negative puts a live
# password in a training corpus where it cannot be taken back. Whole messages are
# quarantined, not masked - the sentence around a password says which account it is.
import _teams_secrets as SECRETS  # noqa: E402
# Name forms derived from the export's participant list, because the hand-written file
# reliably misses the nicknames and initials people actually use.
import _teams_names as NAMES  # noqa: E402

QUARANTINE = ROOT / "standin" / "data" / "out" / "teams_quarantine.md"
HINTS_FILE = ROOT / "validation" / "secrets_hints.txt"    # gitignored

# The lookbehind was `(?<![\w.])` to avoid matching inside an identifier, but that also
# skipped `US$50` and anything where the sign follows a letter - two amounts survived
# the first clean pass because of it. Only a digit before the sign means "not money".
MONEY = re.compile(r"(?<!\d)\$\s?\d[\d ,.]*|(?<![\w.])\d[\d ,.]*\s?\$(?![\w])")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")

# A pasted dashboard/traceback is not conversation. Cheap, high-precision markers.
MACHINE = (
    re.compile(r"^\s*\[#*\s*\]|\brunner state\b|\beta\s*:|\brate\s*:|\bcatalogue\s*:", re.I),
    re.compile(r"^\s*(Traceback|File \"|\s{2,}at )", re.I),
    re.compile(r"={10,}|-{20,}|\.{10,}"),
    re.compile(r"\b(INVOICE|TOKEN CONSUMPTION|DAILY SUB-TOTAL|Billing Period)\b"),
)

# Capitalised words that are NOT somebody's name or a client - platforms, tools, places,
# model names. Kept lowercase for comparison. Extend freely; a false entry here only
# means one more token the audit does not ask about.
SAFE = {
    "shopify", "claude", "chatgpt", "gemini", "openai", "anthropic", "grok", "qwen",
    "kimi", "google", "microsoft", "teams", "python", "toronto", "quebec", "québec",
    "canada", "amazon", "lego", "api", "ia", "ai", "sql", "json", "csv", "http",
    "ok", "oui", "non", "yes", "yep", "loll", "lol", "image", "begin", "user", "you",
    "bon", "ah", "ben", "la", "il", "je", "on", "tu", "ca", "ça", "c", "j", "y", "a",
    "et", "de", "le", "les", "un", "une", "mais", "pis", "pour", "dans", "avec",
}
WORD = re.compile(r"\b[A-ZÀ-Þ][\wÀ-ÿ'-]{1,}\b")


def load_redactions() -> dict[str, str]:
    """`name = REPLACEMENT` per line, from an off-repo file. Longest first at apply time
    so 'Galerie du Jouet' wins over 'Galerie'."""
    out: dict[str, str] = {}
    if REDACT_FILE.exists():
        for line in REDACT_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_text(path: pathlib.Path) -> str:
    """Decode without guessing wrong quietly. utf-8 first; anything that needs a fallback
    is reported, because silent latin-1 is how a corpus ends up full of mojibake."""
    b = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            t = b.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc not in ("utf-8", "utf-8-sig"):
            print(f"  note: decoded as {enc}, not utf-8", file=sys.stderr)
        return unicodedata.normalize("NFC", t)
    raise SystemExit("could not decode the export in any known encoding")


def is_machine(t: str) -> bool:
    return any(p.search(t) for p in MACHINE) or len(t) > 600


def scrub(t: str, red: dict[str, str], hits: Counter) -> str:
    for name in sorted(red, key=len, reverse=True):
        pat = re.compile(re.escape(name), re.I)
        t, n = pat.subn(red[name], t)
        if n:
            hits[f"name:{name}"] += n
    for label, pat in (("email", EMAIL), ("url", URL), ("phone", PHONE), ("money", MONEY)):
        t, n = pat.subn(f"[{label.upper()}]", t)
        if n:
            hits[label] += n
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="path to the chat export (plain text)")
    ap.add_argument("--report-only", action="store_true", help="write the audit, not the corpus")
    a = ap.parse_args()

    src = pathlib.Path(a.export)
    meta = {}
    if src.suffix.lower() == ".json":
        msgs, meta = parse_json(src)
    else:
        msgs = parse(read_text(src))
    if not msgs:
        raise SystemExit("no messages parsed - unrecognised export shape")

    red = load_redactions()
    SECRETS.load_hints(HINTS_FILE)

    # Derive the name forms from the export's own participant list rather than relying
    # on the hand-written file. The hand list caught "Nicolas Cloutier" and missed
    # `Nic`, `JF` and the bare surnames - which is what a hand list always does, because
    # nicknames are the forms nobody thinks to write down.
    texts = [m["text"] for m in msgs]
    parts = [p.get("name") for p in (meta.get("participants") or []) if p.get("name")]
    auto, reported_forms = ({}, [])
    if parts:
        def tag_for(who):
            return red.get(who) or ("[SELF]" if red.get("You") == "[SELF]"
                                    and who == parts[0] else "[PEER]")
        # keep whatever the file already says, and let it win over a derived guess
        explicit = {w: t for w, t in red.items() if t.startswith("[")}
        auto, reported_forms = NAMES.build_map(parts, texts,
                                               lambda w: explicit.get(w, "[PEER]"))
        for form, tag in auto.items():
            red.setdefault(form, tag)
    hits, kept, dropped, candidates = Counter(), [], [], Counter()
    quarantined, reasons = [], Counter()
    for m in msgs:
        # Credentials first, before anything else touches the text. A quarantined
        # message never reaches the corpus, the review list, or the audit's samples.
        why = SECRETS.scan(m["text"])
        if why:
            reasons.update(why)
            quarantined.append({**m, "why": why})
            continue
        if is_machine(m["text"]):
            dropped.append(m)
            continue
        clean = scrub(m["text"], red, hits)
        # Don't offer our own placeholders back as review candidates.
        for w in WORD.findall(re.sub(r"\[[A-Z_]+\]", " ", clean)):
            if w.lower() not in SAFE:
                candidates[w] += 1
        # The SPEAKER field goes through the same redaction map. That is also what
        # merges an export's two names for one person: Teams labels the owner's own
        # messages "You" in some views and by name in others, so both map to [SELF]
        # and the split stops being a split.
        who = red.get(m["speaker"], m["speaker"])
        row = {"speaker": who, "when": m["when"], "text": clean,
               "words": len(clean.split())}
        # Carry the structured export's extra signal through when it exists. The text
        # path simply has none of it, so the key is absent rather than faked.
        for k in ("gap_s", "burst", "reactions", "reply_to", "edited", "attachments"):
            if k in m:
                row[k] = m[k]
        # `reply_to` is a nested object carrying the ORIGINAL AUTHOR'S FULL NAME, and it
        # went into the corpus untouched when the structured fields were added - a leak
        # in the metadata rather than the text, which is exactly what a scrub aimed at
        # message bodies will miss. Found by grepping the output for a name that should
        # have been gone. The author goes through the same map as the speaker; the
        # quoted preview is dropped entirely, since it is a copy of another message that
        # already appears in the corpus in its own scrubbed form.
        rt = row.get("reply_to")
        if isinstance(rt, dict):
            row["reply_to"] = {
                "speaker": red.get(rt.get("author"), "[PEER]") if rt.get("author") else None,
                "when": rt.get("timestamp"),
            }
        kept.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        with open(OUT, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The quarantine report is a ROTATION LIST, not a second copy of the secrets. The
    # username stays (it says which account); the password is blanked. Scrubbing the
    # corpus stops a credential spreading further - it does not un-share it, so the
    # only real fix is rotating what appears here.
    with open(QUARANTINE, "w", encoding="utf-8") as f:
        f.write("# Credential quarantine\n\n")
        f.write(f"**{len(quarantined)} messages** were pulled out before anything else "
                f"ran. They are NOT in the corpus, the audit samples, or the review list.\n\n")
        f.write("Passwords below are blanked; usernames are kept, because the username is "
                "what tells you which account to rotate. **Rotating is the fix** - this file "
                "only stops the credential going any further. It was already sent over the "
                "wire and it is still in the chat history on the server.\n\n")
        f.write("Delete this file once you have worked through it.\n\n")
        f.write("## Why they tripped\n\n")
        for r, n in reasons.most_common():
            f.write(f"- {r} - {n}\n")
        # Split by confidence. Bare high-entropy is the weakest detector by far - it
        # eats product SKUs, filenames and code identifiers - and it fires on ~80% of
        # the quarantine. Dropping those anyway costs ~2% of the corpus, which is
        # nothing, but making someone read 260 entries to find the 20 that need a
        # rotation is how a rotation list gets ignored. So: the ones to ACT on first,
        # then the ones dropped only to be safe.
        confirmed = [q for q in quarantined if q["why"] != ["high-entropy-token"]]
        precaution = [q for q in quarantined if q["why"] == ["high-entropy-token"]]
        f.write(f"\n## Act on these ({len(confirmed)})\n\n")
        f.write("A credential label, a `user:pass`, a known username, or a recognised "
                "key prefix. This is the rotation list.\n\n")
        for q in confirmed:
            f.write(f"- `{q.get('when') or '?'}` **{q['speaker']}** "
                    f"_({', '.join(sorted(set(q['why'])))})_\n  > {SECRETS.mask(q['text'])[:300]}\n")
        f.write(f"\n## Dropped to be safe ({len(precaution)})\n\n")
        f.write("Bare high-entropy strings with no credential label anywhere near them. "
                "Mostly SKUs, filenames and identifiers. Listed for completeness; there is "
                "probably nothing here to rotate.\n\n")
        for q in precaution:
            f.write(f"- `{q.get('when') or '?'}` {q['speaker']}: "
                    f"{SECRETS.mask(q['text'])[:140]}\n")

    who = Counter(r["speaker"] for r in kept)
    with open(AUDIT, "w", encoding="utf-8") as f:
        f.write("# Scrub audit\n\n")
        f.write(f"Source: one chat export. Parsed **{len(msgs)}** messages: "
                f"**{len(kept)}** kept, **{len(dropped)}** dropped as machine output.\n\n")
        f.write("Redaction list is read from an off-repo file and is not committed. "
                "Nothing in this pipeline touches the network - the labelling pass is a "
                "separate script and runs on the scrubbed corpus, after this audit is read.\n\n")
        f.write("## Speakers\n\n" + "\n".join(f"- `{s}` - {n} messages" for s, n in who.most_common()) + "\n\n")

        # The question this corpus exists to answer: does the register show up, and at
        # what RATE? The four-model pilot put sacres in ~1 line in 3. If the real rate
        # is two orders of magnitude lower, that is the finding, and it is the number
        # that makes generated Quebec French unusable rather than merely suspect.
        QC = re.compile(r"\b(pcq|fac|tk|asteur|pis|ouais|ouin|jpense|chu|p-e|toe|quin|"
                        r"dla|ste|icitte|pantoute|ben|dac|fodrais|messemble|tanne|tanné)\b", re.I)
        SACRE = re.compile(r"\b(tabarnak|tabarnac|calisse|câlisse|ostie|osti|criss|crisse|"
                           r"tabarnouche|tabarouette|batince|caline|câline|mosus|viarge|marde)\b", re.I)
        LOL = re.compile(r"\b(lol+|haha+|hihi|mdr|loll+)\b", re.I)
        ANGLO = re.compile(r"\b(setup|setuper|booker|checker|check|run|roule|down|switch|"
                           r"rush|fix|update|shipping|tag|tags|deal|team|job|test|tester|"
                           r"yes|yep|ok|bug|log|batch|sync|syncker)\b", re.I)
        EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
        n = len(kept) or 1
        f.write("## Register profile\n\n")
        f.write("| marker | messages | rate |\n|---|---:|---:|\n")
        for label, pat in (("Quebec markers", QC), ("sacres", SACRE), ("lol / haha", LOL),
                           ("anglicisms", ANGLO), ("emoji", EMOJI)):
            c = sum(1 for r in kept if pat.search(r["text"]))
            f.write(f"| {label} | {c} | {c / n:.1%} |\n")
        react = sum(1 for r in kept if r.get("reactions"))
        gaps = [r["gap_s"] for r in kept if isinstance(r.get("gap_s"), (int, float))]
        fast = sum(1 for g in gaps if g <= 30)
        f.write(f"| carries a reaction | {react} | {react / n:.1%} |\n")
        if gaps:
            f.write(f"| sent within 30s of the last | {fast} | {fast / len(gaps):.1%} |\n")
        f.write(f"\nMedian words per message: "
                f"**{sorted(r['words'] for r in kept)[len(kept) // 2]}**.\n\n")

        if auto or reported_forms:
            f.write("## Name forms derived from the export\n\n")
            f.write("Generated from the participant list and kept only where the form "
                    "actually occurs in the text. This is what the hand-written file "
                    "missed - nicknames and initials are the forms nobody writes down.\n\n")
            f.write("- auto-redacted: " + ", ".join(f"`{k}`" for k in
                                                    sorted(auto, key=len, reverse=True)) + "\n")
            if reported_forms:
                f.write("- reported, NOT replaced (too short or too rare to trust): "
                        + ", ".join(f"`{f}` x{n}" for f, n, _, _ in reported_forms) + "\n")
            tp = NAMES.third_parties(texts)
            if tp:
                f.write("- third parties mentioned (reported only - you know which are "
                        "people): " + ", ".join(f"`{w}` x{n}" for w, n in tp[:20]) + "\n")
            f.write("\n")

        f.write("## Replaced\n\n")
        f.write("\n".join(f"- {k} x{v}" for k, v in hits.most_common()) if hits
                else "- nothing matched - if that is a surprise, the redaction file is empty or missing")
        f.write("\n\n## Dropped as machine output\n\n")
        for m in dropped[:20]:
            f.write(f"- `{m['text'][:100].replace(chr(10), ' ')}...`\n")
        if len(dropped) > 20:
            f.write(f"- ... and {len(dropped) - 20} more\n")
        f.write("\n## Capitalised tokens NOT auto-replaced - review these\n\n")
        f.write("Each of these is either fine (a tool, a place, a platform) or a name that "
                "belongs in the redaction file. Nothing here was rewritten: a heuristic that "
                "silently edits proper nouns also mangles the harmless ones, and edits you "
                "cannot see are worse than edits you have not made.\n\n")
        for w, n in candidates.most_common(60):
            f.write(f"- `{w}` x{n}\n")
        f.write(f"\n({len(candidates)} distinct)\n")

    print(f"parsed {len(msgs)}  kept {len(kept)}  dropped {len(dropped)}")
    print(f"audit  {AUDIT}")
    if not a.report_only:
        print(f"corpus {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
