"""cubbyllm/speech/verb_kinds.py -- what each instruction verb asks for (order 0072, from `verb_table.py`).

A step of an instruction is a verb on an object. When the object is a thing in the conversation (a text, a list, a
table, the reply), the verb takes one of the **reply kinds**: pick, reorder, edit, compute, verdict, write (form only;
a meaning is never marked passed). When it is a file, a service, a device or an account, the verb is an **act**: a
change made by the host through a tool, gated by its class (read; change; ask first) and done only once its read-back
agrees.

`verb(name)` -> {"en", "fr": {infinitive, tu, nous, vous}, "kinds": [{"kind", "note", "args"}], "acts": [{"class",
"when", "read_back"}]}. `KIND_ARGS`: the arguments each kind needs (a verb may name its own, `VERB_ARGS`).

**French forms** (Nick, 2026-11-13): the imperative (tu, nous, vous) made by rule from the infinitive's group, with
`IRREGULAR` by hand; checked against UniMorph French off-repo (`validation/verb_forms_check.py`, the agreement count
committed, never a UniMorph row).

Wiring: STANDALONE (0072's programs and 0067's second reader read it; nothing in the serve path).
"""
from __future__ import annotations

import re

from cubbyllm.speech.verb_table import TABLE

__wiring__ = "STANDALONE"

KINDS = ("pick", "reorder", "edit", "compute", "verdict", "write")
CLASSES = ("read", "change", "ask first")
KIND_ARGS = {"pick": ("object", "condition", "all"), "reorder": ("object", "key"), "edit": ("object", "places", "change"),
             "compute": ("input", "function"), "verdict": ("object", "criteria"), "write": ("subject", "length", "language"),
             "act": ("target", "tool", "read_back")}
VERB_ARGS = {("move", "reorder"): ("object", "item", "place"), ("rename", "edit"): ("object", "old", "new"),
             ("replace", "edit"): ("object", "old", "new"), ("set", "edit"): ("object", "name", "value"),
             ("add", "edit"): ("object", "item", "place"), ("copy", "edit"): ("object", "place"),
             ("delete", "edit"): ("object", "items"), ("remove", "edit"): ("object", "items"),
             ("clear", "edit"): ("object",), ("reset", "edit"): ("object", "default"), ("format", "edit"): ("object", "form"),
             ("merge", "edit"): ("inputs",), ("translate", "write"): ("subject", "target_language"),
             ("summarize", "write"): ("subject", "share"), ("export", "compute"): ("input", "format"),
             ("print", "compute"): ("text",), ("parse", "compute"): ("input", "parser"),
             ("evaluate", "compute"): ("expression",), ("compare", "compute"): ("a", "b"), ("compare", "verdict"): ("a", "b")}
IRREGULAR = {  # infinitive: (tu, nous, vous)
    "construire": ("construis", "construisons", "construisez"), "décrire": ("décris", "décrivons", "décrivez"),
    "écrire": ("écris", "écrivons", "écrivez"), "extraire": ("extrais", "extrayons", "extrayez"),
    "obtenir": ("obtiens", "obtenons", "obtenez"), "ouvrir": ("ouvre", "ouvrons", "ouvrez"),
    "lire": ("lis", "lisons", "lisez"), "traduire": ("traduis", "traduisons", "traduisez"),
    "mettre": ("mets", "mettons", "mettez")}
SECOND_GROUP = {"définir"}                               # -ir with -iss- (finir); every other -ir verb here is irregular


def imperative(inf: str) -> tuple[str, str, str]:
    """(tu, nous, vous) of a French infinitive; a multi-word verb ("mettre à jour") conjugates its first word."""
    head, sep, rest = inf.partition(" ")
    tail = sep + rest
    if head in IRREGULAR:
        return tuple(f + tail for f in IRREGULAR[head])
    if head in SECOND_GROUP:
        st = head[:-2]
        return st + "is" + tail, st + "issons" + tail, st + "issez" + tail
    if not head.endswith("er"):
        raise ValueError(f"no rule for {inf!r}: add it to IRREGULAR")
    st = head[:-2]
    tu, nous, vous = st + "e", st + "ons", st + "ez"
    if st.endswith("c"):
        nous = st[:-1] + "çons"                           # déplacer -> déplaçons
    elif st.endswith("g"):
        nous = st + "eons"                                # charger -> chargeons
    if st.endswith(("oy", "uy")):
        tu = st[:-1] + "ie"                               # envoyer -> envoie
    m = re.search(r"é([^aeiouyéèêàâîôû]+)$", st)
    if m:
        tu = st[:m.start()] + "è" + m.group(1) + "e"      # générer -> génère, régler -> règle
    return tu + tail, nous + tail, vous + tail


def _kinds(verb: str, text: str) -> list[dict]:
    out = []
    for part in [p.strip() for p in text.split(";") if p.strip() and p.strip() != "-"]:
        kind = re.match(r"[a-z]+", part).group(0)
        assert kind in KINDS, (verb, part)
        out.append({"kind": kind, "note": part[len(kind):].strip(" :"), "args": VERB_ARGS.get((verb, kind), KIND_ARGS[kind])})
    return out


def _acts(text: str) -> list[dict]:
    if text.strip() == "-":
        return []
    out, last_rb = [], ""
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        cls = next((c for c in ("ask first", "change", "read") if part.startswith(c)), None)
        if cls is None:                                   # a note on the act before ("run the tests" is test)
            out[-1]["note"] = part
            continue
        rest = part[len(cls):]
        when, _, rb = rest.partition(":")
        rb = rb.strip() or last_rb
        last_rb = rb
        out.append({"class": cls, "when": when.strip(" ,") or None, "read_back": rb or None})
    for a in out:                                         # "change; ask first ...: closed": the read-back is shared
        a["read_back"] = a["read_back"] or last_rb or None
    return out


def verb(name: str) -> dict:
    fr, reply, outside = TABLE[name]
    tu, nous, vous = imperative(fr)
    return {"en": name, "fr": {"infinitive": fr, "tu": tu, "nous": nous, "vous": vous},
            "kinds": _kinds(name, reply), "acts": _acts(outside)}


def _resolve(verbs: dict) -> dict:
    """An act with no read-back of its own: "(as delete)" takes that verb's; a read reads its result back; a change
    reads back what it changed."""
    for v in verbs.values():
        for a in v["acts"]:
            if a["read_back"]:
                continue
            ref = re.search(r"\(as (\w+)\)", a["when"] or "")
            src = [b["read_back"] for b in verbs[ref.group(1)]["acts"] if b["read_back"]] if ref else []
            a["read_back"] = src[0] if src else ("the result, as read" if a["class"] == "read" else "reads back as changed")
    return verbs


VERBS = _resolve({name: verb(name) for name in TABLE})
WHEN = {  # an act's condition -> the context flag it reads (True: the act applies)
    "with unsaved changes": lambda c: c.get("unsaved", False), "unless in a sandbox": lambda c: not c.get("sandbox", False),
    "unless in a sandbox (as execute)": lambda c: not c.get("sandbox", False), "in a sandbox": lambda c: c.get("sandbox", False),
    "for data": lambda c: not c.get("software", False), "for software (as install)": lambda c: c.get("software", False),
    "into a shared branch": lambda c: c.get("shared", False), "(as delete)": lambda c: True,
    "(leaves the machine)": lambda c: True}
STRICT = {"read": 0, "change": 1, "ask first": 2}


def act_class(name: str, ctx: dict | None = None) -> dict | None:
    """The act that applies to `name` in `ctx` (flags: unsaved, sandbox, software, shared), the strictest when several
    do; "ask first unless in a sandbox" is a change inside one; None when nothing outside the reply is done (the host
    refuses: no tool)."""
    ctx = ctx or {}
    live = [a for a in VERBS[name]["acts"] if a["when"] is None or WHEN[a["when"]](ctx)]
    if not live and ctx.get("sandbox"):
        live = [dict(a, **{"class": "change", "when": "in a sandbox"}) for a in VERBS[name]["acts"]
                if (a["when"] or "").startswith("unless in a sandbox")]
    return max(live, key=lambda a: STRICT[a["class"]]) if live else None


__all__ = ["CLASSES", "IRREGULAR", "KINDS", "KIND_ARGS", "VERBS", "act_class", "imperative", "verb"]
