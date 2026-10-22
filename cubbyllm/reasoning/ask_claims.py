"""cubbyllm/reasoning/ask_claims.py -- a claim checked against the store (order 0017's cues: forward, backward, the
causal relations), the talk reply, the user's choice among branches and the question that asks for it, and a gated fact
as the record keeps it. Split out of ask.py (order 0031), verbatim; ask.py re-exports every name.

Wired: WIRED (the ask loop).
"""
from __future__ import annotations

import re as _re
from .ask_words import ABSENT_REPLY, _MONTHS, _vnorm, grounded_prose, has_value, say, value_check

__wiring__ = "WIRED"


# ---------------------------------------------------------------- the claim check (H-E10)
# The value check keeps a reply to the one value the host returned: safe, and why Cubby answers like a form.
# The claim check lets a reply say more of the block -- "It began in 1914 in Sarajevo, when Gavrilo Princip
# shot the archduke" -- and still refuses every statement the block does not make: each thing the reply
# names from the block must be tied, in the same sentence, to something the block ties it to (or to the
# event asked about, which a reply may leave as "it"); a sentence that says one thing led to another must
# say it the way round the block does; a year said to begin or end something must be that line's year.
FORWARD_CUES = ("led to", "lead to", "leads to", "leading to", "caused", "causes", "resulted in", "results in",
                "brought about", "brings about", "triggered", "sparked", "gave rise to", "set off", "prompted",
                "provoked", "paved the way for", "contributed to", "precipitated", "opened the way to", "produced")


BACKWARD_CUES = ("because of", "due to", "as a result of", "in response to", "was caused by", "were caused by",
                 "was triggered by", "was sparked by", "was prompted by", "was provoked by", "stemmed from",
                 "came from", "resulted from", "followed from", "was a response to", "were a response to",
                 "in the wake of", "was brought about by", "grew out of")


_CAUSAL_REL = {"caused": 1, "contributed to": 1, "precursor of": 1, "response to": -1}   # +1: e -> v, -1: v -> e


_BEGIN = _re.compile(r"\b(?:began|begin|begins|started|start|starts|broke out|opened|commenced)\b", _re.I)


_END = _re.compile(r"\b(?:ended|end|ends|finished|concluded|closed|was over)\b", _re.I)


_SENT = _re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"'(])")


_PRONOUN = _re.compile(r"\b(?:it|this|that|these|they|its|their|there)\b", _re.I)


def _aliases(value: str) -> list[str]:
    """The ways a reply may write a block string: itself; an ISO date as 'June 28, 1914', '28 June 1914',
    'June 1914' or the year alone (a year is a less precise statement of the same date)."""
    v = str(value)
    m = _re.fullmatch(r"(-?)(\d{4})-(\d{2})(?:-(\d{2}))?", v.strip())
    if not m:
        return [v]
    y, mo = m.group(2).lstrip("0") or "0", int(m.group(3))
    out = [v]
    if 1 <= mo <= 12:
        name = _MONTHS[mo - 1]
        if m.group(4):
            d = str(int(m.group(4)))
            out += [f"{name} {d} {y}", f"{d} {name} {y}"]
        out.append(f"{name} {y}")
    out.append(y)
    return out


def _mentions(text: str, strings) -> list[tuple[int, int, frozenset]]:
    """(start, end, the block strings it may be) for every block string the text writes, longest first, not
    overlapping. One form can be several strings: '1914' is the July Crisis's year, the war's start year and
    the year of the assassination's date -- the sentence decides which, so all are kept."""
    t = f" {_vnorm(text)} "
    forms: dict[str, set] = {}
    for s in strings:
        for a in _aliases(s):
            if _vnorm(a):
                forms.setdefault(_vnorm(a), set()).add(s)
    found: list[tuple[int, int, frozenset]] = []
    for form in sorted(forms, key=len, reverse=True):
        for m in _re.finditer(r"(?<= )" + _re.escape(form) + r"(?= )", t):
            if not any(m.start() < b and a < m.end() for a, b, _ in found):
                found.append((m.start(), m.end(), frozenset(forms[form])))
    return sorted(found, key=lambda f: f[0])


def _cue(sentence: str, strings) -> tuple[int, int] | None:
    """(+1 | -1, position in the normalised sentence) of the first causal cue, if any."""
    t = f" {_vnorm(sentence)} "
    best = None
    for cues, sign in ((BACKWARD_CUES, -1), (FORWARD_CUES, 1)):          # "was caused by" before "caused"
        for c in cues:
            i = t.find(f" {_vnorm(c)} ")
            if i >= 0 and (best is None or i < best[1] or (i == best[1] and sign == -1)):
                best = (sign, i)
    return best


def claim_check(answer: str, lines: list, returned: list[str], entity: str | list[str]) -> tuple[bool, list[str]]:
    """Every statement the reply makes from the block is one the block makes. `lines`: the block as
    (entity, relation, value) triples; `entity`: the event asked about (its name, or all the names it is
    served under). The reply must state a returned value (as `value_check`). -> (ok, what failed)."""
    asked = {_vnorm(e) for e in ([entity] if isinstance(entity, str) else entity) if _vnorm(e)}
    problems: list[str] = []
    if returned and not any(has_value(answer, v) for v in returned):
        problems.append("returned value missing")
    triples = [(str(e), str(r), str(v)) for e, r, v in lines]
    strings = {x for e, _, v in triples for x in (e, v)} | set(asked)
    for sent in _SENT.split(answer or ""):
        ms = _mentions(sent, strings)
        t = f" {_vnorm(sent)} "
        said = {_vnorm(s) for _, _, ss in ms for s in ss}
        present = said | asked                           # the event asked about may stay "it"
        verbs = sorted([(m.start(), "begin") for m in _BEGIN.finditer(t)] + [(m.start(), "end") for m in _END.finditer(t)])
        for at_m, _, ss in ms:
            if any(_vnorm(s) in asked for s in ss):
                continue
            fits = []                                    # the readings of this mention the sentence supports
            for s in ss:
                n = _vnorm(s)
                ties = [(e, r, v) for e, r, v in triples if n in (_vnorm(e), _vnorm(v))
                        and (_vnorm(v) if n == _vnorm(e) else _vnorm(e)) in present]
                rels = {r for e, r, v in ties if _vnorm(v) == n}
                gov = next((kind for p, kind in reversed(verbs) if at_m - 40 <= p < at_m), None)
                if gov == "begin" and rels and rels <= {"end year"}:
                    continue                             # an end year said as a beginning
                if gov == "end" and rels and rels <= {"start year"}:
                    continue
                if ties:
                    fits.append(s)
            if not fits:
                shown = sorted(ss, key=len)[0]
                problems.append(f"'{shown}' is not tied, as said, to what the sentence says")
        cue = _cue(sent, strings)
        if cue:
            sign, at = cue
            events = {_vnorm(e) for e, _, _ in triples} | asked      # what can cause: events, not a year or a place
            left = {_vnorm(s) for a, _, ss in ms if a < at for s in ss if _vnorm(s) in events} or \
                (asked if _PRONOUN.search(t[:at]) else set())
            right = {_vnorm(s) for a, _, ss in ms if a > at for s in ss if _vnorm(s) in events} or \
                (asked if _PRONOUN.search(t[at:]) else set())
            edges = set()
            for e, r, v in triples:
                d = _CAUSAL_REL.get(r)
                if d:
                    a, b = (_vnorm(e), _vnorm(v)) if d == 1 else (_vnorm(v), _vnorm(e))
                    edges.add((a, b))
            for x in left:
                for y in right:
                    if x == y:
                        continue
                    cause, effect = (x, y) if sign == 1 else (y, x)
                    if (cause, effect) not in edges and not ({x, y} <= asked):
                        problems.append(f"the block does not say '{cause}' led to '{effect}'")
    return (not problems), list(dict.fromkeys(problems))


def talk_reply(returned: list[str], draft: str | None, facts: list[str], entity: str,
               others: list[str], lines: list | None = None, check: str = "value") -> tuple[str | None, str]:
    """What the host says for a question over a facts block. The VM returned nothing -> the host says
    ABSENT_REPLY itself (the model is not asked to decline). Otherwise the draft is spoken only if the
    name-and-number guard AND the value check pass -- or, with check="claims" and the block's `lines`,
    the claim check in its place (H-E10). -> (reply or None to refuse, reason)."""
    if not returned:
        return ABSENT_REPLY, "vm_empty"
    if not draft:
        return None, "no_draft"
    ok, bad = grounded_prose(draft, facts, entity)
    if not ok:
        return None, "ungrounded: " + ", ".join(bad[:4])
    if check == "claims" and lines is not None:
        ok, bad = claim_check(draft, lines, returned, entity)
        if not ok:
            return None, "claim: " + "; ".join(bad[:3])
        return draft, "spoken"
    ok, bad = value_check(draft, returned, others, entity)
    if not ok:
        return None, "value: " + ", ".join(bad[:4])
    return draft, "spoken"


def parse_choose(raw) -> dict[int, str]:
    """The asker's picks among offered branches, as the walk takes them: {hop: normalized object}.
    Accepts one {"hop", "object"} or a list of them (the panel sends back what a choice carried);
    anything malformed is dropped -- a choice can only narrow the store's options, so a bad one
    costs nothing but the narrowing."""
    from cubbyllm.reasoning.planner import normalize
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    out: dict[int, str] = {}
    for c in items:
        try:
            out[int(c["hop"])] = normalize(str(c["object"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def branch_clarify(plan, res, prior: dict[int, str] | None = None) -> dict | None:
    """The choices an ambiguous hop's branches offer the asker (graph-of-thought, 2026-09-24).

    Offered only when the ambiguity is MID-chain: which of Marie's two citizenships the question
    means is the asker's to say, and the rest of the chain is still walked and certified after
    they say it (chosen, not invented). A split at the LAST hop is a set of answers, not a choice
    of path -- offering it would let the asker pick the answer and have it come back 'verified'
    (the poisoned-capital case), so there the refusal names the values and asks nothing."""
    branches = getattr(res, "branches", None) or []
    if res.reason != "ambiguous_hop" or not branches or plan is None:
        return None
    first = min(b["via"][0]["hop"] for b in branches if b.get("via"))
    if first >= plan.n_hop - 1:
        return None
    rel = plan.tail.rsplit(" of ", 1)[0] if first == 0 else (plan.relations[first] or "")
    ent = (plan.tail.rsplit(" of ", 1)[1] if (first == 0 and " of " in plan.tail)
           else (res.trace[first - 1].triple.obj if 0 < first <= len(res.trace) and res.trace[first - 1].triple else ""))
    kept = [{"hop": h, "object": o} for h, o in sorted((prior or {}).items()) if h != first]
    groups: dict[str, list[dict]] = {}
    for b in branches:
        groups.setdefault(b["via"][0]["object"], []).append(b)
    choices = []
    for obj, bs in groups.items():
        answers = sorted({b["answer"] for b in bs if b["status"] == "verified"})
        if len(answers) == 1 and all(b["status"] == "verified" for b in bs):
            detail = f"leads to {answers[0]}"
        elif answers:
            detail = "leads to " + " or ".join(answers)
        elif any(b["status"] == "pruned" for b in bs):
            detail = "the store has no way on from " + next(b["stalled"] for b in bs if b["status"] == "pruned")
        else:
            detail = "the VM did not certify this path"
        choices.append({"label": obj, "detail": detail, "item": None,
                        "choose": kept + [{"hop": first, "object": obj}]})
    return {"question": f"{ent} has more than one {say(rel)}. Which one do you mean?",
            "hop": first, "choices": choices}


def _learned(p) -> dict:
    """A gated fact as the record keeps it: the gate's verdict and the provenance the sleep cycle needs to
    make it durable (when it was fetched, the store's snapshot hash before the write, what it clashed with)."""
    return {"fact": p.fact, "status": p.status, "source": p.source, "entity": p.entity,
            "fetched_at": getattr(p, "fetched_at", None), "snapshot_before": getattr(p, "snapshot_before", None),
            "clash": getattr(p, "clash", None)}
