"""cubbyllm/reasoning/ask_source_namesakes.py -- how the answer reads when the source's rule decided a shared name
(order 0043; the source's side is ``sources.wikidata_namesakes``, the world's twin is ``ask_namesake_answer``).

- **One fits:** the source resolved to it; the answer, then the other namesakes named by their kind, with an offer
  (``ask_namesake_answer.also_line``).
- **Several fit** (two of a kind, or two kinds that both carry the relation): each is walked apart with the source
  pinned to one item, on a scratch first world so one item's facts never land under the other's name (order 0033's
  rule: a value stays tied to its own item), and answered, labelled by one of its own facts (``label_for`` over the
  item's own claims); at most EACH_MAX, then the items are listed and the asker is asked.
- **Namesakes that agree** (item 6): when every answer is the same item (the object the source linked the fact to)
  or the same literal, it is said once, with no offer line.

Behind ``CB_SOURCE_NAMESAKES=1``, as the source's rule is.

Wired: WIRED (``ask_namesakes.after_walk`` and ``ask_namesakes.seed_for``).
"""
from __future__ import annotations

from cubbyllm.reasoning.planner import normalize

from .ask_namesake_answer import EACH_MAX, also_line, label_for

__wiring__ = "WIRED"


def decision(loop, seed: str) -> dict | None:
    src = getattr(loop, "source", None)
    get = getattr(src, "namesake_decision", None)
    return get(seed) if get is not None else None


def scratch_store(loop):
    """A first world for one pinned walk: empty, the read-only worlds behind it as in the loop's own."""
    from cubbyllm.worlds.base import FactStore
    from cubbyllm.worlds.hdc_store import Stores
    world = loop.world
    others = list(getattr(world, "others", []) or []) if hasattr(world, "primary") else []
    fresh = FactStore(name="scratch")
    return Stores(fresh, *others) if others else fresh


def after_walk(loop, rec: dict, seed: str, learn=None) -> bool:
    """The source's decision on `seed`, into the record. True when it shaped the record."""
    d = decision(loop, seed)
    if not d:
        return False
    if len(d["fit"]) == 1:
        if not rec.get("verified"):
            return False                             # the one that fit held no such value: nothing shaped
        others = [{"item": q, "label": lab, "kind": kind} for q, lab, kind in d["others"]][:2]
        rec["namesakes"] = {"rule": "one fits (source)" + (" (label over alias)" if d.get("label") else ""),
                            "item": d["fit"][0], "others": others, "kinds": d["kinds"], "how": d["how"], "cues": d["cues"],
                            "among": d.get("each")}
        line = also_line(others)
        if line:
            rec["namesakes"]["also"] = line          # said once the loop's checks have shaped the reply (`also`)
        return True
    if len(d["fit"]) > 1 and not rec.get("verified") and learn is not None:
        return each(loop, rec, seed, d, learn)
    return False


def _refuse(rec: dict, d: dict) -> None:
    from .ask_namesakes import refuse
    amb = [(q, d["labels"][q], ", ".join(d["kinds"].get(q) or [])) for q in d["fit"]]
    refuse(rec, amb, len(d["labels"]))


def each(loop, rec: dict, seed: str, d: dict, learn) -> bool:
    """Every item that fit walked apart (the source pinned, a scratch first world) and answered, labelled."""
    src = loop.source
    items = list(d["fit"])
    if len(items) > EACH_MAX:
        _refuse(rec, d)                              # more than EACH_MAX: listed, the asker is asked
        return True
    by = {q: src.item_rows(q, d["labels"][q]) for q in items}
    labels = label_for(items, by, d.get("rels") or ()) if all(by.values()) else None
    if not labels:
        labels = {q: f"{d['labels'][q]} ({d['kinds'][q][0]})" if d["kinds"].get(q) else d["labels"][q] for q in items}
        if len(set(labels.values())) < len(items):
            _refuse(rec, d)                          # no fact and no kind tells them apart
            return True
    question = rec.get("question") or ""
    answers = []
    for q in items:
        def walk(q=q):
            store = scratch_store(loop)
            with loop._lock, src.pinned(seed, q, d["labels"][q]):
                lr = learn(store=store)
            times = getattr(store, "times", None) or {}
            if times and getattr(loop.world, "times", None) is not None:
                loop.world.times.update(times)      # the source's dates, where the loop's checks read them
            return lr
        lr = walk()
        ok = bool(lr.result.verified)
        one = {"answer": lr.result.answer if ok else None, "verified": ok, "reason": None if ok else lr.result.reason}
        if hasattr(loop, "_checked"):
            loop._checked(question, lr, one, again=walk)
        ok = bool(one["verified"])
        fact = lr.result.trace[-1].fact if ok and lr.result.trace else None
        linked = list(getattr(src, "links", {}).get(normalize(fact or ""), ())) if fact else []
        answers.append({"item": q, "label": labels[q], "answer": one["answer"], "verified": ok, "reason": one["reason"],
                        "said": str(one.get("reply") or one["answer"]).rstrip(".") if ok else None,
                        "value": linked[0] if len(linked) == 1 else normalize(str(one["answer"] or "")),
                        "trace": [h.fact for h in lr.result.trace if h.fact] if ok else []})
    good = [a for a in answers if a["verified"]]
    if not good:
        _refuse(rec, d)
        return True
    base = {"items": items, "kinds": d["kinds"], "how": d["how"], "cues": d["cues"], "answers": answers}
    if len(good) == len(answers) and len({a["value"] for a in good}) == 1:
        a = next((x for x in good if x["said"] == str(x["answer"]).rstrip(".")), good[0])   # namesakes that agree:
        #   said once, no offer (item 6), in the plainest words any of them was said in
        rec.update(answer=a["answer"], reply=f"{a['said']}.", verified=True, reason=None, trace=a["trace"],
                   namesakes={"rule": "agree (source)", **base})
    else:
        said = "; ".join(f"{a['label']}: {a['said'] if a['verified'] else 'I do not know'}" for a in answers)
        rec.update(answer=said, reply=f"{said}.", verified=True, reason=None, trace=[f for a in good for f in a["trace"]],
                   namesakes={"rule": "several fit (source)", **base})
    rec.pop("clarify", None), rec.pop("candidates", None), rec.pop("refused", None)
    rec["answers"] = answers
    return True


__all__ = ["after_walk", "decision", "each", "scratch_store"]
