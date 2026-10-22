"""standin/worlds_fiction.py -- fiction worlds, one per work (order 0015): what is true only inside a novel, a play or a
poem, and a router that lets a question reach it only when the question names the work.

Ingest (`ingest`): every work's paragraphs go to one passage index (`fiction/fiction.sqlite`, a `books.BookIndex`;
the work's title in its `title`, its world `fiction:<title>`), and its in-fiction facts to a table beside it. A work's
people and places are not in any reality world's name list, so its names are its own: capitalised spans of one to
three words that recur in the work (`work_names`), and frames (`book_frames.read`) read facts about them -- "Arden was
born in Telmar" is true in that work and nowhere else. A Gutenberg work's title comes from its metadata file or the
catalog; any other's from its file name.

Ask (`FictionWorlds`): `work_named(question)` finds a work's title in the question (a title of two words or more, or
any title in quotes or after "In"); `store(work)` is that work's facts as a store; `ask(question, reality, loop_for)`
sends a question that names a work to a loop over that work's store, with the title taken out of the question, and
every other question to `reality`. A fiction world therefore never answers a question about reality.

Wired: STANDALONE (`worlds_ingest`, `validation/exp_worlds_bench.py`).
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sqlite3

import book_frames as bf

__wiring__ = "STANDALONE"

FACTS = """CREATE TABLE IF NOT EXISTS fiction_facts(work TEXT, subj TEXT, rel TEXT, obj TEXT, book TEXT, sentence TEXT);
CREATE INDEX IF NOT EXISTS fiction_facts_work ON fiction_facts(work);"""
_NAME = re.compile(r"\b[A-Z][a-z'’]+(?:\s(?:of\s|de\s|van\s|von\s)?[A-Z][a-z'’]+){0,2}\b")
_NOT_NAMES = {"The", "He", "She", "It", "They", "We", "I", "You", "His", "Her", "Their", "This", "That", "There", "Then",
              "When", "What", "Where", "Who", "Why", "How", "But", "And", "Or", "If", "In", "On", "At", "As", "So", "Yes",
              "No", "Not", "Now", "Mr", "Mrs", "Miss", "Sir", "Chapter", "Book", "Part", "God", "Oh", "Well"}
_CUES = {w for ws in bf.CUES.values() for w in ws}


def work_names(text: str, top: int = 200, at_least: int = 5) -> list[str]:
    """The names a work repeats: capitalised spans, not a sentence's lone opening word, at least `at_least` times."""
    counts = collections.Counter(m.group(0) for m in _NAME.finditer(text) if m.group(0).split()[0] not in _NOT_NAMES)
    low = collections.Counter(re.findall(r"\b[a-z][a-z'’]+\b", text))
    # a name is rarely written in lowercase in its own work: 'which' and 'emperor' are words, 'Arden' is a name
    return [n for n, k in counts.most_common(top * 3) if k >= at_least and n not in _NOT_NAMES
            and (" " in n or low[n.lower()] < 0.2 * k)][:top]


def title_of(p: pathlib.Path, catalog: dict | None) -> str:
    meta = p.with_name(f"{p.stem}_metadata.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("title") or p.stem
        except (json.JSONDecodeError, OSError):
            pass
    if p.stem.isdigit() and catalog and p.stem in catalog:
        return catalog[p.stem]
    from books import meta as book_meta
    return book_meta(p, p.parent)["title"] or p.stem


KIN = {"father": "father", "mother": "mother", "wife": "spouse", "husband": "spouse", "brother": "sibling",
       "sister": "sibling", "son": "child", "daughter": "child"}
_KIN = re.compile(r"\b(?P<s>[A-Z][a-z'’]+(?:\s[A-Z][a-z'’]+)?)['’]s\s(?P<r>" + "|".join(KIN) +
                  r"),?\s(?P<o>[A-Z][a-z'’]+(?:\s[A-Z][a-z'’]+)?)\b")


def kin_facts(text: str, names: list[str]) -> list[tuple]:
    """What a novel says of its people that frames do not read (they read 'X was born in Y', which fiction rarely
    writes): "Paul's father, Leto" -> (Paul, father, Leto). Both ends must be names the work repeats."""
    known = set(names)
    out = []
    for m in _KIN.finditer(text):
        if m["s"] in known and m["o"] in known and m["s"] != m["o"]:
            line = text[max(0, m.start() - 80):m.end() + 40].replace("\n", " ")
            out.append((m["s"], KIN[m["r"]], m["o"], line))
    return out


def read_facts(text: str, names: list[str]) -> list[tuple]:
    """(subject, relation, object, sentence) for every frame that fires on a work's own names, and its kin."""
    out, seen = [], set()
    for s, r, o, line in kin_facts(text, names):
        if (s, r, o) not in seen:
            seen.add((s, r, o)); out.append((s, r, o, line))
    for para in text.split("\n\n"):
        if not set(re.findall(r"[a-z]+", para.lower())) & _CUES:
            continue
        for s in bf.sentences(para):
            low = s.lower()
            for n in names:
                i = s.find(n)
                # a frame reads its subject, then its verb: a cue within reach, or a capital's apposition -- else no
                # read (each read compiles its frames for that name: 8.5 of 8.9 s on a play, all to find nothing)
                if i < 0 or not (set(re.findall(r"[a-z]+", low[i:i + len(n) + 120])) & _CUES or "capital" in low):
                    continue
                for t, sentence, _when in bf.read(s, n):
                    k = (t.subj, t.rel, t.obj)
                    if k not in seen:
                        seen.add(k); out.append((t.subj, t.rel, t.obj, sentence))
    return out


def ingest(entry, dataset, ctx, files=None, catalog=None) -> dict:
    from books import BookIndex, Mentions, paragraphs
    from worlds_adapters import file_hash, text_files
    db_path = ctx.root / "fiction" / "fiction.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    idx = BookIndex(db_path)
    idx.db.executescript(FACTS)
    added = skipped = left = facts = 0
    for p in files if files is not None else text_files(pathlib.Path(entry.path)):
        digest = file_hash(p)
        if ctx.seen(digest):
            skipped += 1; continue
        title = title_of(p, catalog)
        cleared = ctx.release.cleared(p, {"title": title}) if ctx.release else True
        if ctx.release and ctx.release.only and not cleared:
            left += 1; continue
        text = p.read_bytes().decode("utf-8", errors="replace")
        names = work_names(text)
        idx.add(paragraphs(text), digest, f"{dataset}/{p.name}", {"title": title, "topic": f"fiction:{title}"}, Mentions(names))
        rows = read_facts(text, names)
        idx.db.executemany("INSERT INTO fiction_facts VALUES (?,?,?,?,?,?)", [(title, s, r, o, digest, sent) for s, r, o, sent in rows])
        idx.db.commit()
        ctx.record(digest, dataset, f"fiction:{title}", p, cleared); ctx.reg.commit()
        added, facts = added + 1, facts + len(rows)
    c = idx.counts()
    works = idx.db.execute("SELECT COUNT(DISTINCT work) FROM fiction_facts").fetchone()[0]
    return {"store": str(db_path), "rows": c["paragraphs"], "entities": works, "files_added": added, "files_skipped": skipped,
            "left_out": left, "as_of": None, "facts_read_now": facts}


class FictionWorlds:
    def __init__(self, db_path) -> None:
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.titles = sorted({t for (t,) in self.db.execute("SELECT DISTINCT work FROM fiction_facts")}, key=len, reverse=True)
        self._stores: dict[str, object] = {}

    def work_named(self, question: str) -> str | None:
        q = question.lower()
        for t in self.titles:
            tl = t.lower()
            quoted = re.search(rf"[\"“'‘]{re.escape(tl)}[\"”'’]", q) or re.match(rf"\s*in\s+(?:the\s+\w+\s+)?{re.escape(tl)}\b", q)
            if quoted or (len(tl.split()) >= 2 and re.search(rf"\b{re.escape(tl)}\b", q)):
                return t
        return None

    def facts(self, work: str) -> list[str]:
        return [f"{o} is the {r} of {s}" for s, r, o in self.db.execute(
            "SELECT subj, rel, obj FROM fiction_facts WHERE work = ?", (work,))]

    def store(self, work: str):
        if work not in self._stores:
            from worlds import FactStore
            st = FactStore(name=f"fiction:{work}")
            for (r,) in self.db.execute("SELECT DISTINCT rel FROM fiction_facts WHERE work = ?", (work,)):
                st.index.declare_relation(r)                         # a relation is a label before its facts parse
            for f in self.facts(work):
                st.add(f)
            self._stores[work] = st
        return self._stores[work]

    def strip(self, question: str, work: str) -> str:
        """The question without its work: 'In Dune, where was Paul born?' -> 'Where was Paul born?'."""
        q = re.sub(rf"^\s*in\s+(?:the\s+\w+\s+)?[\"“'‘]?{re.escape(work)}[\"”'’]?\s*,?\s*", "", question, flags=re.I)
        q = re.sub(rf"\s*,?\s*in\s+[\"“'‘]?{re.escape(work)}[\"”'’]?\s*(?=\?|$)", "", q, flags=re.I)
        return q[:1].upper() + q[1:]

    def ask(self, question: str, reality, loop_for) -> dict:
        work = self.work_named(question)
        if work is None:
            return {**reality.ask(question), "world": "reality"}
        rec = loop_for(self.store(work)).ask(self.strip(question, work))
        return {**rec, "world": f"fiction:{work}"}
