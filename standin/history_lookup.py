"""The host's lookup over the history graph (standin/history_graph.py): a question in plain words -> the kind
of ask, the event it is about, and the facts block the graph gives for it -- the lines the talk adapter reads
and the values the host returns, which `ask.talk_reply` reads the draft against.

Wired: STANDALONE (read by validation/exp_e8_history_retrieval.py; next, the ask loop's history world).

No model. The kind is read off the question (KINDS: what led to / what came of / when / where / who / what would
change if ... had not happened). The event is found by its names: every name it was read under is indexed by
its words, and an event scores by the IDF-weighted share of its name the question covers, times the root of the
share of the question's words the name covers -- so a question that holds a name word for word scores its full
name coverage, and a reworded one scores by its rarest shared words. A year in the question agreeing with the
event's dates lifts it, a year against them sinks it. The graph holds some events twice, read from two books
under two names ("Sack of Rome by Gauls", "Sack of Rome by Brennus' Gauls"); candidates that are readings of the
top one (`same_event`: overlapping dates, most words shared, no ordinal telling them apart) are served together as
one event. The best event is used only when the match is clear: a score over `min_score` and ahead of the next
distinct event by `margin` -- or, when the events that close would all return the same answer, any of them.
Otherwise the host asks which one was meant (the candidates) instead of guessing: the host's checks read the
draft against the block it retrieved, and a block for the wrong event passes them. (The kind patterns, the word
normalisation and the same-event rule were revised on H-E8's dev split, before its test split was run.)

    lk = HistoryLookup(HistoryGraph.load(path))
    hit = lk.ask("What led to the Battle of Vienna in 1683?")
    hit.status ('clear' | 'ask' | 'none' | 'unparsed'), hit.kind, hit.event, hit.lines, hit.returned, hit.others
"""
from __future__ import annotations

import collections
import math
import re
import time
import unicodedata
from array import array
from dataclasses import dataclass, field

from history_graph import FORWARD, HistoryGraph, key, overlap, year_text

__wiring__ = "STANDALONE"

MIN_SCORE = 0.6            # H-E8's dev split: 0.5 -> 0.6 kept every right match and dropped a third of the wrong
MARGIN = 0.1               # ones (reworded: right 71.1% both, clear-but-wrong 6 -> 4); never tuned on the test split
FACT_MAX_DF = 20_000       # the facts index leaves out words this common ('war', 'battle', 'new', 'york')
FACT_MIN = 0.6             # a description read by its facts: this share of its words covered (year factor in) ...
FACT_MARGIN = 0.2          # ... and this far ahead of the next distinct event (H-E8 dev, second round)
MAX_LINKS = 6              # cause/effect lines a block shows
MAX_DOWN = 8               # downstream lines a block shows

_END = r"\s*[?.!]*\s*$"
_X = r"(?P<x>.*?\w.*?)"
_NOT = r"(?:had\s+not|hadn'?t|had\s+never|did\s+not|didn'?t|never|not)"


_PRON = frozenset("it they this that these those he she him them its their".split())


def _k(p: str) -> re.Pattern:
    return re.compile(p, re.I)


# the kind of ask, first match wins; `x` is the phrase naming the event. Read by the question's shape: an "if ...
# had not" is a branch whatever it opens with; then what it opens with (why / when / where / who) or the noun or
# verb it turns on (led to, consequences, what year, which place). Widened on H-E8's dev split only.
KINDS = [
    ("downstream", _k(r"^\s*(?:what|how|in\s+what\s+ways?)\b.*?\b(?:would|might|could)\b.*?\b(?:if|had)\s+" + _X + r"\s+"
                      + _NOT + r"\b.*$")),
    ("downstream", _k(r"^\s*(?:what\s+if|if|had|suppose|imagine)\s+" + _X + r"\s+" + _NOT + r"\b.*$")),
    ("downstream", _k(r"^\s*(?:what|how)\s+(?:would|might|could)\b.*?\bwithout\s+" + _X + _END)),
    ("downstream", _k(r"^\s*how\s+(?:would|might|could)\s+" + _X + r"\s+have\s+(?:altered|changed|shaped|affected|"
                      r"influenced|impacted|transformed|redirected)\b.*$")),
    ("effect", _k(r"^\s*what\s+(?:did|does|do)\s+" + _X + r"\s+(?:lead\s+to|led\s+to|cause|result\s+in|bring\s+about|"
                  r"trigger|set\s+off|produce|start|mean\s+for)\b.*$")),
    ("effect", _k(r"^\s*what\s+(?:happened|came|resulted|followed|arose|changed)\s+(?:next\s+)?(?:of|from|after|because\s+of|"
                  r"as\s+a\s+result\s+of|in\s+the\s+wake\s+of|following)\s+" + _X + _END)),
    ("effect", _k(r"^\s*what\s+(?:were|was|are|is)\s+(?:the\s+)?(?:\w+\s+){0,2}?(?:consequences?|effects?|results?|outcomes?|"
                  r"aftermath|impacts?|repercussions?|legacy|significance)\s+(?:of|from)\s+" + _X + _END)),
    ("effect", _k(r"^\s*(?:how|in\s+what\s+ways?)\s+did\s+" + _X + r"\s+(?:affect|change|shape|influence|impact|alter|"
                  r"lead\s+to)\b.*$")),
    ("cause", _k(r"^\s*why\b\s*" + _X + _END)),
    ("cause", _k(r"^\s*what\s+(?:\w+\s+){0,3}?(?:led\s+to|lead\s+to|leads\s+to|caused|causes|brought\s+about|triggered|"
                 r"provoked|sparked|prompted|precipitated|gave\s+rise\s+to|set\s+off|started|motivated|drove|pushed)\s+"
                 + _X + _END)),
    ("cause", _k(r"^\s*what\s+(?:were|was|are|is)\s+(?:the\s+)?(?:\w+\s+){0,2}?(?:causes?|reasons?|origins?|roots?|triggers?|"
                 r"background|motives?|motivations?)\s+(?:of|for|behind|to)\s+" + _X + _END)),
    ("cause", _k(r"^\s*what\s+(?:\w+\s+){0,2}?(?:led|drove|pushed|inspired|compelled|forced|encouraged|persuaded|induced|"
                 r"moved|prompted|motivated|caused)\s+(?P<x>.+?\s+to\s+\w.*?)" + _END)),   # "what led Henry II to make ..."
    ("cause", _k(r"^\s*how\s+did\s+" + _X + r"\s+(?:come\s+about|come\s+to\s+(?:happen|pass)|start|begin|arise)" + _END)),
    ("cause", _k(r"^\s*who\s+or\s+what\s+(?:caused|led\s+to|brought\s+about)\s+" + _X + _END)),
    ("when", _k(r"^\s*when\b\s*" + _X + _END)),
    ("when", _k(r"^\s*(?:on|in|at|during|by)?\s*(?:what|which)\s+(?:exact\s+)?(?:date|day|year|month|time|period|century|"
                r"decade)\b\s*" + _X + _END)),
    ("when", _k(r"^\s*what\s+(?:is|was)\s+(?:the\s+)?(?:exact\s+)?(?:date|day|year)\s+(?:of|when|that)\s+" + _X + _END)),
    ("when", _k(r"^\s*how\s+long\s+ago\s+" + _X + _END)),
    ("where", _k(r"^\s*where\b\s*" + _X + _END)),
    ("where", _k(r"^\s*(?:in|at|on)?\s*(?:what|which)\s+(?:(?!is\b|was\b|were\b|are\b)\w+\s+){0,2}?(?:places?|city|cities|"
                 r"country|regions?|locations?|"
                 r"towns?|areas?|state|provinces?|sites?|villages?|ports?|islands?)\b\s*" + _X + _END)),   # not "countries",
    # "states": which countries / states took part is a who
    ("who", _k(r"^\s*who\b\s*" + _X + _END)),
    ("who", _k(r"^\s*(?:which|what)\s+(?:\w+\s+){0,2}?(?:people|groups?|parties|countries|nations|forces|sides|figures|"
               r"individuals|leaders|players|organizations|persons|participants|armies|states|teams|companies|rulers)\b\s*"
               + _X + _END)),
]

# words that name no event: question words, the kinds' own verbs, articles and prepositions
STOP = frozenset("""the a an of in on at to for from by with and or as its it his her their them they this that these those is
was were be been being are what which who whom whose when where why how did does do had has have would could should will not never
if happen happened happens occur occurred take took taken place part led lead leads cause caused causes consequence consequences
result results resulted effect effects came come about after before during against into over between participant participants
involved date year day month time event events history change changed different differ exactly exact main major what's
there then than so such also only just any all some other more most very ever might may must can could much many specific
specifically bc bce ad ce around approximately circa century centuries decade early late mid later earlier begin began begun
beginning today course differently unfolded unfold altered alter arisen arise develop developed developing happening""".split())
ORDINAL = frozenset("first second third fourth fifth sixth seventh eighth ninth tenth last elder younger".split())

_YEAR = re.compile(r"(?<![\d,.])(\d{3,4})(?![\d,])(?:\s*(b\.?\s?c\.?e?\.?|bce))?", re.I)
_SUFFIX = ("izations", "ization", "ations", "ation", "ments", "ment", "ings", "ing", "ions", "ion", "ious", "ous",
           "ists", "ist", "ians", "ian", "ers", "er", "ed", "es", "ia", "s")   # -ian/-ia: 'Thracian'/'Thrace',
# 'Athenian'/'Athens', 'Macedonian'/'Macedonia', 'Persian'/'Persia' (H-E8 dev, second round)


def _stem(t: str) -> str:
    """A light suffix strip, the same on names and questions, then a final 'e' and a doubled last consonant:
    'rebellion'/'rebelled'/'rebel', 'visited'/'visit', 'persecution'/'persecute', 'announcement'/'announced',
    'religious'/'religion' meet (H-E8 dev, second round: the first strip left 'rebell' beside 'rebel' and
    'persecute' beside 'persecut')."""
    for s in _SUFFIX:
        if len(t) - len(s) >= 4 and t.endswith(s) and not (s == "s" and t.endswith("ss")):
            t = t[:-len(s)]
            break
    if len(t) >= 5 and t.endswith("e"):
        t = t[:-1]
    if len(t) >= 5 and t[-1] == t[-2] and t[-1] not in "aeiousl" or len(t) >= 6 and t.endswith("ll"):
        t = t[:-1]
    return t


def _fold(text: str) -> str:
    """Accents off: 'Khwārazm' and 'Khwarazm', 'Teotihuacán' and 'Teotihuacan' are one word."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    """The content words of a name or a question: key()'s normal form with accents off, stop words and numbers out
    (a year is matched as a date, not a word) -- but a number of one or two digits that stands before a word is
    part of a name ('2 Maccabees', '4 Maccabees') -- a suffix off."""
    toks = key(_fold(text)).split()
    out = []
    for i, t in enumerate(toks):
        if t.isdigit():
            if len(t) <= 2 and i + 1 < len(toks) and toks[i + 1].isalpha() and toks[i + 1] not in STOP:
                out.append(t)
        elif t not in STOP and len(t) > 1:
            out.append(_stem(t))
    return out


def years(text: str) -> list[int]:
    return [-int(m.group(1)) if m.group(2) else int(m.group(1)) for m in _YEAR.finditer(text)]


def _quotable(w: dict | None, context: bool = False) -> bool:
    return bool(w) and w.get("basis", "stated") in (("stated", "context") if context else ("stated",))


def own_lines(g: HistoryGraph, i: str, date=True, where=True, who=True) -> list[list[str]]:
    """The event's own lines -- the same lines the talk data shows (build_history_sft.Blocks.own)."""
    ev, out = g.events[i], []
    if date and _quotable(ev.when, context=True):
        w = ev.when
        if w.get("day"):
            out.append([ev.name, "date", w["day"]])
        elif w["y0"] == w["y1"]:
            out.append([ev.name, "year", year_text(w["y0"])])
        else:
            out += [[ev.name, "start year", year_text(w["y0"])], [ev.name, "end year", year_text(w["y1"])]]
    if where:
        out += [[ev.name, "location", x] for x in ev.where[:2]]
    if who:
        out += [[ev.name, "participant", x] for x in ev.who[:4]]
    return out


def serve(g: HistoryGraph, i: str, kind: str) -> tuple[list[list[str]], list[str], list[str]]:
    """(lines, returned, others): the facts block the host shows for `kind` asked about event i, the values it
    returns (empty: the graph does not say, and the host answers "The facts don't say." itself), and every
    other name and value in the block (what `ask.value_check` refuses a draft for naming)."""
    ev, name = g.events[i], g.events[i].name
    lines = own_lines(g, i)
    returned: list[str] = []
    if kind in ("cause", "effect"):
        if kind == "cause":
            links = [(a, r, i) for a, r in g.into(i, FORWARD)] + [(i, "response to", b) for r, b in g.out(i, ("response to",))]
            other = [a if b == i else b for a, _, b in links]
        else:
            links = [(i, r, b) for r, b in g.out(i, FORWARD)] + [(a, "response to", i) for a, r in g.into(i, ("response to",))]
            other = [b if a == i else a for a, _, b in links]
        for (a, r, b), o in list(zip(links, other))[:MAX_LINKS]:
            lines.append([g.events[a].name, r, g.events[b].name])
            lines += own_lines(g, o, where=False, who=False)
            returned.append(g.events[o].name)
    elif kind == "when":
        w = ev.when
        if _quotable(w):
            returned = [w["day"]] if w.get("day") else sorted({year_text(w["y0"]), year_text(w["y1"])})
    elif kind == "where":
        returned = list(ev.where[:2])
    elif kind == "who":
        returned = list(ev.who[:4])
    elif kind == "downstream":
        down = g.downstream(i, limit=64)
        seen, n = set(), 0
        for cur in [i] + down:
            for r, b in g.out(cur, FORWARD):
                if b in down and (cur, r, b) not in seen and n < MAX_DOWN:
                    seen.add((cur, r, b)); n += 1
                    lines.append([g.events[cur].name, r, g.events[b].name]); returned.append(g.events[b].name)
            for a, r in g.into(cur, ("response to", "part of")):
                if a in down and (a, r, cur) not in seen and n < MAX_DOWN:
                    seen.add((a, r, cur)); n += 1
                    lines.append([g.events[a].name, r, g.events[cur].name]); returned.append(g.events[a].name)
        returned = list(dict.fromkeys(returned))
    keep = {name, *returned}
    others = list(dict.fromkeys(x for e, _, v in lines for x in (e, v) if x not in keep))
    return lines, returned, others


# the ask the talk adapter reads over the block: the kind asked, about the event the host found, by its name -- the
# asker's own words stay the asker's, the adapter gets the program they came to (the training's first template)
CANON = {"cause": "What led to {x}?", "effect": "What did {x} lead to?", "when": "When did {x} happen?",
         "where": "Where did {x} take place?", "who": "Who took part in {x}?",
         "downstream": "What would change if {x} had not happened?"}
_THE = re.compile(r"^[A-Z][a-z]+ (?:of|at|on|in|to|from|for|against|with|by|between|over)\b")


def the(name: str) -> str:
    """'Fall of Richmond' -> 'the Fall of Richmond' (build_history_sft.the)."""
    return f"the {name}" if _THE.match(name) else name


def canonical(kind: str, name: str) -> str:
    return CANON[kind].format(x=the(name))


def parse_kind(question: str) -> tuple[str | None, str]:
    """(kind, the phrase naming the event) -- (None, question) when no kind reads off it. A question that sets the
    scene first ("In 1208, a ruler ... ; where did this conflict occur?") is read by its last clause, and the whole
    question names the event."""
    for kind, pat in KINDS:
        m = pat.match(question)
        if m:
            x = m.group("x")
            after = question[m.end("x"):]
            if not [t for t in key(x).split() if t not in _PRON]:  # "how might Alexander's campaign have changed if it
                x = question                                       # never happened": the event is in the rest
            elif kind == "downstream" and re.match(r"\s+" + _NOT + r"\b", after, re.I):
                x += re.split(r"[,;?]", after, maxsplit=1)[0]      # "if Hippias had never disarmed the citizen
            return kind, x                                         # militia": the event is in the verb too
    clauses = [c for c in re.split(r"[;:,]\s*", question) if c.strip()]
    for c in reversed(clauses[1:]):
        for kind, pat in KINDS:
            if pat.match(c):
                return kind, question
    return None, question


def _markers(name: str) -> set:
    """Ordinals and numerals in a name: 'First'/'Second', 'II', '1st' -- two names that differ here are two events."""
    return {t for t in key(name).split() if t in ORDINAL or re.fullmatch(r"[ivxl]+|\d+(?:st|nd|rd|th)?", t)}


@dataclass
class Hit:
    question: str
    kind: str | None
    phrase: str
    status: str                                   # clear | ask | none | unparsed
    event: str | None = None
    name: str = ""
    score: float = 0.0
    members: list = field(default_factory=list)      # the event ids served as one event (the top one's other readings)
    canonical: str = ""                              # the ask the adapter reads: CANON[kind] about the found event
    candidates: list = field(default_factory=list)   # [{"event", "name", "score"}], best first
    offered: list = field(default_factory=list)      # status "ask": the candidates the host asks between (the close ones)
    via: str = "name"                                # found by a name ("name") or by the description's facts ("facts")
    lines: list = field(default_factory=list)
    returned: list = field(default_factory=list)
    others: list = field(default_factory=list)
    ms: float = 0.0


class HistoryLookup:
    def __init__(self, g: HistoryGraph, min_score: float = MIN_SCORE, margin: float = MARGIN, max_df: int = 100_000,
                 max_lines: int = 14):
        self.g, self.min_score, self.margin, self.max_df = g, float(min_score), float(margin), int(max_df)
        self.max_lines = int(max_lines)
        self._name_years: dict[str, frozenset] = {}
        self.ids = list(g.events)
        self.name_event: list[int] = []            # name index -> event index
        self.name_words: list[tuple[str, ...]] = []
        self.name_key: list[str] = []
        post = collections.defaultdict(list)
        for ei, eid in enumerate(self.ids):
            for nm in dict.fromkeys(g.events[eid].names or [g.events[eid].name]):
                ws = tuple(dict.fromkeys(words(nm)))
                if not ws:
                    continue
                ni = len(self.name_event)
                self.name_event.append(ei)
                self.name_words.append(ws)
                self.name_key.append(key(nm))
                for w in ws:
                    post[w].append(ni)
        n = len(self.name_event)
        self.idf = {w: math.log(1 + n / (1 + len(p))) for w, p in post.items()}
        self.post = dict(post)
        self.name_mass = [sum(self.idf[w] for w in ws) for ws in self.name_words]

    def _year_factor(self, eid: str, qyears: list[int]) -> float:
        if not qyears:
            return 1.0
        ev = self.g.events[eid]
        named = self._name_years.get(eid)
        if named is None:
            named = self._name_years[eid] = frozenset(y for n in (ev.name, *ev.names) for y in years(n))
        if named.intersection(qyears):
            return 1.2                  # the year is in the event's name ('... Largest Since 1933', dated 1938): it agrees
        w = ev.when
        if not w:
            return 0.9
        return 1.2 if any(w["y0"] - 1 <= y <= w["y1"] + 1 for y in qyears) else 0.5

    def _facts_index(self):
        """Every event as the words of its names, places and parties (built on first use): the index a description
        is read against when no name fits -- "the 1601 clash near Cork involving Spain, the English and Hugh
        O'Neill" is the Battle of Kinsale by its facts, not by its name."""
        if getattr(self, "_fpost", None) is not None:
            return
        post = collections.defaultdict(lambda: array("i"))
        for ei, eid in enumerate(self.ids):
            ev = self.g.events[eid]
            ws = set()
            for text in (*(ev.names or [ev.name]), *ev.where, *ev.who):
                ws.update(words(text))
            for w in ws:
                post[w].append(ei)
        n = len(self.ids)
        self._fpost = {w: p for w, p in post.items() if len(p) <= FACT_MAX_DF}
        self._fidf = {w: math.log(1 + n / (1 + len(p))) for w, p in self._fpost.items()}

    def find_facts(self, phrase: str, question: str = "", k: int = 5) -> list[dict]:
        """Events ranked by how much of the phrase their names, places and parties cover (IDF-weighted, words too
        common to tell events apart left out), times the year factor: [{event, name, score, shared}]."""
        self._facts_index()
        q = list(dict.fromkeys(w for w in words(phrase) if w in self._fidf))
        if len(q) < 2:
            return []
        qmass = sum(self._fidf[w] for w in q)
        hit, shared = collections.defaultdict(float), collections.defaultdict(int)
        for w in q:
            for ei in self._fpost[w]:
                hit[ei] += self._fidf[w]
                shared[ei] += 1
        qyears = years(question or phrase)
        scored = sorted(((m / qmass * self._year_factor(self.ids[ei], qyears), ei) for ei, m in hit.items()
                         if shared[ei] >= 2), key=lambda t: (-t[0], self.ids[t[1]]))[:k]
        return [{"event": self.ids[ei], "name": self.g.events[self.ids[ei]].name, "score": round(s, 4),
                 "shared": shared[ei]} for s, ei in scored]

    def find(self, phrase: str, question: str = "", k: int = 5) -> list[dict]:
        """Events ranked for `phrase` (the question's years read from `question`): [{event, name, score}]. A name
        counts only when the question covers at least 40% of it (IDF-weighted)."""
        q = list(dict.fromkeys(w for w in words(phrase) if w in self.idf))
        if not q:
            return []
        qmass = sum(self.idf[w] for w in q)
        rare = [w for w in q if len(self.post[w]) <= self.max_df] or q
        hit = collections.defaultdict(float)
        for w in rare:
            for ni in self.post[w]:
                hit[ni] += self.idf[w]
        common = [w for w in q if w not in rare]
        best: dict[int, float] = {}
        qkey = f" {key(question or phrase)} "
        qyears = years(question or phrase)
        for ni, m in hit.items():
            m += sum(self.idf[w] for w in common if w in self.name_words[ni])
            cov = m / self.name_mass[ni]
            if cov < 0.4:
                continue
            s = cov * math.sqrt(m / qmass)
            if len(self.name_key[ni].split()) >= 2 and f" {self.name_key[ni]} " in qkey:
                s *= 1.15                                  # the name as a phrase, not its words scattered
            ei = self.name_event[ni]
            if s > best.get(ei, 0.0):
                best[ei] = s
        scored = []
        for ei, s in best.items():
            eid = self.ids[ei]
            ev = self.g.events[eid]
            s *= self._year_factor(eid, qyears) * (1 + 0.03 * min(math.log1p(len(ev.sources)), 3.0))
            scored.append((s, eid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [{"event": eid, "name": self.g.events[eid].name, "score": round(s, 4)} for s, eid in scored[:k]]

    def same_event(self, a: str, b: str) -> bool:
        """Two readings of one event the graph did not merge (two books naming it differently): dates that overlap,
        names sharing two thirds of the shorter one's words (one of them not a common word), no ordinal or numeral
        telling them apart. 'Sack of Rome by Gauls' / "Sack of Rome by Brennus' Gauls"; not 'Reign of Darius' /
        'Revolts against Darius I', not the First and the Second Battle of Bull Run."""
        ea, eb = self.g.events[a], self.g.events[b]
        if not ea.when or not eb.when or not overlap(ea.when, eb.when):
            return False
        ma, mb = _markers(ea.name), _markers(eb.name)
        if ma and mb and ma != mb:
            return False
        wa, wb = set(words(ea.name)), set(words(eb.name))
        shared = wa & wb
        if not shared or len(shared) < 0.66 * min(len(wa), len(wb)):
            return False
        return any(len(self.post.get(w, ())) <= 5000 for w in shared)

    def serve_all(self, members: list[str], kind: str) -> tuple[list, list, list]:
        """One block for the readings of one event: their lines and returned values together, up to `max_lines`."""
        lines, returned, names = [], [], {self.g.events[m].name for m in members}
        for m in members:
            ls, rs, _ = serve(self.g, m, kind)
            for l in ls:
                if l not in lines and len(lines) < self.max_lines:
                    lines.append(l)
            shown = {x for l in lines for x in (l[0], l[2])}
            returned += [r for r in rs if r in shown and r not in returned]
        keep = names | set(returned)
        # a place or a party inside the event's own name ('Fire at Long Branch' / Long Branch) is the event being
        # named, not another entity the draft must not mention
        in_name = " ".join(f" {key(n)} " for n in names)
        others = list(dict.fromkeys(x for e, _, v in lines for x in (e, v)
                                    if x not in keep and not (key(x) and f" {key(x)} " in in_name)))
        return lines, returned, others

    def ask(self, question: str, k: int = 5) -> Hit:
        kind, phrase = parse_kind(question)
        return self.resolve(question, kind, phrase, k)

    def resolve(self, question: str, kind: str | None, phrase: str, k: int = 5) -> Hit:
        """The event for an ask whose kind is already read (`ask`, or the context graph with its own reading)."""
        t0 = time.perf_counter()
        cands = self.find(phrase, question, 10)
        hit = Hit(question, kind, phrase, "none", candidates=cands[:k])
        if cands:
            top = cands[0]
            hit.members = [top["event"]] + [c["event"] for c in cands[1:] if self.same_event(top["event"], c["event"])]
        if kind is None:
            hit.status = "unparsed"
        elif cands and cands[0]["score"] >= self.min_score:
            top = cands[0]
            rest = [c for c in cands[1:] if c["event"] not in hit.members]
            close = [c for c in rest if top["score"] - c["score"] < self.margin]
            if close:
                mine = tuple(sorted(self.serve_all(hit.members, kind)[1]))
                hit.status = "clear" if all(tuple(sorted(serve(self.g, c["event"], kind)[1])) == mine for c in close) else "ask"
                if hit.status == "ask":                    # the top one's readings, the close ones and theirs -- not a
                    near = [top] + close                   # far fifth candidate a year in the reply could also fit
                    hit.offered = [c for c in cands if c["event"] in hit.members or any(
                        c["event"] == x["event"] or self.same_event(x["event"], c["event"]) for x in near)]
            else:
                hit.status = "clear"
            if hit.status == "clear":
                self.fill(hit, top["event"], kind, hit.members)
                hit.score = top["score"]
        if hit.status == "none" and kind is not None:
            # no name fits: read the description against the events' places and parties, and use the best one only
            # when it stands well clear of the next (H-E8 dev, second round: the right tops scored 0.68-1.01 with the
            # next distinct event 0.28-0.61 below; every wrong top was within 0.1 of its next, and under 0.55)
            fc = self.find_facts(phrase, question, 10)
            if fc and fc[0]["score"] >= FACT_MIN:
                top = fc[0]
                mem = [top["event"]] + [c["event"] for c in fc[1:] if self.same_event(top["event"], c["event"])]
                nxt = next((c["score"] for c in fc[1:] if c["event"] not in mem), 0.0)
                if top["score"] - nxt >= FACT_MARGIN:
                    hit.status, hit.via, hit.score = "clear", "facts", top["score"]
                    hit.candidates = fc[:k]
                    self.fill(hit, top["event"], kind, mem)
        hit.ms = (time.perf_counter() - t0) * 1000
        return hit

    def fill(self, hit: Hit, event: str, kind: str, members: list[str] | None = None) -> Hit:
        """Serve `kind` about `event` (and its other readings) into the hit: block, returned, others, the host's ask."""
        hit.event, hit.name, hit.members = event, self.g.events[event].name, list(members or [event])
        hit.lines, hit.returned, hit.others = self.serve_all(hit.members, kind)
        hit.canonical = canonical(kind, hit.name)
        return hit
