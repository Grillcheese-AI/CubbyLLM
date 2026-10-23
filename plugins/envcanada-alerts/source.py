"""envcanada-alerts -- a plugin source (order 0028): the weather warnings in effect in Canada, from the national
weather service's Datamart (CAP alerts, Open Government Licence - Canada), read through ``ctx.fetch`` only.

``covers`` (order 0028 review 3) claims a question only when it asks all three of:
- about a warning, watch, advisory or statement (an alert; their French);
- about its status NOW: a yes/no "is there / any / y a-t-il ..." question, or now, tonight, today, in effect ...;
- for a place in Canada (one the gazetteer below or the feed's own regions name), or for no place at all ("here",
  "my area").

What a warning means, how warnings are issued, how many there were, and anywhere outside Canada are not its own.
Weather in general ("what causes freezing rain?", "is there heavy snow in Ottawa?") is not either. A woken source
must claim the question before it fetches (review 2), and ``covers`` itself fetches nothing. ``answer`` reads today's alerts (at most once every
half hour, over the last twelve hours of files; each alert file is fetched once and kept in the plugin's folder) and returns, for the places the question
names, the alerts in effect -- as facts in plain words, dated, the publisher never named in them. A place it cannot find
gets nothing (a refusal), never "no warning": an empty answer is not evidence that the sky is clear.
"""
import datetime as dt
import re
import time
import unicodedata
import xml.etree.ElementTree as ET

BASE = "https://dd.weather.gc.ca/today/alerts/cap/"
CAP = "{urn:oasis:names:tc:emergency:cap:1.2}"
NOW =re.compile(r"\b(now|right now|today|tonight|this (morning|afternoon|evening|week)|in effect|currently|active|"
                 r"maintenant|aujourd'hui|ce soir|cette semaine|en (vigueur|cours)|actuellement)\b", re.I)
KIND = re.compile(r"\b(warn(ings?|s)?|watch(es)?|advisor(y|ies)|statements?|alerts?|avertissements?|veilles?|avis|"
                  r"bulletins?|alertes?)\b", re.I)
ASKS = re.compile(r"^\s*(is|are|any|has|have|should i worry|y a.t.il|est.ce qu|des|un|une)\b", re.I)
NOT_STATUS = re.compile(r"\b(mean|means|meaning|stand for|signifi\w*|veut dire|d[eé]finition|how (are|is|do|does|often|"
                        r"many|much|long)|combien|comment|pourquoi|why|history|historique|last (year|winter|summer|spring|"
                        r"fall|autumn|month|week)|yesterday|hier|derni[eè]re?|in \d{4}|en \d{4}|was there|were there|did)\b",
                        re.I)
PREP = re.compile(r"(?<!\w)(in|for|near|at|around|across|over|à|au|aux|en|dans|pour|près de|pres de)\s+", re.I)
NOT_PLACE = re.compile(r"^(the |le |la |les |l')?(effect|vigueur|cours|the next|the morning|the afternoon|the evening|"
                       r"my (area|region|city|town)|ma (région|region|ville)|mon (coin|quartier)|here|ici|chez moi|"
                       r"this (area|region)|ce coin|the area|the region|la région|la region)\b", re.I)
#: places in Canada a question may name without the feed read yet (folded; the feed's regions are added once read)
CANADA = ("canada", "british columbia", "colombie-britannique", "alberta", "saskatchewan", "manitoba", "ontario", "quebec",
          "new brunswick", "nouveau-brunswick", "nova scotia", "nouvelle-ecosse", "prince edward island",
          "ile-du-prince-edouard", "newfoundland", "labrador", "terre-neuve", "yukon", "northwest territories",
          "territoires du nord-ouest", "nunavut", "the maritimes", "maritimes", "the prairies", "prairies",
          "atlantic canada", "vancouver", "victoria", "kelowna", "kamloops", "prince george", "nanaimo", "whistler",
          "calgary", "edmonton", "red deer", "lethbridge", "banff", "fort mcmurray", "jasper", "regina", "saskatoon",
          "prince albert", "winnipeg", "brandon", "churchill", "toronto", "ottawa", "hamilton", "london", "kitchener",
          "waterloo", "windsor", "sudbury", "thunder bay", "kingston", "barrie", "niagara", "mississauga", "brampton",
          "peterborough", "sault ste. marie", "north bay", "timmins", "guelph", "oshawa", "markham", "montreal",
          "laval", "longueuil", "quebec city", "ville de quebec", "levis", "gatineau", "sherbrooke", "trois-rivieres",
          "saguenay", "chicoutimi", "rimouski", "drummondville", "granby", "saint-jean-sur-richelieu", "rouyn-noranda",
          "val-d'or", "sept-iles", "baie-comeau", "gaspe", "gaspesie", "charlevoix", "estrie", "monteregie",
          "laurentides", "lanaudiere", "outaouais", "mauricie", "abitibi", "cote-nord", "bas-saint-laurent",
          "chaudiere-appalaches", "beauce", "lac-saint-jean", "iles-de-la-madeleine", "fredericton", "moncton",
          "saint john", "bathurst", "halifax", "dartmouth", "sydney", "truro", "cape breton", "charlottetown",
          "summerside", "st. john's", "corner brook", "gander", "happy valley-goose bay", "whitehorse", "yellowknife",
          "inuvik", "iqaluit", "rankin inlet")
REFRESH_S = 1800
HOURS = 12              # alerts are re-issued as updates: the last 12 hours hold every one in effect


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()


class Alerts:
    def __init__(self, ctx) -> None:
        self.ctx, self.alerts, self.read_at = ctx, [], 0.0

    def covers(self, question: str) -> float:
        """0.8 when the question asks about an alert's status now, for a place in Canada or for none; else 0.0."""
        q = " ".join((question or "").split())
        if not KIND.search(q) or NOT_STATUS.search(q) or not (ASKS.search(q) or NOW.search(q)):
            return 0.0
        return 0.8 if self._in_canada(q) else 0.0

    def _in_canada(self, q: str) -> bool:
        """Every place the question names is in Canada (the gazetteer, or a region the feed has named); no place at all
        ("here", "my area", none) is the asker's own, so it counts."""
        known = set(CANADA) | {fold(a["area"]) for a in self.alerts if a.get("area")}
        cuts = [m.end() for m in PREP.finditer(q)]
        for i, start in enumerate(cuts):
            phrase = q[start:(cuts[i + 1] if i + 1 < len(cuts) else len(q))]
            phrase = NOW.split(phrase)[0]                                  # "Vancouver this week" -> "Vancouver"
            phrase = fold(re.sub(r"[?!.,;:]+", " ", phrase)).strip()
            phrase = re.sub(r"\b(in|for|near|at|around|across|over|au|aux|en|dans|pour|pres de)$", "", phrase).strip()
            if not phrase or NOT_PLACE.match(phrase) or not re.search(r"[a-z]", phrase):
                continue
            if not any(re.search(rf"\b{re.escape(k)}\b", phrase) for k in known):
                return False
        return True

    # -- reading the Datamart ----------------------------------------------------------------------------------------
    def _list(self, url: str, pattern: str) -> list[str]:
        code, body = self.ctx.fetch(url)
        return re.findall(pattern, body.decode("utf-8", "replace")) if code == 200 else []

    def refresh(self, now: float | None = None) -> None:
        now = now or time.time()
        if now - self.read_at < REFRESH_S:
            return
        self.read_at, latest = now, {}
        utc = dt.datetime.now(dt.timezone.utc)
        want = {((utc - dt.timedelta(hours=h)).strftime("%Y%m%d"), (utc - dt.timedelta(hours=h)).strftime("%H")) for h in range(HOURS)}
        for day in sorted({d for d, _h in want}):
            for office in self._list(f"{BASE}{day}/", r'href="([A-Z]{4})/"'):
                for hour in sorted(h for d, h in want if d == day):
                    for name in self._list(f"{BASE}{day}/{office}/{hour}/", r'href="([^"]+\.cap)"'):
                        for a in self._read(f"{BASE}{day}/{office}/{hour}/{name}", name):
                            key = (a["event"], a["area"])
                            if key not in latest or a["sent"] > latest[key]["sent"]:
                                latest[key] = a
        stamp = dt.datetime.now(dt.timezone.utc).isoformat()
        self.alerts = sorted((a for a in latest.values() if a["kind"] != "Cancel" and a["expires"] > stamp
                              and in_effect(a["headline"])),
                             key=lambda a: (a["area"], a["event"]))
        self.ctx.log(f"plugin:envcanada-alerts: {len(self.alerts)} alerts in effect")

    def _read(self, url: str, name: str) -> list[dict]:
        body = self.ctx.load(f"cap/{name}")
        if body is None:
            code, body = self.ctx.fetch(url)
            if code != 200:
                return []
            self.ctx.save(f"cap/{name}", body)
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []
        sent, kind = root.findtext(f"{CAP}sent") or "", root.findtext(f"{CAP}msgType") or ""
        out = []
        for info in root.findall(f"{CAP}info"):
            if not (info.findtext(f"{CAP}language") or "").startswith("en"):
                continue
            for area in info.findall(f"{CAP}area"):
                out.append({"event": info.findtext(f"{CAP}event") or "", "headline": info.findtext(f"{CAP}headline") or "",
                            "area": area.findtext(f"{CAP}areaDesc") or "", "sent": sent, "kind": kind,
                            "expires": _utc(info.findtext(f"{CAP}expires"))})
        return out

    # -- answering ---------------------------------------------------------------------------------------------------
    def places(self, question: str) -> list[str]:
        q = fold(question)
        return sorted({a["area"] for a in self.alerts if fold(a["area"]) and re.search(rf"\b{re.escape(fold(a['area']))}\b", q)})

    def answer(self, question: str) -> list[str]:
        self.refresh()
        facts = []
        for place in self.places(question):
            for a in (x for x in self.alerts if x["area"] == place):
                until = a["expires"][:16].replace("T", " ")
                facts.append(f"{said(a['headline'])} is in effect for {place} until {until} UTC.")
        return facts

    def panel(self) -> dict:
        """The board's card: the alerts in effect, by headline, and the first ten."""
        self.refresh()
        counts = {}
        for a in self.alerts:
            counts[a["headline"]] = counts.get(a["headline"], 0) + 1
        return {"count": len(self.alerts), "by_headline": counts,
                "items": [{"area": a["area"], "headline": a["headline"], "until": a["expires"][:16]} for a in self.alerts[:10]]}


def in_effect(headline: str) -> bool:
    """An update that says the alert has ended is not one in effect (found live: "yellow advisory - fog - ended")."""
    h = headline.lower()
    return not re.search(r"\b(ended|terminé|terminée|cancelled|annulé)\b", h)


def said(headline: str) -> str:
    """"yellow warning - wind - in effect" -> "A yellow wind warning"; "special weather statement in effect" ->
    "A special weather statement"."""
    parts = [re.sub(r"\s*in effect$", "", x.strip()) for x in headline.split(" - ")]
    parts = [x for x in parts if x]
    if len(parts) >= 2 and len(parts[0].split()) == 2:
        colour, kind = parts[0].split()
        text = f"{colour} {parts[1]} {kind}"
    else:
        text = " ".join(parts)
    return ("An " if text[:1].lower() in "aeiou" else "A ") + text


def _utc(stamp: str | None) -> str:
    try:
        return dt.datetime.fromisoformat((stamp or "").replace("Z", "+00:00")).astimezone(dt.timezone.utc).isoformat()
    except ValueError:
        return ""
