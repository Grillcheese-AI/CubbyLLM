import urllib.request, urllib.robotparser as rp, gzip, json, time, xml.etree.ElementTree as ET
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
RDF = "{http://purl.org/rss/1.0/}"
AI = {"anthropic-ai", "claudebot", "claude-web", "claude", "gptbot", "ccbot", "google-extended", "chatgpt-user",
      "perplexitybot", "perplexity-user", "cohere-ai", "bytespider", "oai-searchbot"}
# (region, language, outlet, kind, url)
C = [
 ("Europe", "en", "BBC World", "public", "https://feeds.bbci.co.uk/news/world/rss.xml"),
 ("Europe", "en", "The Guardian World", "private", "https://www.theguardian.com/world/rss"),
 ("Europe", "en", "Deutsche Welle (EN)", "public", "https://rss.dw.com/rdf/rss-en-all"),
 ("Europe", "fr", "France 24 (FR)", "public", "https://www.france24.com/fr/rss"),
 ("Europe", "en", "France 24 (EN)", "public", "https://www.france24.com/en/rss"),
 ("Europe", "fr", "RFI (FR)", "public", "https://www.rfi.fr/fr/rss"),
 ("Europe", "fr", "Le Monde (une)", "private", "https://www.lemonde.fr/rss/une.xml"),
 ("Europe", "en", "Euronews", "private", "https://www.euronews.com/rss"),
 ("Europe", "es", "El País (portada)", "private", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"),
 ("Europe", "de", "Tagesschau", "public", "https://www.tagesschau.de/xml/rss2/"),
 ("Europe", "en", "Kyiv Independent", "private", "https://kyivindependent.com/news-archive/rss/"),
 ("Europe", "en", "Meduza (EN)", "private", "https://meduza.io/rss/en/all"),
 ("Europe", "en", "TASS (EN)", "state", "https://tass.com/rss/v2.xml"),
 ("Middle East", "en", "Al Jazeera (EN)", "state-funded", "https://www.aljazeera.com/xml/rss/all.xml"),
 ("Middle East", "en", "Times of Israel", "private", "https://www.timesofisrael.com/feed/"),
 ("Middle East", "en", "Middle East Eye", "private", "https://www.middleeasteye.net/rss"),
 ("Middle East", "en", "Anadolu Agency (EN)", "state", "https://www.aa.com.tr/en/rss/default?cat=guncel"),
 ("Middle East", "en", "Arab News", "private", "https://www.arabnews.com/rss.xml"),
 ("Africa", "en", "Africanews (EN)", "private", "https://www.africanews.com/feed/rss"),
 ("Africa", "en", "AllAfrica (latest)", "aggregator", "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf"),
 ("Africa", "en", "Premium Times (Nigeria)", "private", "https://www.premiumtimesng.com/feed"),
 ("Africa", "en", "Daily Maverick (South Africa)", "private", "https://www.dailymaverick.co.za/dmrss/"),
 ("Africa", "fr", "Jeune Afrique", "private", "https://www.jeuneafrique.com/feed/"),
 ("Asia", "en", "NHK World (EN)", "public", "https://www3.nhk.or.jp/nhkworld/en/news/rss.xml"),
 ("Asia", "en", "Japan Times", "private", "https://www.japantimes.co.jp/feed/"),
 ("Asia", "en", "Yonhap (EN)", "state-funded", "https://en.yna.co.kr/RSS/news.xml"),
 ("Asia", "en", "South China Morning Post (world)", "private", "https://www.scmp.com/rss/91/feed"),
 ("Asia", "en", "Xinhua (EN)", "state", "https://english.news.cn/rss/worldrss.xml"),
 ("Asia", "en", "The Hindu (international)", "private", "https://www.thehindu.com/news/international/feeder/default.rss"),
 ("Asia", "en", "Dawn (Pakistan)", "private", "https://www.dawn.com/feeds/home"),
 ("Asia", "en", "CNA (Singapore)", "state-funded", "https://www.channelnewsasia.com/rssfeeds/8395986"),
 ("Asia", "en", "The Diplomat", "private", "https://thediplomat.com/feed/"),
 ("Oceania", "en", "ABC News (Australia)", "public", "https://www.abc.net.au/news/feed/51120/rss.xml"),
 ("Oceania", "en", "RNZ World", "public", "https://www.rnz.co.nz/rss/world.xml"),
 ("Americas", "en", "NPR News", "public", "https://feeds.npr.org/1001/rss.xml"),
 ("Americas", "en", "CBS News World", "private", "https://www.cbsnews.com/latest/rss/world"),
 ("Americas", "es", "Infobae (América)", "private", "https://www.infobae.com/arc/outboundfeeds/rss/?outputType=xml"),
 ("Americas", "pt", "Folha de S.Paulo (mundo)", "private", "https://feeds.folha.uol.com.br/mundo/rss091.xml"),
 ("Americas", "en", "MercoPress", "private", "https://en.mercopress.com/rss/"),
 ("Americas", "en", "Buenos Aires Times", "private", "https://www.batimes.com.ar/feed"),
 ("Americas", "en", "Global News (Canada)", "private", "https://globalnews.ca/feed/"),
 ("World", "en", "Reuters (no public feed expected)", "private", "https://www.reutersagency.com/feed/"),
 ("World", "en", "Associated Press via Google News (aggregator)", "aggregator", "https://news.google.com/rss/search?q=when:1d+source:Associated+Press&hl=en-US&gl=US&ceid=US:en"),
]
def robots(site):
    try:
        t = urllib.request.urlopen(urllib.request.Request(site + "/robots.txt", headers=UA), timeout=20).read().decode("utf-8", "replace")
    except Exception as e:
        return "robots:" + type(e).__name__, None
    groups, cur = [], None
    for raw in t.splitlines():
        l = raw.split("#")[0].strip()
        if not l: continue
        k, _, v = l.partition(":"); k = k.strip().lower(); v = v.strip()
        if k == "user-agent":
            if cur is None or cur["rules"]: cur = {"agents": [], "rules": []}; groups.append(cur)
            cur["agents"].append(v)
        elif k in ("allow", "disallow") and cur is not None: cur["rules"].append((k, v))
    blocked = sorted({a for g in groups if ("disallow", "/") in g["rules"] for a in g["agents"] if a.lower() in AI})
    return blocked, t
def check(row):
    region, lang, name, kind, u = row
    rec = {"region": region, "lang": lang, "outlet": name, "kind": kind, "url": u}
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
        b = r.read(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
        root = ET.fromstring(b.lstrip(b"\xef\xbb\xbf").lstrip())
        items = root.findall("./channel/item") or root.findall(A + "entry") or root.findall(RDF + "item")
        rec.update(status=r.status, items=len(items))
    except Exception as e:
        rec.update(status="FAIL " + type(e).__name__ + " " + str(e)[:40], items=0)
    site = "{0.scheme}://{0.netloc}".format(urlsplit(u))
    blocked, t = robots(site)
    rec["ai_blocked"] = blocked
    if t:
        p = rp.RobotFileParser(); p.parse(t.splitlines()); rec["generic_may_fetch"] = p.can_fetch("CubbyLLM/0.1", u)
    return rec
with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(check, C))
for r in out:
    ok = isinstance(r["status"], int) and r["items"] > 0
    print(f"{'OK  ' if ok else 'FAIL'} {r['region']:11s} {r['lang']} {r['outlet'][:34]:34s} items={r['items']:3d} AI-blocked={len(r['ai_blocked']) if isinstance(r['ai_blocked'], list) else r['ai_blocked']} {'' if ok else r['status']}")
json.dump(out, open(r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\logs\_intl_feeds.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("working", sum(1 for r in out if isinstance(r["status"], int) and r["items"] > 0), "of", len(out))
