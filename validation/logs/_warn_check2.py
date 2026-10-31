import urllib.request, gzip, json, re, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40)
    b = r.read(); return r.status, r.headers.get("Content-Type", ""), (gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)
for name, u in [("ReliefWeb API v2 disasters", "https://api.reliefweb.int/v2/disasters?appname=cubbyllm&limit=3&sort[]=date:desc"),
                ("EC datamart today/alerts/cap", "https://dd.weather.gc.ca/today/alerts/cap/"),
                ("EC weather alerts Atom (Quebec City)", "https://weather.gc.ca/rss/warning/qc-147_e.xml")]:
    try:
        st, ct, b = get(u); t = b.decode("utf-8", "replace")
        if t.lstrip().startswith("{"):
            j = json.loads(t); print("OK", st, name, "| keys", list(j)[:5], "| count", j.get("count"), j.get("totalCount"),
                                    "| first", [d.get("fields", {}).get("name") for d in j.get("data", [])[:3]])
        else:
            print("OK", st, name, "| type", ct, "| bytes", len(b), "| sample", re.sub(r"\s+", " ", t[:160]))
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:100])
for name, u in [("GDACS rss", "https://www.gdacs.org/xml/rss.xml"), ("WHO news", "https://www.who.int/rss-feeds/news-english.xml")]:
    st, ct, b = get(u); root = ET.fromstring(b.lstrip(b"\xef\xbb\xbf")); items = root.findall("./channel/item")
    print("OK", name, "| items", len(items), "| newest", items[0].findtext("pubDate") if items else "-", "| first", (items[0].findtext("title") or "")[:70] if items else "-")
