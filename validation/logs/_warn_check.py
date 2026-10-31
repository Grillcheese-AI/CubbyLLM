import urllib.request, gzip, json, re, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40)
    b = r.read(); return r.status, r.headers.get("Content-Type", ""), (gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)
tests = [
 ("GDACS RSS (all alerts)", "https://www.gdacs.org/xml/rss.xml"),
 ("GDACS RSS (7 days)", "https://www.gdacs.org/xml/rss_7d.xml"),
 ("USGS quakes M4.5+ past day (GeoJSON)", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"),
 ("USGS significant past month", "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"),
 ("Environment Canada CAP alerts index (datamart)", "https://dd.weather.gc.ca/alerts/cap/"),
 ("ReliefWeb API disasters", "https://api.reliefweb.int/v1/disasters?appname=cubbyllm&limit=3&sort[]=date:desc"),
 ("WHO news RSS", "https://www.who.int/rss-feeds/news-english.xml"),
 ("GDELT events last update", "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"),
]
for name, u in tests:
    try:
        st, ct, b = get(u)
        t = b.decode("utf-8", "replace")
        info = ""
        if "json" in ct or t.lstrip().startswith("{"):
            j = json.loads(t)
            if "features" in j: info = f"features={len(j['features'])} newest={max((f['properties']['time'] for f in j['features']), default=None)}"
            elif "data" in j: info = f"items={len(j['data'])} total={j.get('totalCount')} first={[d.get('fields', {}).get('name') for d in j['data'][:2]]}"
            else: info = f"keys={list(j)[:6]}"
        elif t.lstrip().startswith("<?xml") or t.lstrip().startswith("<rss"):
            root = ET.fromstring(b.lstrip(b"\xef\xbb\xbf"))
            items = root.findall("./channel/item")
            info = f"items={len(items)} newest={items[0].findtext('pubDate') if items else '-'} title={root.findtext('./channel/title')}"
        else:
            dirs = re.findall(r'href="(\d{8}/)"', t)
            info = f"html/listing bytes={len(b)} date-dirs={dirs[-3:] if dirs else '-'}"
        print(f"OK   {st} {name} | {info}")
    except Exception as e:
        print(f"FAIL {name} | {type(e).__name__} {str(e)[:90]}")
