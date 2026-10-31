import urllib.request, re, gzip, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
main = {e.findtext(A + "id") for e in ET.fromstring(get("https://www.globalissues.org/news/feed")).findall(A + "entry")}
h = get("https://www.globalissues.org/news/topic").decode("utf-8", "replace")
topics = sorted(set(re.findall(r'href="(/news/topic/\d+)"', h)), key=lambda s: int(s.rsplit("/", 1)[1]))
print("topic pages listed:", len(topics))
t1 = get("https://www.globalissues.org/news/topic/168").decode("utf-8", "replace")
print("feed links on topic 168:", sorted(set(re.findall(r'<link[^>]+(?:rss|atom)[^>]*>', t1)))[:4], sorted(set(re.findall(r'href="([^"]*feed[^"]*)"', t1)))[:6])
for u in ["https://www.globalissues.org/news/topic/168/feed", "https://www.globalissues.org/news/feed/topic/168"]:
    try:
        b = get(u)
        try:
            ids = {e.findtext(A + "id") for e in ET.fromstring(b).findall(A + "entry")}
            print("XML", u, "entries", len(ids), "shared with main", len(ids & main))
        except ET.ParseError:
            print("HTML", u, "bytes", len(b))
    except Exception as ex:
        print("FAIL", u, type(ex).__name__)
