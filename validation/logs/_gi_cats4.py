import urllib.request, re, gzip, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
root = ET.fromstring(get("https://www.globalissues.org/news/feed"))
for e in root.findall(A + "entry")[:5]:
    cats = [c.get("term") or c.get("label") for c in e.findall(A + "category")]
    print("categories:", cats, "| child tags:", sorted({c.tag.replace(A, "") for c in e}))
# Does an issue page list news items of its own?
t = get("https://www.globalissues.org/issue/178/climate-change-and-global-warming").decode("utf-8", "replace")
news = re.findall(r'href="(/news/\d{4}/\d{2}/\d{2}/\d+)"', t)
print("news article links on the climate issue page:", len(news), news[:3])
for cand in ["https://www.globalissues.org/news/archive/climate-change-and-global-warming",
             "https://www.globalissues.org/news/topic/climate-change", "https://www.globalissues.org/news/archive"]:
    try:
        h = get(cand).decode("utf-8", "replace")
        print("page", cand, "| news links", len(set(re.findall(r'href="(/news/\d{4}/\d{2}/\d{2}/\d+)"', h))),
              "| topic/issue links", sorted(set(re.findall(r'href="(/news/(?:archive|topic|issue)[^"]*)"', h)))[:8])
    except Exception as ex:
        print("page", cand, "FAIL", type(ex).__name__, str(ex)[:50])
