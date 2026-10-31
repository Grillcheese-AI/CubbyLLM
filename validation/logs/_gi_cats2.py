import urllib.request, re, gzip, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    return r.headers.get("Content-Type", ""), b
ct, b = get("https://www.globalissues.org/issue/2/causes-of-poverty")
t = b.decode("utf-8", "replace")
news_links = sorted(set(re.findall(r'href="(/news/[^"#]*)"', t)))
print("news links on the issue page:", [l for l in news_links if not re.search(r"/news/\d{4}/", l)][:20])
for u in ["https://www.globalissues.org/news/feed/issue/2", "https://www.globalissues.org/issue/2/feed",
          "https://www.globalissues.org/news/issue/2/feed", "https://www.globalissues.org/whatsnew/whatsnew.xml",
          "https://www.globalissues.org/news/archive/issue/2"]:
    try:
        ct, b = get(u)
        try:
            root = ET.fromstring(b); n = len(root.findall(A + "entry")) or len(root.findall("./channel/item"))
            print("XML ", u, "| entries", n, "| title", root.findtext(A + "title") or root.findtext("./channel/title"))
        except ET.ParseError:
            print("HTML", u, "|", ct, "| bytes", len(b), "| feed links:", sorted(set(re.findall(r'href="([^"]*feed[^"]*)"', b.decode("utf-8", "replace"))))[:6])
    except Exception as e:
        print("FAIL", u, type(e).__name__, str(e)[:60])
