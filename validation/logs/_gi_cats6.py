import urllib.request, re, html, gzip, json, time, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
h = get("https://www.globalissues.org/news/topic").decode("utf-8", "replace")
topics, seen = [], set()
for href, tid, txt in re.findall(r'<a[^>]+href="(/news/topic/(\d+))"[^>]*>(.*?)</a>', h, re.S):
    name = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(txt))).strip()
    if tid in seen or not name: continue
    seen.add(tid); topics.append((int(tid), name))
out = []
for tid, name in topics:
    u = f"https://www.globalissues.org/news/topic/{tid}/feed"
    try:
        es = ET.fromstring(get(u)).findall(A + "entry")
        newest = max((e.findtext(A + "updated") or "") for e in es) if es else "-"
        out.append({"topic": name, "id": tid, "url": u, "entries": len(es), "newest": newest})
        print(f"{tid:>5} {name[:44]:44s} entries={len(es):3d} newest={newest[:10]}")
    except Exception as ex:
        print(f"{tid:>5} {name[:44]:44s} FAIL {type(ex).__name__}")
    time.sleep(0.5)
json.dump({"publisher": "Global Issues", "checked": "2026-10-22", "listing": "https://www.globalissues.org/news/topic",
           "pattern": "https://www.globalissues.org/news/topic/<id>/feed (Atom; /news/feed/issue/<id> ignores the id)",
           "feeds": out}, open(r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\logs\_globalissues_feeds.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("topics", len(topics), "feeds ok", len(out))
