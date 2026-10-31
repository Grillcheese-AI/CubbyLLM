import urllib.request, re, html, gzip, json, xml.etree.ElementTree as ET
UA = {"User-Agent": "CubbyLLM/0.1"}
A = "{http://www.w3.org/2005/Atom}"
def get(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30)
    b = r.read(); return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
def ids(u):
    root = ET.fromstring(get(u)); return [e.findtext(A + "id") or e.findtext(A + "link") for e in root.findall(A + "entry")]
main = set(ids("https://www.globalissues.org/news/feed"))
home = get("https://www.globalissues.org/").decode("utf-8", "replace")
issues, seen = [], set()
for m in re.findall(r'<a[^>]+href="([^"]*/issue/(\d+)/[^"]*)"[^>]*>(.*?)</a>', home, re.S):
    iid, name = m[1], re.sub(r"<[^>]+>", "", html.unescape(m[2])).strip()
    if iid in seen or not name: continue
    seen.add(iid); issues.append((iid, name))
out = []
for iid, name in issues:
    u = f"https://www.globalissues.org/news/feed/issue/{iid}"
    try:
        e = ids(u); same = len(set(e) & main)
        out.append({"issue": name, "id": int(iid), "url": u, "entries": len(e), "shared_with_main_feed": same})
        print(f"{iid:>5} {name[:40]:40s} entries={len(e):3d} shared_with_main={same}")
    except Exception as ex:
        print(f"{iid:>5} {name[:40]:40s} FAIL {type(ex).__name__}")
json.dump({"publisher": "Global Issues", "checked": "2026-10-22", "pattern": "https://www.globalissues.org/news/feed/issue/<id>",
           "feeds": out}, open(r"C:\Users\grill\Documents\GitHub\CubbyLLM\validation\logs\_globalissues_feeds.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("issues", len(issues))
