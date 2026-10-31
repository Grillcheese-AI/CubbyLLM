import sys, urllib.request, gzip, xml.etree.ElementTree as ET
A = "{http://www.w3.org/2005/Atom}"
for u in sys.argv[1:]:
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "CubbyLLM/0.1"}), timeout=30)
    b = r.read(); b = gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b
    root = ET.fromstring(b)
    print(u, "| root", root.tag, "| type", r.headers.get("Content-Type"))
    rss = root.findall("./channel/item"); atom = root.findall(A + "entry")
    if atom:
        ds = sorted(e.findtext(A + "updated") or e.findtext(A + "published") or "" for e in atom)
        print("   Atom entries", len(atom), "| title", root.findtext(A + "title"), "| oldest", ds[0], "| newest", ds[-1])
    else:
        print("   RSS items", len(rss), "| first bytes", b[:120])
