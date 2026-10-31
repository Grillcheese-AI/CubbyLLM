import io, re, sys
path, new_path = sys.argv[1], sys.argv[2]
text = io.open(path, encoding="utf-8").read()
new = io.open(new_path, encoding="utf-8").read().strip()
start = text.index("**H-K7 — The knowledge budget split")
end = text.index("\n", start)
old = text[start:end]
assert old.endswith("(same mechanism, places → entities).**"), old[-80:]
text = text[:start] + new + text[end:]
old_intro = ('"decoupling the information without trillion of tokens and still have the same power" (H-K7)')
new_intro = ('"décupler [tenfold] the information without trillion of tokens and still have the same power" (H-K7, data efficiency)')
assert text.count(old_intro) == 1
text = text.replace(old_intro, new_intro)
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched; H-K7 length", len(new))
