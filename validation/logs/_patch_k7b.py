import io, sys
path, new_path = sys.argv[1], sys.argv[2]
text = io.open(path, encoding="utf-8").read()
new = io.open(new_path, encoding="utf-8").read().strip()
start = text.index("**H-K7 — Décupler: the same power from ten times fewer")
end = text.index("\n", start)
text = text[:start] + new + text[end:]
old_intro = '"décupler [tenfold] the information without trillion of tokens and still have the same power" (H-K7, data efficiency)'
new_intro = ('"décupler [tenfold] the information without trillion of tokens and still have the same power … 10 fold could be '
             '1,000,000 fold too … 40B token to train the model to be par to a 1T token model" (H-K7, data efficiency)')
assert text.count(old_intro) == 1
text = text.replace(old_intro, new_intro)
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched; H-K7 length", len(new))
