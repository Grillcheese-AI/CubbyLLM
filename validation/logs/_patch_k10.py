import io, sys
path, new_path = sys.argv[1], sys.argv[2]
text = io.open(path, encoding="utf-8").read()
new = io.open(new_path, encoding="utf-8").read().strip()
assert "**H-K10 — Hypertokens" not in text
k9 = text.index("**H-K9 — Jumps in time with a composition law")
end = text.index("\n", k9)
text = text[:end] + "\n\n" + new + text[end:]
old = "**The thread through all nine:**"
assert text.count(old) == 1
text = text.replace(old, "**The thread through all ten:**")
old_h = "and a repo to read, `google-research/swirl-dynamics` (H-K9)."
assert text.count(old_h) == 1
text = text.replace(old_h, "a repo to read, `google-research/swirl-dynamics` (H-K9); and \"if tokens are the problem why not simply replace them by something more efficient? like hypertokens\" (H-K10).")
old_row = "| K — Knowledge, the map, the state (research pause) | H-K1–H-K9 |"
assert text.count(old_row) == 1
text = text.replace(old_row, "| K — Knowledge, the map, the state (research pause) | H-K1–H-K10 |")
old_item = "H-K9 on BlockWorld traces, H-K5's rule generator.\n"
assert text.count(old_item) == 1
text = text.replace(old_item, "H-K9 on BlockWorld traces, H-K5's rule generator; H-K10 gate 1 (hypertokens: a fact shown once, read back beyond the window) is a CPU hour on the tiny trunk and comes right after K1.\n")
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched; H-K10 length", len(new))
