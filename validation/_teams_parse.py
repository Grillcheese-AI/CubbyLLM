"""Replacement parser for teams_harvest - spliced in by the patch step below.

The Teams copy-out is not one shape, it is two, and the first version of this parser
only understood the simpler one. That is why the audit reported `Nicolas` x62 and
`Wednesday` x28 sitting INSIDE message bodies: the chrome was being read as content.

  SHAPE A   speaker / date / message
            Sabin Mancheron | 09-01 | pas facile

  SHAPE B   a PREVIEW line, then speaker and date in EITHER order, then the body:
            "celui la ? A2DBI 687 by Jean-Francois Dallaire"   <- preview, ends " by NAME"
            "Jean-Francois Dallaire"                            <- speaker
            "Wednesday 2:03 pm"                                 <- date
            ""
            "celui la ? A2DBI 687"                              <- the real message
            ""
            ("image")                                           <- attachment placeholder

The reliable anchor in shape B is the preview's trailing " by <Name>", so that is what
separates records; speaker and date are then whichever of the next two lines is which.
The preview is a TRUNCATED COPY of the body - keeping it would duplicate every message
and teach a model to repeat itself, so it is used as a delimiter and discarded.

Emoji stay. They are carrying real affect in this register (a shrug, a laugh, a thumbs
up doing the work a sentence would otherwise do), and the preview's alt-text names for
them ("Wink", "Laugh reaction") are chrome for exactly the same emoji, so the alt-text
goes and the character stays.
"""
import re

# " ... by Firstname Lastname" at end of line - the shape-B record separator.
PREVIEW_BY = re.compile(r"^(?P<prev>.*\S)\s+by\s+(?P<who>[^\d]{2,40})$")
STAMP = re.compile(r"^(\d{2}-\d{2}|\d{4}-\d{2}-\d{2}|(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day|"
                   r"Yesterday|Today)\b.*$|^\d{1,2}:\d{2}\s?(am|pm)$", re.I)

# Teams interface strings and attachment placeholders. `image` alone on a line is an
# attachment, not a word anyone typed.
CHROME = {"message list", "image", "find", "enter", "escape", "send", "press",
          "from", "messages", "meet now", "has context menu", "begin quote"}
REACTION = re.compile(r"^\d+\s+\w+\s+reaction", re.I)


def parse(text: str) -> list[dict]:
    lines = [l.rstrip() for l in text.splitlines()]

    # Pass 1: who are the speakers? A name recurring before a timestamp, plus every
    # name the previews attribute to.
    names = {}
    for i, l in enumerate(lines):
        s = (l or "").strip()
        m = PREVIEW_BY.match(s)
        if m:
            names[m.group("who").strip()] = names.get(m.group("who").strip(), 0) + 1
        if s and i + 1 < len(lines) and STAMP.match((lines[i + 1] or "").strip()):
            names[s] = names.get(s, 0) + 1
    speakers = {n for n, c in names.items() if c >= 2 and 1 < len(n) < 60}

    # Pass 2: cut the file into records at each separator, then read each record.
    marks = []
    for i, l in enumerate(lines):
        s = (l or "").strip()
        m = PREVIEW_BY.match(s)
        if m and m.group("who").strip() in speakers:
            marks.append((i, m.group("who").strip(), "B"))
        elif s in speakers and i + 1 < len(lines) and STAMP.match((lines[i + 1] or "").strip()):
            if not (marks and marks[-1][0] >= i - 2 and marks[-1][2] == "B"):
                marks.append((i, s, "A"))

    msgs = []
    for k, (start, who, shape) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        block = [(l or "").strip() for l in lines[start + 1:end]]
        when, body, seen_when = "", [], False
        for s in block:
            if not s:
                continue
            low = s.lower().rstrip(":")
            if low in CHROME or REACTION.match(s):
                continue
            if s in speakers:                        # the speaker line inside the record
                continue
            if not seen_when and STAMP.match(s):
                when, seen_when = s, True
                continue
            body.append(s)
        text_out = "\n".join(body).strip()
        if text_out:
            msgs.append({"speaker": who, "when": when, "text": text_out, "shape": shape})
    return msgs
