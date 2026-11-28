"""cubbyllm/emit/talk_format.py -- the talk seat's second prompt shape, and the one-emoji guard (order 0079's
perspective addendum, items 1 and 3).

Both are off by default; off, `talk.talk_prompt` and `talk.talk_answer` are what they were, byte for byte.
- ``CB_TALK_FORMAT=2`` (read by `talk.talk_prompt` and `talk.talk_answer` on every call): one label for each speaker,
  to the end. The conversation runs ``User:`` / ``Cubby:`` and the prompt ends ``User: <the turn>`` then ``Cubby:``;
  a first turn is the same with nothing above it. The lines above stay as they are (``Explicit:``, ``Premise:``,
  ``Mood:``, ``Shade:``, ``They seem:``, ``Facts:``), and ``\\nCubby:`` joins the answer's stops. Today's shape labels
  the earlier turns ``User:`` / ``Assistant:`` and the last ``Question:`` / ``Answer:``, so nothing tied the reply's
  "I" to the ``Assistant:`` lines or "you" to ``User:`` (the board's first casual turns, 2026-11-27). talk_v3a was
  trained on today's shape; serving keeps it until an adapter trained on this one passes 0079's gate.
- ``CB_TALK_ONE_EMOJI=1`` (read by `talk.talk_answer`, through `guard`), held to the sheet: a reply to a turn with no
  emoji keeps none; otherwise it keeps its first and drops the rest, until a seat trained on the character sheet of
  2026-11-28 lands ("only when the user uses one in that turn; at most one").

`render` is the one function for serving and for every talk row of the final mix: `parse_v1` reads a row in today's
shape back into the context `render` takes (`validation/sft/talk_format2_0079.py` converts the mix with the two).

Wired: WIRED (`talk.talk_prompt` and `talk.talk_answer` import it; 0079's conversion and step tool render rows with it).
"""
from __future__ import annotations

import os
import re

__wiring__ = "WIRED"

UNLOCK = "Explicit: allowed"                             # talk.TALK_UNLOCK (pinned equal)
DASH = " — "
STOPS = ("\nCubby:",)
LABELS = ("User:", "Cubby:", "Assistant:", "Question:", "Answer:", "Conversation:")
_HEADS = (("premise", "Premise:"), ("mood", "Mood:"), ("shade", "Shade:"), ("seems", "They seem:"))

# an emoji, as one unit: a flag (two regional letters), or a pictograph with its variation selector, its skin tone and
# any zero-width-joined parts (one family emoji is one emoji)
_PICT = "[\U0001F300-\U0001FAFF☀-➿⬀-⯿⌚⌛⏩-⏺]"
_EMOJI = ("(?:[\U0001F1E6-\U0001F1FF]{2}|" + _PICT + "️?[\U0001F3FB-\U0001F3FF]?(?:‍" + _PICT
          + "️?[\U0001F3FB-\U0001F3FF]?)*)")
EMOJI = re.compile(_EMOJI)
_EXTRA = re.compile("[ \t]*" + _EMOJI)


def on() -> bool:
    """``CB_TALK_FORMAT=2``: the second shape (above). Anything else: today's."""
    return os.environ.get("CB_TALK_FORMAT", "1") == "2"


def one_emoji_on() -> bool:
    """``CB_TALK_ONE_EMOJI=1``: `guard` keeps a reply's first emoji only."""
    return os.environ.get("CB_TALK_ONE_EMOJI") == "1"


def emojis(text: str) -> list[str]:
    return EMOJI.findall(text or "")


def one_emoji(text: str) -> str:
    """The text with its first emoji kept and every later one dropped, with the spaces before it."""
    first = EMOJI.search(text or "")
    if first is None:
        return text
    head, tail = text[:first.end()], text[first.end():]
    tail = _EXTRA.sub("", tail)
    return (head + tail).rstrip(" \t") if tail != text[first.end():] else text


def no_emoji(text: str) -> str:
    """The text with every emoji dropped, with the spaces before it."""
    return _EXTRA.sub("", text).strip(" \t") if EMOJI.search(text or "") else text


_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])|(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])")


def checked_on() -> bool:
    """``CB_TALK_CHECKED=1`` (order 0079, item 5; `cubbyllm/speech/sheet_checks.py`): here, `guard` takes markdown
    emphasis out of every talk reply (the board shows plain text)."""
    return os.environ.get("CB_TALK_CHECKED") == "1"


def plain(text: str) -> str:
    """The text without markdown emphasis: **bold**, __bold__, *italic*, _italic_ (a lone star, "5 * 3", stays)."""
    text = _BOLD.sub(lambda m: m.group(2), text)
    return _ITALIC.sub(lambda m: m.group(1) or m.group(2), text)


def guard(reply: str, user: str | None = None) -> str:
    """The talk seat's answer as the host lets it out. ``CB_TALK_CHECKED=1``: no markdown emphasis.
    ``CB_TALK_ONE_EMOJI=1``, held to the sheet: a reply to a turn with no emoji keeps none, otherwise its first only;
    a caller that has no turn to give keeps the first. Both off: as is."""
    if checked_on():
        reply = plain(reply)
    if not one_emoji_on():
        return reply
    return no_emoji(reply) if user is not None and not emojis(user) else one_emoji(reply)


def user_turn(prompt: str, context=None) -> str:
    """The turn a talk call answers: the last of the context's turns, else its question, else the prompt."""
    ctx = context if isinstance(context, dict) else {}
    if ctx.get("turns"):
        return str(ctx["turns"][-1][1])
    return str(ctx.get("question") or prompt or "")


def _value(key: str, v) -> str:
    if key == "mood" and isinstance(v, dict):            # the eight scales, as `mood.line` writes them
        from ..affect import mood
        return mood.line(v)[len("Mood: "):]
    if key == "shade" and isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v) or "none"
    return str(v).strip()


def _fact(f) -> str:
    if isinstance(f, (list, tuple)) and len(f) == 3:
        e, r, v = f
        return f"- {e}{DASH}{r}: {v}"
    return f"- {str(f).strip()}"


def render(prompt: str, context=None) -> str:
    """The prompt in the second shape, from the host's context (talk_prompt's keys: `explicit`, `facts`, `question`,
    `turns` as [(role, text)], the last one the user's; and `premise`, `mood`, `shade`, `seems` when a row or the
    host has them), else `prompt` as the user's turn. Unlike today's shape, facts and turns can come together."""
    ctx = context if isinstance(context, dict) else {}
    lines = [UNLOCK] if ctx.get("explicit") else []
    for key, label in _HEADS:
        if ctx.get(key) is not None:
            lines.append(f"{label} {_value(key, ctx[key])}")
    if ctx.get("facts"):
        lines += ["Facts:"] + [_fact(f) for f in ctx["facts"]]
    turns = ctx.get("turns")
    if turns:
        *before, (_, last) = turns
        lines += [f"{'User:' if str(role).lower() == 'user' else 'Cubby:'} {str(text).strip()}" for role, text in before]
    else:
        last = ctx.get("question") or prompt
    lines += [f"User: {str(last).strip()}", "Cubby:"]
    return "\n".join(lines)


def _starts(line: str, label: str) -> bool:
    return line == label or line.startswith(label + " ")


def parse_v1(prompt: str) -> dict | None:
    """A talk row's prompt in today's shape -> the context `render` takes. None when a line cannot be placed for sure:
    a fact or a turn whose wrapped line starts with a speaker's label, or a question holding one (they would read as
    turns in the second shape). The texts are kept as written, wrapped lines included."""
    if not prompt.endswith("\nAnswer:"):
        return None
    lines = prompt[:-len("\nAnswer:")].split("\n")
    ctx: dict = {}
    i, n = 0, len(lines)
    if i < n and lines[i] == UNLOCK:
        ctx["explicit"], i = True, i + 1
    for key, label in _HEADS:
        if i < n and _starts(lines[i], label):
            ctx[key], i = lines[i][len(label):].strip(), i + 1
    if i < n and lines[i] == "Facts:":
        facts: list[str] = []
        i += 1
        while i < n and not (lines[i] == "Conversation:" or _starts(lines[i], "Question:")):
            if lines[i].startswith("- "):
                facts.append(lines[i][2:])
            elif not facts or any(_starts(lines[i], lb) for lb in LABELS):
                return None
            else:
                facts[-1] += "\n" + lines[i]
            i += 1
        ctx["facts"] = facts
    turns: list[list[str]] = []
    if i < n and lines[i] == "Conversation:":
        i += 1
        while i < n and not _starts(lines[i], "Question:"):
            if _starts(lines[i], "User:") or _starts(lines[i], "Assistant:"):
                role, _, text = lines[i].partition(":")
                turns.append(["user" if role == "User" else "assistant", text.strip()])
            elif not turns or any(_starts(lines[i], lb) for lb in LABELS):
                return None
            else:
                turns[-1][1] += "\n" + lines[i]
            i += 1
    if i >= n or not _starts(lines[i], "Question:"):
        return None
    rest = lines[i + 1:]
    if any(_starts(x, lb) for x in rest for lb in LABELS):
        return None
    question = "\n".join([lines[i][len("Question:"):].strip()] + rest)
    if turns:
        ctx["turns"] = [tuple(t) for t in turns] + [("user", question)]
    else:
        ctx["question"] = question
    return ctx
