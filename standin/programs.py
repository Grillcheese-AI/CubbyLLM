"""programs - the library of moves he forged, and how combos are born and retire.

Wired: WIRED (the pac agent). STANDALONE of any world: it prices, names,
breeds and retires patterns over the grammar in `world`, and never asks what
the patterns move through.

Split out of `pacman.py` so a second world inherits the whole forge - the
audit, the value function, the consolidation, the notebook - rather than
reimplementing it. `moves_program` is the only part that knows a move has a
name, and it takes them as arguments.
"""
from __future__ import annotations

import json
import pathlib
import re

from world import JUMP_COST, MOVES, OPP  # noqa: E402
__wiring__ = "WIRED"


PROGRAMS_PATH = pathlib.Path(__file__).resolve().parent / "data" / "out" / "cubbyman_programs.json"



class ProgramLibrary:
    """His generated programs — each with its source, the REASONING that led
    to it (what triggered it, the situation he was in, how the proposal was
    sampled, the VM's verdict) and a log of every use and what it earned.
    Nothing is ever deleted: a pattern that never pays is RETIRED (kept in
    the notebook, no longer offered) — he does not forget what he learned.
    Kept on disk as JSON plus a readable Markdown notebook (pacman_live
    persists procedural memory the same way) so the knowledge survives a
    restart and can be read afterwards."""

    def __init__(self, path: pathlib.Path | None = None, ledger=None) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.ledger = ledger                             # ledger.Ledger: decisions hashed + signed in SQLite (2026-09-04)
        if path is not None:
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
            except (OSError, ValueError):
                self.entries = {}
        if self.ledger is not None:
            self.audit()

    def audit(self) -> dict[str, str]:
        """Every entry that carries a certificate hash is checked against the ledger: the program text must hash to a
        signed, certified row made by the current VM build — otherwise the entry is RETIRED (never deleted) with the
        reason. Entries without a certificate (pre-ledger) are left as they are, flagged `uncertified`. -> {name: reason}."""
        out = {}
        for name, e in self.entries.items():
            cert = e.get("cert")
            if not cert:
                e["uncertified"] = True
                continue
            ok, why = self.ledger.verify(cert, e.get("program"))
            if not ok and not e.get("retired"):
                e["retired"] = True
                e["retired_reason"] = f"certificate: {why}"
                out[name] = why
            elif ok:
                e.pop("uncertified", None)
        if out:
            self.save()
        return out

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def names(self) -> list[str]:
        return [n for n, e in self.entries.items() if not e.get("retired")]

    def active(self) -> dict[str, dict]:
        return {n: e for n, e in self.entries.items() if not e.get("retired")}

    def add(self, name: str, pattern: str | None, program: str, kind: str, step: int,
            reasoning: dict, cert: str | None = None) -> None:
        self.entries[name] = {"pattern": pattern, "kind": kind, "program": program,
                              "born": step, "legal": 0, "used": 0, "saved": 0,
                              "reasoning": reasoning, "uses": [], "retired": False,
                              "cert": cert}                # the ledger's decision hash (None: pre-ledger / no VM run)
        self.save()

    def note_legal(self, name: str) -> None:
        if name in self.entries:
            self.entries[name]["legal"] += 1

    def note_used(self, name: str, saved: int, context: dict) -> None:
        if name in self.entries:
            e = self.entries[name]
            e["used"] += 1
            e["saved"] += saved
            if len(e["uses"]) < 200:
                e["uses"].append({"saved": saved, **context})
            self.save()

    def value(self, name: str) -> float:
        e = self.entries[name]
        return e["saved"] / (e["used"] + 1) + 0.05 * min(e["legal"], 20)

    def retire(self, step: int, keep: int = 10) -> str | None:
        """Stop OFFERING the least valuable never-used pattern once the active
        set is crowded (every legal move is an ASK candidate; a useless one is
        noise). It stays in the notebook with the reason — not forgotten."""
        act = self.active()
        cands = [n for n, e in act.items()
                 if e["kind"] == "pattern" and e["used"] == 0 and step - e["born"] > 40]
        if len(act) > keep and cands:
            worst = min(cands, key=self.value)
            self.entries[worst]["retired"] = True
            self.entries[worst]["retired_reason"] = (f"never used in {step - self.entries[worst]['born']} "
                                                     f"steps; legal {self.entries[worst]['legal']} times")
            self.save()
            return worst
        return None

    # ── ONE program, a reusable function per combo ─────────────────────────
    @staticmethod
    def fn_name(name: str) -> str:
        """A move's function in the Moves program: `COMBO-AAB` -> `combo_aab`."""
        return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_") or "move"

    def moves_program(self, extra: dict | None = None) -> str:
        """The one program he keeps editing: `program Moves implements ISolve`
        with a function per ACTIVE move (jump + patterns) — `extra` = {name:
        steps} adds the candidate function being certified. solve() is the
        catalogue smoke: it recovers how many moves the program holds."""
        fns = {}
        for name, e in self.active().items():
            if e["kind"] == "jump":
                fns[name] = ["hop", "hop"]
            elif e["kind"] == "pattern" and e.get("pattern"):
                fns[name] = list(e["pattern"])
        if extra:
            fns.update(extra)
        out = ["use vsa;\n\n# cubby-man's moves: one program, a reusable function per combo (edited in play)\n"
               "program Moves implements ISolve {\n"
               "    public function solve(mention: str): str {\n        create frame: number;\n"
               f'        bind frame, H1_MOVES, "{len(fns)}";\n        return recover(frame, H1_MOVES);\n    }}\n']
        for name, steps in fns.items():
            binds = "".join(f'        bind frame, H{i + 1}_STEP, "{s}";\n' for i, s in enumerate(steps))
            out.append(f"\n    public function {self.fn_name(name)}(): str {{\n        create frame: number;\n"
                       f"{binds}        bind frame, H{len(steps) + 1}_NAME, \"{name}\";\n"
                       f"        return recover(frame, H{len(steps) + 1}_NAME);\n    }}\n")
        out.append("}\n")
        return "".join(out)

    def consolidate(self, level: int, keep: int = 6) -> list[str]:
        """After a level: keep the `keep` most valuable ACTIVE patterns, retire
        the rest with the reason (never deleted). JUMP is structural and
        always stays. Returns the names retired."""
        act = {n: e for n, e in self.active().items() if e["kind"] == "pattern"}
        if len(act) <= keep:
            return []
        ranked = sorted(act, key=self.value, reverse=True)
        out = []
        for n in ranked[keep:]:
            e = self.entries[n]
            e["retired"] = True
            e["retired_reason"] = (f"consolidated after level {level}: value {self.value(n):.2f} below the "
                                   f"top {keep} (used {e['used']}x, saved {e['saved']}, legal {e['legal']}x)")
            out.append(n)
        if out:
            self.save()
        return out

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"entries": self.entries}, indent=1), encoding="utf-8")
            self.path.with_suffix(".md").write_text(self.notebook(), encoding="utf-8")
            self.path.with_name("cubbyman_moves.cube").write_text(self.moves_program(), encoding="utf-8")
        except OSError:
            pass

    def notebook(self) -> str:
        """The readable record: the ONE Moves program as it stands, then one
        section per move/tool with its reasoning (each stores the program as
        it was when certified, so the evolution is readable)."""
        out = ["# cubby-man's programs\n",
               "Every move he composed himself: why, the situation, the program the VM certified, "
               "and what it earned. Retired moves are kept — nothing learned is forgotten.\n",
               "\n## The Moves program (as it stands — a reusable function per active combo)\n",
               "\n```cubelang\n" + self.moves_program().rstrip() + "\n```\n"]
        for name, e in self.entries.items():
            r = e.get("reasoning", {})
            out.append(f"\n## {name}  ({'pattern ' + e['pattern'] if e.get('pattern') else e['kind']})"
                       f"{'  — RETIRED: ' + e.get('retired_reason', '') if e.get('retired') else ''}\n")
            out.append(f"- **trigger:** {r.get('why', '?')} — {r.get('because', '')}\n")
            s = r.get("situation", {})
            if s:
                out.append("- **situation:** " + ", ".join(f"{k} {v}" for k, v in s.items()) + "\n")
            if r.get("rationale"):
                out.append(f"- **proposal:** {r['rationale']}\n")
            out.append(f"- **VM verdict:** {r.get('verdict', '?')}\n")
            if e["kind"] == "tool":
                out.append(f"- **task:** expected `{r.get('expected')}` · got `{r.get('got')}` · "
                           f"{'PASS' if r.get('ok') else 'FAIL'}\n")
            else:
                out.append(f"- **earned:** used {e['used']}× · saved {e['saved']} steps · legal {e['legal']}×\n")
            out.append("\n```cubelang\n" + e["program"].rstrip() + "\n```\n")
            if e.get("uses"):
                out.append("\n| step | level | move | landed on | saved |\n|---|---|---|---|---|\n")
                for u in e["uses"][-12:]:
                    out.append(f"| {u.get('step')} | {u.get('level')} | {u.get('move')} | "
                               f"{u.get('landed', '')} | {u.get('saved')} |\n")
        return "".join(out)
