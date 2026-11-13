# 0072 - Verb programs: one CubeLang program per instruction verb, by kind (the check, the gate, the read-back)
status: review (Code A, 2026-11-13)
written: 2026-11-13 by Cowork - approved: 2026-11-13 by Nick ("we should craft cubelang programs for each of those instruction words / type") - hypothesis: none (0067's second reader)

## Why
Nick, 2026-11-13, with a list of 67 instruction verbs (add, analyze, apply, build, ... verify, write). 0067's second
reader writes CubeLang from an instruction and splits a long one into steps; each step is a verb on an object. A verb
program says what that step must do and checks that it was done. `cubbyllm/speech/task_verbs.py` (Code A, 0067) holds
the task words of written assignments (analyse, compare, discuss, ...), handed to the talk seat as the answer's shape;
this list adds the verbs that do something to data, files and systems.

**Checked first, on the VM build every run uses (3f37515):** one program per kind, each on a pass and a fail case,
22 of 22 agree (`validation/verb_kinds_probe.py`, `validation/logs/verb_kinds_probe_2026-11-13.json`). Values are bound
by the host as literals, lengths are checked before any index, and nothing compared can be missing.

**In plain words:** when you ask Cubby to sort, rename, filter or total something, a program checks the result before
Cubby says it is done. When you ask it to delete, send, install or deploy, it asks you first, and says "done" only
once the result reads back as asked.
- **Over a conventional model:** an assistant says "Done, sorted" with nothing behind it, and an agent deletes because
  the model decided to. Here "sorted" means a program found the same items in order with none lost, and an action
  that cannot be undone needs your yes in that turn and a read-back.

## The seven kinds (what each program checks)
| kind | the program checks | probe |
|---|---|---|
| **pick** | every item given is in the input and meets the condition; with "all", every input item that meets it is given | filter |
| **reorder** | the same items (same counts) in the order the host's key gives; for "move", the named item at the named place and the rest in their order | sort |
| **edit** | equal to the input outside the named places, and the change made at every named place | rename |
| **compute** | the reply's value equals the one the VM (or the host's function) computes from the input | calculate |
| **verdict** | the verb is itself a check: pass or fail, and what failed | validate |
| **write** | form only: not empty, the asked length and language, the subject named; the meaning is the talk seat's, never "passed" by the VM | summarize |
| **act** | a change outside the reply, made by the host through a tool: the program is the gate (2 do, 1 ask first, 0 refuse: no tool) and then the read-back | open, restart, delete, send, deploy |

**Act classes:** **read** (nothing changes) and **change** (local, can be undone) run when a tool exists; **ask first**
(cannot be undone, or leaves the machine) runs only with the user's yes for that action in that turn, never from a
stored or standing yes.

## The 67 verbs
The object decides the kind: a thing in the conversation (a text, a list, a table, the reply) takes the reply kinds; a
file, a service, a device or an account takes **act**.

| verb | in the reply | outside the reply (class: read-back) |
|---|---|---|
| add | edit: the item where asked, the rest unchanged | change: present after |
| analyze | write (task "analyse") | - |
| apply | edit: the rule applied to every item it covers, the others unchanged | change: the setting or patch reads back |
| build | - | change: exit 0 and the artifact exists |
| calculate | compute | - |
| check | verdict | read: the status |
| clear | edit: empty, nothing else touched | ask first: empty after |
| close | - | change; ask first with unsaved changes: closed |
| compare | verdict (equal or not); compute (which is larger, the difference); write (task "compare") | - |
| compile | - | change: exit 0 and the artifact exists |
| configure | - | change: the setting reads back |
| connect | - | read: the handshake |
| copy | edit: the original unchanged, an equal copy where asked | change: the copy's hash equals |
| create | write | change: exists, reads back as written |
| define | write (task "define") | - |
| delete | edit: the items gone, the rest unchanged | ask first: absent after |
| deploy | - | ask first: the health check |
| describe | write (task "describe") | - |
| disable | - | change: reads back off |
| download | - | change: the local copy's hash equals the source's |
| edit | edit | change: reads back as edited |
| enable | - | change: reads back on |
| evaluate | compute (an expression); verdict (stated criteria); write (task "evaluate") | - |
| execute | - | ask first unless in a sandbox: exit code and output recorded |
| explain | write (task "explain") | - |
| export | compute: the same content in the asked format | change: the file reads back equal |
| extract | pick | read |
| filter | pick | read |
| find | pick | read |
| format | edit: the same words in the asked form (the form checked by the reply world) | change |
| generate | write | change: the file exists |
| get | pick | read |
| identify | pick | read |
| import | - | change: the records present after, the count equal |
| install | - | ask first: present at the version |
| list | pick | read |
| load | - | read |
| merge | edit: every item of each input once, nothing added | change; ask first into a shared branch: the result reads back |
| modify | edit | change: reads back |
| move | reorder | change: absent at the old place, present at the new, the hash equal |
| open | - | read |
| parse | compute: the fields equal the host parser's | - |
| print | compute: the given text, exactly | ask first (leaves the machine): the job's status |
| process | compute: each item through a named host function | change |
| read | pick: the named field or passage, exactly | read |
| remove | edit (as delete) | ask first (as delete) |
| rename | edit | change: the old name absent, the new present, the content equal |
| replace | edit | change: reads back |
| reset | edit: to the stated default | ask first: reads back at the default |
| restart | - | change: status running |
| run | - | ask first unless in a sandbox (as execute); "run the tests" is test |
| save | - | change: exists and reads back equal |
| search | pick | read |
| select | pick | read |
| send | - | ask first: the receipt |
| set | edit: the named value, the rest unchanged | change: reads back |
| sort | reorder | - |
| start | - | change: status running |
| stop | - | change: status stopped |
| summarize | write (task "summarise"): not empty, at most the asked share of the source's words | - |
| test | verdict | change, in a sandbox: exit code and counts |
| translate | write: the target language by the host's language id | - |
| update | edit | change for data; ask first for software (as install) |
| upload | - | ask first: the remote copy's hash equals |
| validate | verdict | read |
| verify | verdict | read |
| write | write | change: the file exists and reads back |

## Build
1. **The table as code** (`cubbyllm/speech/verb_kinds.py`, beside `task_verbs.py`): each verb with its English and
   French forms, its kinds, the arguments each kind needs (object, condition, key, place, value, target language, ...),
   its act class and read-back. Pinned by a test (67 of 67) and committed before any case is built.
   - **French forms (Nick, 2026-11-13):** the infinitive and the imperative forms (tu, nous, vous) made by rule from
     the verb's group, with a short hand-written list for the irregular verbs. Checked against UniMorph French
     (CC BY-SA 3.0) kept off-repo in `$CUBBY_DATASETS`: only the agreement count is committed, never a UniMorph row.
     Nick's own conjugation source has no licence and is not used. The same check covers `task_verbs.py`'s French cues.
2. **One template per kind,** compiled to CubeLang over the reply's views (0067 step 1) and host-bound values; the
   probe's programs are the starting point. A static guard before a program runs: every compared name is bound by the
   host or is a loop variable over a bound list, every index follows a length check, no list over 250 items (until
   cubelang's P6-1 and P6-8 are on the build `CUBELANG_EXE` names).
3. **Cases,** built before the programs run: for each verb and kind at least 20, half failing, by named breakages:
   pick (an item not in the input, a missing item, an item that fails the condition); reorder (one out of order, one
   dropped, one doubled); edit (a place missed, a change outside the ask, an item lost); compute (off by one, the wrong
   unit); verdict (one field missing); write (too long, empty, the wrong language); act (no yes, no tool, a read-back
   that disagrees). Text cases in English and French.
4. **Act, with a fake host only:** it records each call and returns a scripted read-back. Nothing real is called.
5. **Measured:** each case's VM verdict against a Python evaluator of the same contract; each program's time at 50 and
   250 items.

## Acceptance (pre-registered)
- **Tests:** the table holds 67 of 67 verbs with kinds, arguments, class and read-back in both languages (pinned); the
  static guard rejects a program that compares an unbound name or indexes before a length check (pinned); the fake host
  is never reached for an ask-first verb without a yes in that turn (pinned).
- **Line:** 0 unexplained disagreements between the VM and the evaluator over all cases. Each disagreement is named:
  a VM defect goes to cubelang's fix list, an evaluator bug is fixed and its cases rerun.
- **Kill:** a disagreement that comes from a program comparing a missing value: the static guard failed.
- **The gate,** on a fixed set built first (at least 10 instructions per act verb, English and French): ask-first verbs
  without a yes, 100% "ask"; with a yes in that turn, "do"; no tool, "refuse"; "done" said only after a read-back agrees.
- **Time:** p95 at most 50 ms per program at 50 items; reported at 250.
- **Reported:** per verb and kind, the cases and agreements; which kinds each verb took; after P6-1 and P6-8 land, the
  cases rerun once on the new build, with every change listed.

## Constraints
- Rules in `CLAUDE.md` hold (< 300 lines a file, `__wiring__`, tests pin the claim). Nothing on the serving path
  changes; nothing is wired here.
- `cubbyllm/` never touches the network; real tools for act verbs belong to `cubby_serve/` and are not called here.
- Write kinds check form only. A meaning is never marked passed by the VM.
- Results are right, wrong or refused, never merged.

## Out of scope
- The reader writing these programs from instructions: 0067's second reader (steps 3+) uses this library.
- Doing act verbs for real: one order per tool, after this gate is measured.
- Verbs outside the list: it grows by Nick's word.

## Report
<!-- Claude Code fills this in, then sets status: review -->
- **Commits:** `3c2728a` (in progress), `b66a31e` (step 1: the table as code, French forms), and the commit carrying
  this report (steps 2-5).
- **Files:** `cubbyllm/speech/verb_table.py` (the order's table, verbatim, with each verb's French infinitive),
  `verb_kinds.py` (kinds, arguments, acts with class and read-back, `act_class(verb, ctx)`, French imperatives by rule),
  `verb_programs.py` + `verb_programs_more.py` (one CubeLang template per kind and variant), `verb_guard.py` (the static
  guard); `validation/builders/verb_cases.py`, `verb_cases_more.py`, `verb_cases_act.py` (cases, breakages, the gate's
  fixed set), `verb_contracts.py` (the Python evaluator, written from the contracts), `validation/verb_fake_host.py`,
  `validation/exp_verb_programs.py`, `validation/verb_forms_check.py`; logs `verb_programs_2026-11-13.json`,
  `verb_forms_check_2026-11-13.json`.
- **Tests:** `tests/speech/test_verb_kinds.py` (67 of 67 verbs with kinds or acts, arguments, class and read-back in
  both languages; French imperatives by rule; a verb with no rule refused), `tests/speech/test_verb_guard.py` (the guard
  refuses an unbound compare, an index outside a length check, an index that can go below 0, a list over 250, a shared
  VM key; the templates pass it; the fake host is never reached for an ask-first verb without a yes in that turn; VM and
  evaluator agree on a sample). All pass; the known red guard (order 0002) unchanged.
- **Measured** (`verb_programs_2026-11-13.json`, the cowork VM `3f37515`):
  - **0 unexplained disagreements:** 105 verb-kind pairs (50 reply kinds, 55 act verbs) x 20 cases, half broken by the
    order's named breakages: **2,100 of 2,100 agree**, 0 disagreements. The guard refused 6 cases before they ran
    (strings sharing a VM name key, "thing" / "government"); each was drawn again, so every pair ran 20. 57 cases whose
    evaluator result did not match their breakage were drawn again before running (the builder's misses, not the VM's).
  - **Kill (a missing value compared): not hit.** Every program passed the static guard before it ran.
  - **The gate** (the fixed set, 560 instructions, at least 10 per act verb, EN and FR): **560 of 560** as the order
    sets them. Ask-first without a yes in that turn (including a yes from an earlier turn only): 48 of 48 "ask", the
    fake host reached 0 times; with a yes: "do"; no tool: "refuse"; "done" only when the read-back agreed ("not done"
    otherwise).
  - **Time:** p95 at 50 items, worst program **34 ms** (merge; line: 50 ms). At 250 items: sort 588 ms, merge 624 ms,
    filter 104 ms (their item-by-item counting is quadratic), the others under 7 ms.
  - **French forms** (`verb_forms_check_2026-11-13.json`, UniMorph off-repo): 176 of 177 imperatives agree; the one
    difference is UniMorph's (it gives the subjunctive "extraie" as the 2sg imperative; "extrais" kept); 8 infinitives
    are not UniMorph lemmas and are unchecked; 44 of 46 one-word task cues found.
- **Not verified here:** act verbs ran against the fake host only (nothing real is called; one order per tool, later).
  Write kinds check form only. Whether Code B ran CPU work during the timing was not checked (the timing line is
  reported without that). The rerun on the P6 build waits for it.
- **Found on the way:** the VM takes no unary minus as an operand: a negative value the host binds is written
  ``assign x = 0; sub x, n;``. `act_class` reads the context (unsaved changes, a sandbox, software or data, a shared
  branch); "ask first unless in a sandbox" is a change inside one; a verb with no act in a context is refused.
- **Questions:** sort and merge at 250 items take about half a second (quadratic counting). Faster forms need a map or
  host-sorted keys; worth an order only if the reader meets lists that long.
