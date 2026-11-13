# 0072 - Verb programs: one CubeLang program per instruction verb, by kind (the check, the gate, the read-back)
status: open (Nick sets who builds it: Code A after 0067's step 2, or Code B after 0069)
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
   French forms (French from Nick's conjugation source, which Code A is reading: infinitive and the imperative forms),
   its kinds, the arguments each kind needs (object, condition, key, place, value, target language, ...), its act class
   and read-back. Pinned by a test (67 of 67) and committed before any case is built.
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
- **Commits:**
- **Files:**
- **Tests:**
- **Measured:** <against each acceptance line>
- **Not verified here:**
- **Found on the way:**
- **Questions:**
