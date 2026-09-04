# Cubby, explained simply

*A plain-language description for people who are not engineers. Owner's draft of 2026-09-04, reworked. Lines marked
"coming" describe the design, not something you can use today.*

## What Cubby is

Cubby is a small AI assistant built by Grillcheese Research Lab that runs on your own computer, answers only when it
can check its answer, and says "I don't know yet" when it can't. Most assistants are one big model that guesses the
next word. Cubby is a small model attached to a checker: the model proposes, a separate verifier, the CubeLang virtual
machine, runs the proposal as a little program and only lets the answer out if the facts behind it hold. Everything it
claims, you can trace back to the facts it used.

## The list, in plain words

| what the design says | what it means for you |
|---|---|
| small and energy efficient | It runs on a normal gaming PC, not a data centre. Your electricity bill notices, the planet less so. |
| says it doesn't know instead of inventing | When the facts are not there, you get "I don't know yet" rather than a confident wrong answer. |
| thoughts are grounded and auditable | A factual answer is not a string of likely words. It is a small program that was run and checked, and you can read the facts and steps it used. |
| self-tunes through CubeLang | Its reasoning steps are written in a language built for checking, so a wrong step is caught by the machine, not by luck. |
| no retraining to gain a skill | A new skill is a program, not a new model. Cubby can write one, have it certified, and keep it. |
| separation of concerns | One part talks, one part reasons, specialists handle specific jobs. When one part learns something new, the others do not forget what they knew. |
| worlds grounded in a global world | Each task gets its own working memory of facts, all anchored to one shared set. What it learns in one place stays consistent everywhere. |
| encrypted chats | What you tell it is stored encrypted on your machine. Nothing is sent anywhere. |
| programmable, agentic, tool-calling via ToolForge | It can decide to use a tool, ask for one to be built when none fits, and every tool it uses is certified before it runs. |
| trained on empathy, in balance | It reads the emotional temperature of what you write and adjusts its tone. It does not pretend to have feelings. |
| chain and graph of thought managed by the machine | Multi-step answers are walked step by step by the verifier. A broken chain stops instead of finishing with a guess. |
| self-improvement from the past and the internet — *coming* | Learning from its own history is built; watching the web for trends and threats is designed, behind an explicit permission, and not yet on. |
| live visual cortex, webcam — *coming* | Seeing through a camera exists in the lab lineage and is not part of what you get today. |

## What can it do for me that other assistants can't?

**It tells you when it doesn't know, and offers to find out.** Ask it something it has no facts for and you get the same
short line every time, in your language, instead of an invented answer. On its own test set it has answered 96% correctly
and been wrong 0%, the rest being "I don't know yet". When you have allowed it to search, the line continues: "I can look
it up for you if you want." Say yes and it searches, checks what it found against what it already knows, keeps the facts
with their source, and answers. The next time anyone asks, it knows, and can say where it learned it. That is how its
knowledge grows: not by guessing, by looking things up with your permission and keeping the receipt.

**It remembers what you tell it, and you can check that it did.** Say "remember that the cottage wifi is bluebird42"
and it is answerable the next moment. Tell it something that contradicts an earlier fact and it points at the earlier one
instead of quietly overwriting it.

**It shows its work.** Every factual answer comes with the facts it used and the steps it ran. If you disagree, you can
see exactly where.

**It stays on your machine, and your conversations stay yours.** No account, no cloud, no usage logs on someone else's
server. What it stores about your chats is encrypted with a key that only exists on your computer.

**It speaks your French.** It was trained on Quebec expressions and on how people here actually text, not only on
textbook French, and it answers in the language you write in.

**It reads the room.** It picks up whether you are frustrated, worried or pleased and adjusts its tone, without
claiming feelings of its own. If you write "nothing works today" you get a calmer, shorter Cubby.

**It learns new skills without being retrained.** Ask for something it has no tool for, a timer, a unit conversion, a
dice roll, and it can ask its own tool factory to build one. The new tool is written as a program, tested by the
verifier, and kept for next time. No update, no download, no waiting for a new version.

**It plays, and you can watch it think.** In cubby-man, its maze game, it learns the map from the moves it is refused,
writes its own shortcut moves, gets them certified, sets traps, gets scared, rests when tired, and narrates what it is
doing in its own words. It is the same brain you chat with.

**It refuses the right things.** It recognises attempts to manipulate it or make it break its rules, and any tool it
could reach is off by default until you turn it on.

**Coming:** ask it "what's in the news about the Canadiens tonight" and it fetches it, through a tool you have
explicitly allowed, with the source attached. The mechanism is built and being trained; the switch is still off.

## What it is not, for now

Today Cubby is not an all-knowing oracle, not a search engine, and not a coding agent that rewrites your files. It is a
small model that thinks big, built to be right or to say so.

## Where it is going

The goal is not to catch up with the big models at what they do. It is to beat them at the things they cannot do right,
and never will, because of how they are built. A model that predicts the next word cannot check itself, cannot reliably
say "I don't know", cannot learn one new fact tonight without being retrained, cannot show you a trace you could audit,
and forgets old skills when it is taught new ones. Those are not bugs to be patched in the next version. They are the
shape of the thing.

Cubby is built the other way round, from the verifier outward: a thought is a program that runs and is checked, a fact is
kept with its source and a signed certificate, a new skill is a program written and certified rather than a model
retrained, and the parts that talk, reason and specialise are separate so that learning in one never erases another.
Every piece above is small today. The bet is that an assistant that is never confidently wrong, that learns while you use
it, and that can show its work, ends up more useful than one that knows more and cannot be trusted with any of it.
