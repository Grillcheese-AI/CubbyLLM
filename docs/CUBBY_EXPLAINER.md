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
| chain and graph of thought managed by the machine | Multi-step answers are walked step by step by the verifier, in milliseconds, not by writing pages of "thinking". A broken chain stops instead of finishing with a guess. |
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

**It thinks cheaply on hard problems.** Other assistants reason by writing out hundreds of words of "thinking" every
single time you ask, and you pay for every word, in time and in electricity. Cubby writes a short program once, the
verifier runs it in milliseconds, and a program that worked is kept and run again at no cost in words at all: a
shortcut it invented in the maze saves it four moves every time it uses it, and a multi-step lookup that would be a page
of text for another model is a fifty-millisecond walk here. The harder the reasoning, the bigger the difference.

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

## Real use cases

The thread through all of these: Cubby does not come with a fixed set of abilities. It comes with a way of gaining them.
A need shows up in your environment, Cubby writes the program for it, the verifier certifies the program on your own
examples, and from then on it is a skill it keeps. With the right program you can make it do what you need, and the
program is something it learns to write as it lives with you. Each case below says what runs today and what is the next
step, so nobody mistakes a plan for a feature.

**For a student.** Cubby becomes a study partner that never invents a citation. Feed it your course notes and it keeps
them as facts with a source, so "when did the Treaty of Paris get signed" comes back with the date and the page it came
from, and "what does this expression mean" gets an answer in your French, Quebec included. Ask past what the notes cover
and it says it does not know yet, then offers to look it up, and what it finds joins the notes with its source attached.
It quizzes you from the facts it holds, not from things it made up, and it shows every step of a multi-step answer so you
can see where your own reasoning went wrong. *Today: the facts with provenance, the don't-know, the steps, the two
languages. Next: the search-and-learn loop.*

**For medical alerts.** A rule like "flag the sample if the reading is above the threshold" is exactly the kind of small
decision program Cubby already writes and has certified, and every one of those decisions is signed and kept in a ledger
with the reading, the threshold, the verdict and which verifier made the call. An alert therefore never comes from a
guess, a value it never measured cannot be spoken, and a contradiction between two readings stops the chain instead of
picking one. When someone asks for a dosage it has no fact for, the answer is the same short line, not a plausible
number. *Today: certified decision programs, the signed ledger, the contradiction gate, the don't-know. Next: the sensor
plugin. And to be clear, none of this is a medical device until it has been validated as one.*

**For secure environments.** Cubby runs on the machine in the room, with no cloud, no account and no traffic out. What
it stores about conversations is encrypted with a key that never leaves that machine. Every tool it could reach is off by
default and turned on one capability at a time, so a program it writes can compose the tools it has but never grant
itself a new one. Every decision it certifies is signed and logged, which gives an auditor a trail rather than a
transcript, and it recognises attempts to manipulate it into breaking its rules before they reach anything. Reasoning
happens inside a verifier that runs programs, not inside a model that can be talked into things. *Today: local, encrypted
at rest, the signed ledger, the manipulation read, tools off by default. Next: the per-tool capability switches.*

**For video games.** cubby-man is the demonstration you can watch: a character that starts with no map, learns the
maze from the moves it is refused, invents its own shortcut moves and has them certified before it trusts them, sets
traps, gets scared when a ghost is close, rests when it is tired, and says what it is thinking in its own words. Nothing
it learns is scripted, and nothing it learns is thrown away: a move that stops paying is retired, not deleted, so the
character you meet in level seven is shaped by levels one to six. The same brain, dropped into another game, is a
character that remembers you between sessions, has moods that come from its own state, and grows skills across a
playthrough instead of picking from a menu of animations. *Today: all of it, in the maze. Next: the same engine behind a
character in a game that is not the maze.*

**For manufacturing, soldering for instance.** A typical robot cell repeats one fixed path and only learns that
something went wrong after it went wrong. Cubby already reasons in three dimensions: the maze is a stack of levels with
six directions, and it derives where things are, which cells are dead ends and what a path costs, as verified programs
over that space. A soldering path is the same kind of object as one of its maze moves, a program certified before it
runs, and a defect check is the same kind of object as its threshold decisions, a rule that flags a reading, signed and
kept. Add the sensor, and the rule can run before the joint is made rather than after. *Today: the 3D reasoning and the
certified move and decision programs. Next: the vision and sensor cortex, which exists in the lab lineage and is not yet
attached.*

Five very different rooms, one mechanism: see the need, write the program, certify it, keep it. That is what "with the
right program you can make it do whatever you need" means in practice, and why the right program is something Cubby
learns to write for you as it evolves in your environment.

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
