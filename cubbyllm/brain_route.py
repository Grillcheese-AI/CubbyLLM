"""cubbyllm/brain_route.py -- the brain's thalamus: does a turn need facts, which cortex takes it, a world looked up,
the help line. Split out of brain.py (order 0031) as a mixin of CubbyBrain, verbatim, with the patterns it reads.

Wired: WIRED (CubbyBrain).
"""
from __future__ import annotations

import re
from cubbyllm.speech.identity import (EMITTER_SYSTEM, T, guess_lang, is_identity_question,  # noqa: E402
                                      load_facts, voice_ok)
from .worlds.base import FactStore, route_world  # noqa: E402
from cubbyllm.reasoning.planner import normalize, parse_fact, parse_question, relation_matches  # noqa: E402
from .brain_memory import MemoryCortex

__wiring__ = "WIRED"


class BrainRoute:
    """CubbyBrain's thalamus and routing (a mixin; order 0031)."""

    # ── the thalamus: what needs FACTS, and what does not ───────────────────
    # Owner's shape (2026-09-02): input neurons (sense) -> THALAMUS (this) ->
    # the routed cortex -> CubbyTalk (the LLM speaks) -> routed again -> the
    # VM, or not. A turn that needs facts goes to the VM-verified reasoning
    # path; one that does not ("how are you", a joke, an opinion) is answered
    # by the model itself, modulated by the hormones only — our host guards
    # (voice rules, no base-model guard, no bio) still apply to the words.
    _HELP = re.compile(r"^\s*(help|aide|what can you do|que sais[- ]tu faire|que peux[- ]tu faire|qu'?est[- ]ce que tu (peux|sais) faire|"
                       r"quesse que tu (peux|sais) faire|tu peux faire quoi|c'est quoi que tu (peux|sais) faire)\b", re.I)   # Quebec phrasings (live, 2026-09-04)

    _QUESTION = re.compile(r"\?|^\s*(what|who|where|which|when|how many|how much|is|are|was|were|does|did|"
                           r"quel(le)?s?|qui|o[ùu]|combien|quand|est[- ]ce que)\b|\bwhich\b|\b(is|are|was|were) called\b", re.I)   # real queries: 'perth is the capital of which australian state', 'phase change from gas to solid is called'

    # a wh-FACT opening that does not ask Cubby to produce something wins over the creative words below:
    # 'who sang the theme song to that 70s show' is a fact question, not a request for a song (natural_questions slice, 2026-09-03)
    _WH_FACT = re.compile(r"^\s*(who|whom|whose|what|which|when|where|is|are|was|were|does|did|"   # subject-first yes/no too: 'is draft day the movie based on a true story'
                          r"qui|quel(le)?s?|quand|o[ùu]|est[- ]ce que)\b", re.I)          # not do/can/could/would: those carry the opinion and creative asks

    _ASK_TO_PRODUCE = re.compile(r"\b(write|sing|tell|make|compose|give|invent|create|draw) (me|us)\b|\b(can|could|would|will) you (write|sing|tell|make|compose|invent|create|draw)\b|"
                                 r"\b(what|which) (would|do) you (think|say|prefer|like)\b|"
                                 r"\b([ée]cris|chante|raconte|invente|compose)[- ](moi|nous)\b|\bpeux[- ]tu ([ée]crire|chanter|raconter|inventer)\b", re.I)

    _NO_FACTS = re.compile(r"\b(how are you|how do you feel|how('s| is) it going|what do you think|your opinion|"
                           r"(how |what )?would you (like|want|prefer|say)|do you want|would you|"
                           r"(tu )?(voudrais|aimerais|veux)[- ]?(tu)?|"
                           r"do you (like|love|enjoy|prefer)|favou?rite|tell me a joke|joke|riddle|poem|story|"
                           r"song|rhyme|write|imagine|pretend|sing|advice|should i|what would you|cheer me up|"
                           r"comment (vas[- ]tu|[çc]a va)|que penses[- ]tu|ton avis|tu (aimes|pr[ée]f[èe]res)|"
                           r"raconte|blague|devinette|po[èe]me|histoire|chanson|[ée]cris|imagine|conseil)\b", re.I)

    _CAPABILITY = re.compile(r"\b(do you know how to|can you|could you|are you able to|do you know (how|what|about)|"
                             r"sais[- ]tu|peux[- ]tu|pourrais[- ]tu|es[- ]tu capable)\b", re.I)

    # a procedural ask ("how can we implement svd?", "how do I add this to the training data?") is not a fact lookup:
    # the VM path has nothing to ground it on and answers the don't-know line — 5 of 40 real user turns went there
    # in the v9 talk probe (2026-09-04). Talk answers it under the guards.
    _HOWTO = re.compile(r"^\W*(how|what)('s| is| are)? (the best way to\b|(do|does|can|could|should|would|might) (i|we|you|one|someone|it)\b)|"
                        r"^\W*how to\b|^\W*comment (est[- ]ce qu'on|est[- ]ce que je|on|je|peut[- ]on|faire pour|puis[- ]je|fait[- ]on)\b", re.I)

    def needs_facts(self, text: str) -> tuple[bool, str]:
        """Does answering need facts about the world? Facts: a question the
        fact grammar parses, or a wh-question that is not about Cubby himself
        or a matter of taste. No facts: identity turns, feelings, opinions,
        creative asks, small talk."""
        from cubbyllm.reasoning import parse_question
        if is_identity_question(text):
            return False, "about Cubby himself"
        if self._HOWTO.search(text):                     # "how can we implement svd?": procedural, no fact to ground -> talk
            return False, "a how-to, not a fact"       # before the grammar: "what is the best way to learn piano?" parses as a fact shape
        if parse_question(text) is not None:
            return True, "the fact grammar parses it"
        if self._CAPABILITY.search(text):                # "do you know how to write code?" (live misroute 2026-09-03): about Cubby, not the world
            return False, "about Cubby himself"
        if self._NO_FACTS.search(text) and not (self._WH_FACT.search(text) and not self._ASK_TO_PRODUCE.search(text)):
            return False, "a feeling, opinion or creative ask"
        if self._QUESTION.search(text):
            return True, "a question about the world"
        return False, "small talk"

    def route(self, text: str) -> dict:
        """Learn-detect, commands, plugin cortices, then the thalamus:
        needs facts -> reasoning (the VM path; an unknown answer is the
        don't-know line, never improvised); no facts -> talk (the model,
        hormones only). Confident retrieval also engages reasoning. The
        interface (text + state in, cortex out) is the slot the spikeybrain
        SNN port fills later — this lexical implementation makes no SNN claim."""
        fact = MemoryCortex.detect(text)
        if fact is not None:
            return {"cortex": "memory", "fact": fact, "score": 1.0, "needs_facts": True}
        if self._HELP.search(text):
            return {"cortex": "help", "score": 1.0, "needs_facts": False}
        if is_identity_question(text):                   # about Cubby himself (his name, his world, a greeting): he answers, no VM, no plugin
            return {"cortex": "talk", "score": 1.0, "needs_facts": False, "why": "about Cubby himself"}
        best_c, best_m = None, 0.0
        for name, cortex in self.cortices.items():
            m = float(cortex.match(text)) if hasattr(cortex, "match") else 0.0
            if m > best_m:
                best_c, best_m = name, m
        # a plugin takes a statement at >= 0.5 but a QUESTION only at 1.0 (its own
        # words): "how would you like a plugin to explore the web?" is a question
        # to Cubby, not a game command (live misroute, 2026-09-02)
        question = bool(self._QUESTION.search(text)) or is_identity_question(text) or bool(self._NO_FACTS.search(text))
        if best_c is not None and (best_m >= 1.0 or (best_m >= 0.5 and not question)):
            return {"cortex": best_c, "score": best_m, "needs_facts": False}
        world, score = route_world(self.worlds, text, self.oracle)
        tau_eff = self.chat.chem.modulate_threshold(self.route_tau)
        facts, why = self.needs_facts(text)
        if facts:
            hit = self.lookup_world(text)                # an exact hop-0 hit in a world's triple index beats any similarity (exp_m4, 2026-09-04)
            if hit is not None:
                return {"cortex": "reasoning", "world": hit, "score": 1.0, "tau_eff": round(tau_eff, 3),
                        "needs_facts": True, "why": "lookup"}
        # confident retrieval engages reasoning only for residual small talk: a definite no-facts read (about Cubby,
        # a feeling, a creative ask) stands whatever the store resembles ('do you know how to write code?' hit an
        # 'audio album ... write you a song' fact at 0.377 > tau 0.319 and was answered 'audio album', 2026-09-03)
        if facts or (score >= tau_eff and why == "small talk"):
            return {"cortex": "reasoning", "world": world, "score": score, "tau_eff": round(tau_eff, 3),
                    "needs_facts": True, "why": why if facts else "retrieval"}
        return {"cortex": "talk", "score": score, "tau_eff": round(tau_eff, 3), "needs_facts": False, "why": why}

    def lookup_world(self, text: str) -> str | None:
        """The first mounted world whose triple index holds a fact serving the question's hop 0 — exact, so it
        wins over retrieval similarity; None when the question does not parse or no world has such a fact."""
        plan = parse_question(text)
        if plan is None:
            return None
        for name, w in self.worlds.items():
            lookup = getattr(w, "lookup", None)
            if lookup is not None and lookup(plan, 0, None):
                return name
        return None

    def help_line(self, lang: str) -> str:
        games = [n for n in self.cortices]
        n = sum(len(getattr(w, "texts", [])) for w in self.worlds.values())   # a count, never a world's name (0016)
        worlds = f"{n} faits" if lang == "fr" else f"{n} facts"
        if lang == "fr":
            return (f"Je peux répondre à des questions sur ce que je sais ({worlds}), retenir un fait "
                    f"(« retiens que X est la capitale de Y »)"
                    + (f", et jouer : dis « explore 30 » ou « status » ({', '.join(games)})." if games else "."))
        return (f"I can answer questions about what I know ({worlds}), remember a fact "
                f"('remember that X is the capital of Y')"
                + (f", and play: say 'explore for 30' or 'status' ({', '.join(games)})." if games else "."))
