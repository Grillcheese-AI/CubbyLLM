---
name: talk
version: 0.1.0
description: Casual conversation in Cubby's own voice -- greetings, small talk, how it feels, chatting about the day.
examples:
  - hi Cubby, how are you?
  - good morning!
  - how was your day?
  - tell me about yourself
  - what do you like to do?
  - thanks, that was helpful
  - I'm feeling a bit down today
  - can we just chat for a while?
  - what's your favourite colour?
  - bonjour Cubby, ça va?
not:
  - what is the capital of Canada?
  - convert 5 miles to km
  - In Romeo and Juliet, who is Juliet's mother?
  - is there a snowfall warning in Ottawa?
  - when was the Eiffel Tower built?
  - who wrote Pride and Prejudice?
  - how many moons does Mars have?
  - what happened in Ottawa yesterday?
  - who is the prime minister of Canada?
  - what is 20 celsius in fahrenheit?
kinds: [adapter]
audience: public
licence: internal weights (Grillcheese Research Lab)
requires: {plugins: [], base: null}
permissions: {network: [], write: false, gpu: true}
adapter: {role: talk, dir_env: CUBBY_TALK_ADAPTER}
entry: {}
---

# talk

Today's talk adapter, repackaged: the LoRA that gives the 450M its conversational voice, loaded as the `talk` role on
the base when a turn needs it (instead of the `--talk-adapter` flag), and unloaded after it has been idle for
`--plugin-idle` minutes. The adapter's files (`talk_lora.json`, `talk_lora.safetensors`) stay beside the models, not in
the repo: `CUBBY_TALK_ADAPTER` names their folder, and approval pins their hashes too. A layout (rank, alpha, targets)
other than the base's, or a `requires.base` hash naming another export, is refused with the reason.
