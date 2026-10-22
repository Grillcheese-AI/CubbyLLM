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
  - hey there!
  - good evening Cubby
  - how's it going?
  - what have you been up to?
  - I had a rough day at work
  - do you ever get bored?
  - tell me something fun about you
  - I just wanted to say thank you
  - what makes you happy?
  - I'm so excited about my trip next week!
  - can you keep me company for a bit?
  - what's on your mind today?
  - good night, talk tomorrow
  - I'm a little nervous about my exam
  - salut Cubby !
  - bonsoir, comment vas-tu ?
  - j'ai passé une mauvaise journée
  - merci beaucoup, t'es gentil
  - on peut jaser un peu ?
  - qu'est-ce que tu aimes faire ?
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
  - how are the Canadiens doing this season?
  - what's the weather like in Ottawa?
  - how do I say good morning in Spanish?
  - who is Cubby the bear in the cartoon?
  - how was the stock market today?
  - tell me about the history of Montreal
  - what is the colour of the sky on Mars?
  - how many days until Christmas?
  - what happened in the news today?
  - who sings 'How Are You'?
  - what is the best way to learn French?
  - how do I calm down before an exam?
  - what's 15 percent of 80?
  - is there a storm warning in Halifax?
  - quelle heure est-il à Paris ?
  - qui a gagné la coupe Stanley en 1993 ?
  - combien de kilomètres entre Lévis et Québec ?
  - quelle est la météo à Montréal ?
  - dans Hamlet, qui est la mère d'Hamlet ?
  - qui est le premier ministre du Québec ?
kinds: [adapter]
audience: public
licence: internal weights (Grillcheese Research Lab)
requires: {plugins: [], base: 695f05b9f99790e355bbc7935d20939cb9ab5df732085a75f5cc551610db737c}
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
