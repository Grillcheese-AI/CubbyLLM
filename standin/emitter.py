"""emitter — the trunk-facing interface the stand-in sits behind (a facade since order 0030: the code is in
emitter_base / emitter_llama / emitter_450m / emitter_talk; every name is re-exported here, so importers do not
change).

Wired: STANDALONE (stand-in; nothing in cubbyllm/ imports this — guard-pinned).

`Emitter` is the contract: text in, CubeLang program out. Three implementations:

  LlamaCppEmitter     in-process GGUF via llama-cpp-python (installed here with
                      the Vulkan backend bundled — it finds the RX 6750 XT);
                      the default for the local VM eval:
                        python -c "from standin.emitter import LlamaCppEmitter as E; print(E('emitter-q4_k_m.gguf').emit('What is 3 plus 4?'))"
  LlamaServerEmitter  HTTP against an OpenAI-compatible server — either a real
                      llama.cpp `llama-server` (not installed on this machine) or
                      the bindings' own:  python -m llama_cpp.server --model emitter-q4_k_m.gguf --n_gpu_layers -1 --port 8080
  ReplayEmitter       replays recorded generations (the Colab notebook's val_generations.json)

The 2B CubbyLLM trunk replaces the stand-in by implementing the same protocol
(a `CubbyEmitter` over `cubbyllm.core.generation`) — nothing above this
interface changes. No third-party dependency: urllib only.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                                # imported as standin.emitter: its parts sit beside it
    sys.path.append(_HERE)

from emitter_base import (_THINK_RE, GPU_LOCK, SYSTEM, Emitter, ReplayEmitter, clean_reply,  # noqa: E402,F401
                          strip_fences, strip_think)
from emitter_llama import (CHAT_FAMILIES, NO_THINK_PREFILL, NON_LATIN, LlamaCppEmitter,  # noqa: E402,F401
                           LlamaServerEmitter, ScriptBan, chat_family, render_chatml, render_gemma)
from emitter_450m import Cubby450mEmitter, Cubby450mTrunk  # noqa: E402,F401
from emitter_talk import (TALK_DASH, TALK_UNLOCK, ContextualEmitter, Cubby450mTalk, context_role,  # noqa: E402,F401
                          talk_answer, talk_prompt)

__wiring__ = "STANDALONE"
