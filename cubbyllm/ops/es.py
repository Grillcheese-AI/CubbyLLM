"""Evolution strategies over a LoRA adapter, through grilly.

Wired: WIRED — `validation/exp_v9_es.py` (H-A9) tunes the program adapter
through this, and nothing else in the package reaches grilly's ES or its
LFM2 support.

**Why ES and not a gradient.** The reward is the cubelang VM's verdict on
an emitted program: it parses or it does not, it executes or it does not,
the number it returns matches the gold or it does not. None of that is
differentiable, so the RLVR rung in TODO.md needs a gradient-free
optimizer, and ES is the one the plan already named as the alternative to
GRPO. It is also forward-only, which is what makes it fit on one consumer
card beside a 2.6B base.

**What moves.** Only the LoRA factors. The base is int8 and frozen, and the
adapter is *not merged into it* — merging would mean dequantizing a 2.6B
weight to add a delta, and would leave nothing for ES to perturb. At r=64
the factors are 80.5M values, 2.72% of the base, which is the size of
search space ES is good at.

Everything here is a thin call into `grilly.learning.es`,
`grilly.infer.lora` and `grilly.huggingface`; the mechanisms live there and
work with CubbyLLM uninstalled. grilly is imported inside the functions, so
importing `cubbyllm.ops` still works on a machine that has only torch.
"""
from __future__ import annotations

from ..core.protocols import Wiring

__all__ = ["available", "load_emitter", "evolution_strategy", "greedy"]


def available() -> bool:
    """Whether this backend has ES and LFM2 at all."""
    try:
        import grilly.learning.es  # noqa: F401
        from grilly.infer.models import Lfm2ForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


def _require():
    if not available():
        raise RuntimeError(
            "grilly.learning.es and grilly.infer.models.Lfm2ForCausalLM are needed "
            "for the ES loop; this interpreter has neither. Run it where grilly2 "
            "is importable."
        )


def load_emitter(model_dir: str, adapter_dir: str | None = None, *, int8: bool = True):
    """The emitter: an LFM2 base, quantized, with its LoRA adapter attached.

    Returns ``(model, tokenizer, factors)`` where ``factors`` are the
    adapter's tensors — what :func:`evolution_strategy` moves. With no
    adapter, ``factors`` is empty and the model is the plain base.

    The adapter goes on **unmerged**, so the base stays int8 and the
    factors stay addressable.
    """
    _require()
    import grilly
    from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer, load_lora_adapter
    from grilly.infer.lora import lora_parameters
    from grilly.infer.quantization import Int8WeightOnlyConfig

    grilly.set_grad_enabled(False)          # inference: the state is updated in place
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=Int8WeightOnlyConfig() if int8 else None)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_dir)
    factors: list = []
    if adapter_dir is not None:
        load_lora_adapter(model, adapter_dir)
        factors = lora_parameters(model)
    return model, tokenizer, factors


def evolution_strategy(factors, *, sigma: float = 0.01, lr: float = 0.02,
                       population: int = 32, seed: int = 0):
    """An ES optimizer over ``factors``.

    Population 32 as 16 antithetic pairs: the pairing removes the
    estimator's first-order noise, so a component of the perturbation that
    flatters one member and hurts its twin cancels instead of being read as
    signal.
    """
    _require()
    from grilly.learning import EvolutionStrategy

    return EvolutionStrategy(factors, sigma=sigma, lr=lr, population=population,
                             antithetic=True, seed=seed)


#: LFM2.5's chat template, rendered as `standin/emitter.py::render_chatml`
#: does. The **prefill matters**: this model opens a `<think>` block on its
#: own, so without an empty one the whole generation is reasoning that the
#: harness then strips, leaving nothing to score. That failure is silent —
#: it reads as a fitness of zero, not as an error, which is how it survived
#: a first smoke run here.
NO_THINK_PREFILL = "<think>\n</think>\n"
STOP = "<|im_end|>"


def render_chatml(system, user: str, prefill: str = NO_THINK_PREFILL) -> str:
    """ChatML as LFM2.5 was trained on it. The BOS token is the
    tokenizer's job, not this function's."""
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefill}"


def greedy(model, tokenizer, prompt: str, max_new_tokens: int = 320,
           system=None) -> str:
    """One greedy continuation, in the format the emitter was trained on.

    Greedy because the fitness has to be a property of the *weights*, not
    of a sampling draw — two members judged on different samples are not
    comparable, and ES ranks members against each other.
    """
    _require()
    import grilly
    import numpy as np

    text = render_chatml(system, prompt)
    ids = grilly.from_numpy(np.asarray([tokenizer.encode(text)], dtype=np.int64))
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    produced = out[0].detach().numpy().tolist()[ids.shape[1]:]
    return tokenizer.decode(produced, skip_special_tokens=False).split(STOP)[0]


def graph_stats(step):
    """Capture counters, when the step is wrapped. See ``ops.graph``."""
    from .graph import graph_stats as stats

    return stats(step)


__wiring__ = Wiring.WIRED
