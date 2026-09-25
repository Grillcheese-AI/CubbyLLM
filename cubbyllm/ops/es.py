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

__all__ = ["available", "load_emitter", "evolution_strategy", "optimizer", "update",
           "greedy", "greedy_many", "population_greedy", "population_greedy_many"]


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
    import pathlib as _pl
    if adapter_dir is not None and "merged" in _pl.Path(str(model_dir)).name.lower():
        # the adapter is attached unmerged, so on a checkpoint it was already merged into it counts twice
        raise ValueError(f"{model_dir} looks like a merged checkpoint; attach the adapter to the base it "
                         "was trained on instead (adapter_config.json: base_model_name_or_path)")
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


def evolution_strategy(factors, *, sigma: float = 0.003, lr: float = 3e-6,
                       population: int = 32, seed: int = 0,
                       shaping: str = "prompt-centred"):
    """An ES optimizer over ``factors``.

    Population 32 as 16 antithetic pairs: the pairing removes the
    estimator's first-order noise, so a component of the perturbation that
    flatters one member and hurts its twin cancels instead of being read as
    signal.

    **sigma is measured, not copied.** EGGROLL's frozen-base LoRA run
    (Sarkar et al., 2026, Table 4) uses 0.01, but sigma only means anything
    against the scale of what it perturbs, and this adapter's factors are
    not that paper's. Measured on v14e: ``std(lora_A) = 1.33e-2`` but
    ``std(lora_B) = 1.72e-3``, so 0.01 perturbs B by **5.8x its own
    standard deviation**. A member that far out is not a neighbour of the
    trained adapter, it is a different model — and a different model does
    not emit ``<|im_end|>``. One non-terminating row holds the whole batch
    open, because grilly's compaction can only drop rows that finish:

        sigma   sigma/std(B)  terminated  steps  seconds  distinct outputs
        0.010       5.83         31/32     768    150.7        32
        0.006       3.50         32/32     177     34.4        28
        0.004       2.33         32/32     160     31.4        26
        0.003       1.75         32/32      99     20.0        22
        0.002       1.17         32/32      95     19.2        16
        0.001       0.58         32/32      93     18.8         5
        0.0005      0.29         32/32      88     17.9         1

    (32 members, one prompt, max_new_tokens 768, RX 6750 XT, 2026-09-24.)

    Both ends fail, differently. Above the knee a single hung member costs
    the batch 7.5x; below it every member decodes the *same* greedy tokens,
    fitness is identical across the population, and ES takes no step at
    all. 0.003 is the knee on both axes — it also maximises distinct
    outputs per second.

    ``lr`` is **not** in units of sigma: the step is set by ``lr/sigma``,
    the paper's ``alpha``, which is 0.001 here as it is in its Tables 3 and
    4. **Change one and you must change the other**; the 0.02 this once
    carried was an alpha of 2.0, a measured per-element step of 7.4 sigma,
    i.e. the update leaving the neighbourhood the fitness was measured in
    on the very first iteration.

    ``shaping`` is ``"prompt-centred"`` — EGGROLL §6.3, what the paper
    uses for its reasoning fine-tunes. It is linear in the score where
    ``"ranks"`` is linear in the rank, so a member that solved the whole
    batch counts for more than one that edged out a tie by 0.001; on the
    ladder here, that difference is real information and ranks throw it
    away. It needs the **per-prompt** scores rather than their mean, so
    :meth:`step` takes a ``(population, prompts)`` matrix.
    """
    _require()
    from grilly.learning import EvolutionStrategy

    return EvolutionStrategy(factors, sigma=sigma, lr=lr, population=population,
                             antithetic=True, seed=seed, shaping=shaping)


def optimizer(factors, kind: str = "sgd", lr: float = 1e-4):
    """The optimizer that takes the ES estimate, or ``None`` for plain ES.

    ``None`` (``kind="sgd"``) is EGGROLL's own choice for the run closest
    to this one — Countdown, Table 10, "Optimiser: Gradient descent" —
    and it is the cheap one: :meth:`EvolutionStrategy.step` folds the
    estimate into the weights in one pass and never stores a gradient.

    ``"adam"`` / ``"adamw"`` is what the paper uses for its RL sweeps
    (Tables 7-9) and its quantised distillation (§K.2). Momentum is worth
    more against an ES estimate than against a real gradient, because a
    population of 32 is mostly noise and the moment estimate is what
    averages it across iterations. **It costs memory**: ``.grad`` plus two
    moment buffers is three copies of the factors, ~966 MB at r=64 over
    this 2.6B base, on top of a 3.7 GB model and the KV cache.

    ``lr`` here is **not** the ES ``lr`` and the two cannot share a sweep.
    Adam normalises by its own second moment, so its step is about ``lr``
    per coordinate whatever the estimate's scale; the ES ``lr`` is a
    multiplier on an estimate whose scale is ``1/sigma``. 1e-4 is a
    starting point, not a measured one.
    """
    if kind == "sgd":
        return None
    _require()
    import grilly

    if kind not in ("adam", "adamw"):
        raise ValueError(f"optimizer must be sgd, adam or adamw; got {kind!r}")
    cls = grilly.optim.Adam if kind == "adam" else grilly.optim.AdamW
    return cls(factors, lr=lr)


def update(es, opt, scores) -> None:
    """One ES iteration's update, through ``opt`` or straight into the
    weights when it is ``None``.

    Which path runs has to be decided in one place: they advance the same
    noise stream, so doing both in an iteration applies one step's
    perturbation twice and the second one against the wrong member.
    """
    if opt is None:
        es.step(scores)
        return
    opt.zero_grad()
    es.gradient(scores)
    opt.step()


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
    tokens = tokenizer.encode(text)
    # The chat template does not carry BOS and this tokenizer's `encode`
    # does not add one, but llama.cpp does and the emitter was trained that
    # way. Leaving it off is a different prefix, which costs accuracy
    # without ever looking like an error: measured 0.625 verify-to-gold
    # against 0.800 for the same weights through llama.cpp.
    bos = tokenizer.bos_token_id
    if bos is not None and (not tokens or tokens[0] != bos):
        tokens = [bos] + list(tokens)
    ids = grilly.from_numpy(np.asarray([tokens], dtype=np.int64))
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    produced = out[0].detach().numpy().tolist()[ids.shape[1]:]
    return tokenizer.decode(produced, skip_special_tokens=False).split(STOP)[0]


def _prompt_ids(tokenizer, prompt: str, system=None) -> list:
    """The prompt as the emitter was trained on it: ChatML with the empty think block, BOS first."""
    tokens = tokenizer.encode(render_chatml(system, prompt))
    bos = tokenizer.bos_token_id
    if bos is not None and (not tokens or tokens[0] != bos):
        tokens = [bos] + list(tokens)
    return list(tokens)


def population_greedy(model, tokenizer, es, prompt: str, max_new_tokens: int = 320,
                      system=None) -> list:
    """Every ES member's greedy continuation of ONE prompt, as one batch.

    ``grilly.infer.lora.population`` makes each batch row apply its own
    member's perturbation against the shared base, so the whole population
    decodes together: one generate over ``population`` identical rows
    instead of ``population`` serial ones. Identical rows need no padding;
    :func:`population_greedy_many` batches several prompts, left-padded.
    Returns the texts in member order.

    The same arithmetic as :func:`greedy` under ``es.member(m)`` up to float
    reassociation (``x(A + E)`` against ``xA + xE``), so a greedy decode
    can part from the serial one at a near-tie; the fitness is a property
    of whichever path measures it, and a run uses one path throughout.
    """
    _require()
    import grilly
    import numpy as np
    from grilly.infer.lora import population

    tokens = _prompt_ids(tokenizer, prompt, system)
    ids = grilly.from_numpy(np.asarray([tokens] * es.population, dtype=np.int64))
    with population(model, es, repeat=1):
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    rows = out.detach().numpy().tolist()
    return [tokenizer.decode(r[len(tokens):], skip_special_tokens=False).split(STOP)[0] for r in rows]


def _left_padded(rows: list, pad: int) -> tuple:
    """Token rows as one left-padded batch: ``(ids, attention_mask, width)``."""
    width = max(len(r) for r in rows)
    ids = [[pad] * (width - len(r)) + list(r) for r in rows]
    mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
    return ids, mask, width


def _pad_id(tokenizer) -> int:
    """Any real token id does for the prompt's padding (it is masked out);
    the tokenizer's own pad id when it has one."""
    pad = getattr(tokenizer, "pad_token_id", None)
    return int(pad) if pad is not None else 0


def _generate_padded(model, rows: list, pad: int, max_new_tokens: int) -> tuple:
    import grilly
    import numpy as np

    ids, mask, width = _left_padded(rows, pad)
    out = model.generate(grilly.from_numpy(np.asarray(ids, dtype=np.int64)),
                         attention_mask=grilly.from_numpy(np.asarray(mask, dtype=np.int64)),
                         max_new_tokens=max_new_tokens, do_sample=False)
    return out.detach().numpy().tolist(), width


def greedy_many(model, tokenizer, prompts: list, max_new_tokens: int = 320, systems=None,
                batch: int = 16) -> list:
    """:func:`greedy` over several prompts, ``batch`` at a time: the prompts
    go in left-padded, and grilly drops each row as it reaches the stop
    token, so a batch costs its longest answer, not ``batch`` answers.

    The same arithmetic as :func:`greedy` up to rounding (a padded row's
    positions are shifted, which a rotary score does not see), so a greedy
    decode can part from the one-at-a-time one at a near-tie; a run uses
    one path throughout."""
    _require()
    systems = list(systems) if systems is not None else [None] * len(prompts)
    rows = [_prompt_ids(tokenizer, p, s) for p, s in zip(prompts, systems)]
    pad, texts = _pad_id(tokenizer), []
    for start in range(0, len(rows), batch):
        out, width = _generate_padded(model, rows[start:start + batch], pad, max_new_tokens)
        texts.extend(tokenizer.decode(r[width:], skip_special_tokens=False).split(STOP)[0] for r in out)
    return texts


def population_greedy_many(model, tokenizer, es, prompts: list, max_new_tokens: int = 320,
                           systems=None, per_batch: int = 4) -> list:
    """Every ES member's greedy continuation of every prompt:
    ``texts[member][prompt]``.

    ``per_batch`` prompts go into one generation, every member a row per
    prompt — ``population * per_batch`` rows, member-major, which is the
    layout ``population(model, es, repeat=per_batch)`` gives the noise. The
    prompts are left-padded, and grilly drops rows as they reach the stop
    token (each LoRA drops the same rows of its noise), so a batch costs
    its longest answer.

    What bounds ``per_batch`` is memory: the prompt runs as one forward
    over ``population * per_batch`` rows, and the KV cache holds all of
    them at full length until rows finish."""
    _require()
    from grilly.infer.lora import population

    systems = list(systems) if systems is not None else [None] * len(prompts)
    rows = [_prompt_ids(tokenizer, p, s) for p, s in zip(prompts, systems)]
    pad, n = _pad_id(tokenizer), es.population
    texts = [[None] * len(rows) for _ in range(n)]
    for start in range(0, len(rows), per_batch):
        chunk = rows[start:start + per_batch]
        k = len(chunk)
        with population(model, es, repeat=k):
            out, width = _generate_padded(model, [chunk[j] for _ in range(n) for j in range(k)],
                                          pad, max_new_tokens)
        for m in range(n):
            for j in range(k):
                row = out[m * k + j][width:]
                texts[m][start + j] = tokenizer.decode(row, skip_special_tokens=False).split(STOP)[0]
    return texts


def graph_stats(step):
    """Capture counters, when the step is wrapped. See ``ops.graph``."""
    from .graph import graph_stats as stats

    return stats(step)


__wiring__ = Wiring.WIRED
