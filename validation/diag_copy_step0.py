"""diag_copy_step0 -- the copy path added to a trained link, before any training (order 0001, C).

Loads a linked checkpoint (``CB_CKPT``, e.g. run 3's ``spatial_link_3_final.pt``) twice: as it is,
and with the link's copy path added the way ``CB_INIT_FROM`` adds it (``train_base.init_copy``:
every tensor loads, the copy keeps its init -- lam 0.1, tau ``CB_SPATIAL_COPY_TAU``), and scores both
with ``train_base.fact_probe`` (the held-out questions of ``CB_PROBE_SHARDS``, teacher-forced, as
``exp_hypertoken_recall`` scores them). What the read already carries, handed to the output with
nothing learned: the floor run 4 starts from.

    CB_CKPT=<linked ckpt> CB_CORPUS=<token cache> CUBBY_SPM=<tokenizer.json> CB_SPATIAL_COPY=ring
    CB_SPATIAL_COPY_TAU=3.04 [CB_PROBE_SHARDS=facts_far1k_lead,spatial_here] [CB_PROBE_DOCS=24]
    [CB_OUT=<json>] python validation/diag_copy_step0.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("CB_PROBE_SHARDS", "facts_far1k_lead,spatial_here")
os.environ.setdefault("CB_PROBE_DOCS", "24")
os.environ.setdefault("CB_BATCH", "4")
import torch  # noqa: E402

import train_base as tb  # noqa: E402


def main() -> None:
    t0, dev = time.time(), torch.device("cpu")
    if not tb.SPATIAL_COPY:
        raise SystemExit("set CB_SPATIAL_COPY (ring | table)")
    ck = torch.load(os.environ["CB_CKPT"], map_location="cpu", weights_only=False, mmap=True)
    meta = dict(ck["meta"])
    del ck
    report = {"ckpt": os.environ["CB_CKPT"], "copy": tb.SPATIAL_COPY, "tau": tb.SPATIAL_COPY_TAU, "docs": tb.PROBE_DOCS,
              "env": f"{time.strftime('%Y-%m-%d %H:%M')} python {platform.python_version()} torch {torch.__version__} "
                     f"{platform.processor() or platform.machine()} (CPU)"}
    for arm in ("as_trained", "copy_added"):
        m = dict(meta, spatial=dict(meta["spatial"], copy=tb.SPATIAL_COPY, copy_tau=tb.SPATIAL_COPY_TAU)) if arm == "copy_added" else meta
        model = tb.build_model(m, dev)
        c = tb.init_from(os.environ["CB_CKPT"], model, dev, m)
        assert arm == "as_trained" or c.get("copy_added"), "the copy path was not added"
        del c
        report[arm] = pr = tb.fact_probe(model, dev)
        print(f"  {arm:10s} " + "  ".join(f"{k}: first {v['first_token']:.3f} exact {v['exact']:.3f} ({v['questions']} q)"
              for k, v in pr.items() if not k.startswith("_")) + (f"  lam {pr['_lam']:.3f}" if pr["_lam"] is not None else "")
              + f"   [{time.time() - t0:.0f}s]", flush=True)
        del model
    if os.environ.get("CB_OUT"):
        json.dump(report, open(os.environ["CB_OUT"], "w", encoding="utf-8"), indent=1)
        print(f"  wrote {os.environ['CB_OUT']}")


if __name__ == "__main__":
    main()
