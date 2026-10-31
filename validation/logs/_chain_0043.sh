#!/bin/bash
# Scratch (untracked): order 0043's live measurements, one after another (the source's cache and link memory are shared).
cd /c/Users/grill/Documents/GitHub/CubbyLLM || exit 1
export CB_SOURCE_NAMESAKES=1 PYTHONIOENCODING=utf-8
P=.venv-dml/Scripts/python
rm -f validation/logs/_chain_0043.done
$P validation/audit_source_namesakes.py --n 50 > validation/logs/_audit_0043.out 2>&1
echo "audit $?" >> validation/logs/_chain_0043.log
$P validation/exp_everyday_webq.py --parsed-only --online --tense on --out validation/logs/everyday_webq_on_live_tense_0043_2026-10-27.json > validation/logs/_webq_0043.out 2>&1
echo "webq $?" >> validation/logs/_chain_0043.log
$P validation/exp_everyday_webq.py --stack --online --out validation/logs/0043_stack_live_2026-10-27.json > validation/logs/_stack_0043.out 2>&1
echo "stack $?" >> validation/logs/_chain_0043.log
DB=$($P -c "from cubbyllm.worlds.hdc_store import DEFAULT; print(DEFAULT)")
$P validation/exp_hdc_bench.py --src E:/datasets/hdc --db "$DB" --out validation/logs/exp_hdc_bench_2026-10-27_0043.json > validation/logs/_bench_0043.out 2>&1
echo "bench $?" >> validation/logs/_chain_0043.log
$P validation/namesake_probe.py --tag _0043 > validation/logs/_probe_0043.out 2>&1
echo "probe $?" >> validation/logs/_chain_0043.log
CB_SOURCE_NAMESAKES=0 $P validation/exp_everyday_webq.py --parsed-only --online --tense on --out validation/logs/_everyday_webq_on_live_tense_0043_off.json > validation/logs/_webq_0043_off.out 2>&1
echo "webq_off $?" >> validation/logs/_chain_0043.log
touch validation/logs/_chain_0043.done
