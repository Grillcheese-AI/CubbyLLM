#!/bin/bash
# Scratch (untracked): order 0043's live measurements after the two refinements, one after another.
cd /c/Users/grill/Documents/GitHub/CubbyLLM || exit 1
export CB_SOURCE_NAMESAKES=1 PYTHONIOENCODING=utf-8
P=.venv-dml/Scripts/python
rm -f validation/logs/_chain_0043c.done validation/logs/_chain_0043c.log
$P validation/logs/_diag_0043_regress.py > validation/logs/_diag_0043_regress2.out 2>&1
echo "regress $?" >> validation/logs/_chain_0043c.log
$P validation/exp_everyday_webq.py --parsed-only --online --tense on --out validation/logs/everyday_webq_on_live_tense_0043_2026-10-27.json > validation/logs/_webq_0043b.out 2>&1
echo "webq $?" >> validation/logs/_chain_0043c.log
$P validation/exp_everyday_webq.py --stack --online --out validation/logs/0043_stack_live_2026-10-27.json > validation/logs/_stack_0043b.out 2>&1
echo "stack $?" >> validation/logs/_chain_0043c.log
touch validation/logs/_chain_0043c.done
