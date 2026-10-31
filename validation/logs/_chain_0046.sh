#!/bin/bash
# Scratch (untracked): order 0046's R1 (dev), R0, then R1's one read of test -- one after another (shared source cache).
cd /c/Users/grill/Documents/GitHub/CubbyLLM || exit 1
export PYTHONIOENCODING=utf-8
P=.venv-dml/Scripts/python
rm -f validation/logs/_chain_0046.done validation/logs/_chain_0046.log
$P validation/retrieval_bench/r1_finding.py --split dev --n 500 --n-mintaka 1000 --workers 6 --api-threads 3 > validation/logs/_r1_dev.out 2>&1
echo "r1_dev $?" >> validation/logs/_chain_0046.log
$P validation/retrieval_bench/r0_denominator.py --n 300 --workers 3 > validation/logs/_r0.out 2>&1
echo "r0 $?" >> validation/logs/_chain_0046.log
$P validation/retrieval_bench/r1_finding.py --split test --n 500 --n-mintaka 1000 --workers 6 --api-threads 3 > validation/logs/_r1_test.out 2>&1
echo "r1_test $?" >> validation/logs/_chain_0046.log
touch validation/logs/_chain_0046.done
