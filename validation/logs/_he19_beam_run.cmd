@echo off
rem H-E19 beam: calibrate tau on the emitter's own GSM8K val split, then the 236-question loop at width 1 and with
rem the calibrated tau (baseline adapter, plan-line trie scoring: parity exact for the trie, 8/8 top-1 for the line)
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set AD=H:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_step_cont
set EX=C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly
%PY% validation\exp_he19_beam.py --mode calibrate --to-line 1 --chunk 128 --export %EX% --adapter "%AD%" --tokenizer "%TOK%" --limit 80 --tag dev80 > validation\logs\exp_he19_beam_calibrate_dev80.out 2>&1
%PY% validation\exp_he19_beam.py --mode loop --taus=-inf --tau-from validation\logs\exp_he19_beam_calibrate_dev80.json --to-line 1 --max-steps 10 --chunk 128 --export %EX% --adapter "%AD%" --tokenizer "%TOK%" --tag v1 > validation\logs\exp_he19_beam_loop_v1.out 2>&1
echo done > validation\logs\exp_he19_beam_run.done
