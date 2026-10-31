@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set AD=H:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_step_cont
set HE19_VERBOSE=1
%PY% validation\exp_he19_beam.py --mode loop --taus=-inf --to-line 1 --max-steps 10 --chunk 128 --export C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly --adapter "%AD%" --tokenizer "%TOK%" --limit 2 --tag smoke > validation\logs\exp_he19_beam_smoke_loop.out 2>&1
echo done > validation\logs\exp_he19_beam_smoke.done
