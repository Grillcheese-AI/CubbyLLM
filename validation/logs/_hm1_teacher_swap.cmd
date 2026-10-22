@echo off
rem H-M1 teacher swap: the v4 build with paraphrase-MiniLM-L6-v2 as the teacher (CPU), then the same two screens
set PY=%CUBBY_GITHUB%\grilly2\.venv\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=
set SH=%CUBBY_DRIVE%\cubbyllm\token_cache_base\wiki_full.u32
set TOK=%CUBBY_DRIVE%\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set TAB=%CUBBY_MODELS%\fastword_table_v5p.npz
cd /d %CUBBY_GITHUB%\mowm
%PY% scripts\build_fastword_table.py --shard "%SH%" --teacher st:sentence-transformers/paraphrase-MiniLM-L6-v2 --top-words 60000 --out %TAB% > %CUBBY_MODELS%\fastword_table_v5p.build.out 2>&1
cd /d %CUBBY_REPO%
%PY% validation\exp_hm1_ngram_hv.py --table %TAB% --shard "%SH%" --tokenizer "%TOK%" --tag v5p > validation\logs\exp_hm1_ngram_hv_v5p.out 2>&1
%PY% validation\exp_hm1b_soft_address.py --table %TAB% --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tag v5p_t3m8 > validation\logs\exp_hm1b_soft_address_v5p_t3m8.out 2>&1
%PY% validation\exp_hm1b_soft_address.py --table %TAB% --shard "%SH%" --tokenizer "%TOK%" --blends 0 --tables 26 --tag v5p_t3m8_tab26 > validation\logs\exp_hm1b_soft_address_v5p_t3m8_tab26.out 2>&1
echo done > validation\logs\exp_hm1_teacher_swap.done
