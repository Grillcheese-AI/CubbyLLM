@echo off
cd /d C:\Users\grill\Documents\GitHub\CubbyLLM
set PY=C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe
set TAB=C:\CUBBY-TRAINED-MODELS\fastword_table_v4.npz
set SH=G:\My Drive\cubbyllm\token_cache_base\wiki_full.u32
set TOK=H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
%PY% validation\exp_hm1b_soft_address.py --table %TAB% --shard "%SH%" --tokenizer "%TOK%" --t 3 --m 8 --tag v4_t3m8 > validation\logs\exp_hm1b_soft_address_v4_t3m8.out 2>&1
%PY% validation\exp_hm1b_soft_address.py --table %TAB% --shard "%SH%" --tokenizer "%TOK%" --t 8 --m 16 --batch 250 --tag v4_t8m16 > validation\logs\exp_hm1b_soft_address_v4_t8m16.out 2>&1
echo done > validation\logs\exp_hm1b_soft.done
