@echo off
rem H-M1c: the Rubik's-cube encoding (role-filler majority keys, Hamming nearest neighbour) on every table, full store
set PY=%CUBBY_GITHUB%\grilly2\.venv\Scripts\python.exe
set SH=%CUBBY_DRIVE%\cubbyllm\token_cache_base\wiki_full.u32
set TOK=%CUBBY_DRIVE%\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json
set D=%CUBBY_MODELS%
cd /d %CUBBY_REPO%
%PY% validation\exp_hm1c_hamming.py --table %D%\fastword_table_v4.npz --shard "%SH%" --tokenizer "%TOK%" --n 300 --blends 0,0.5 --tag v4 > validation\logs\exp_hm1c_hamming_v4.out 2>&1
%PY% validation\exp_hm1c_hamming.py --table %D%\fastword_table_v5p.npz --shard "%SH%" --tokenizer "%TOK%" --n 300 --blends 0 --tag v5p > validation\logs\exp_hm1c_hamming_v5p.out 2>&1
for %%T in (potion_ar paraminilm_ar paraminilm_ar4) do (
  %PY% validation\exp_hm1c_hamming.py --table %D%\fastword_table_%%T.npz --shard "%SH%" --tokenizer "%TOK%" --n 300 --blends 0 --tag %%T > validation\logs\exp_hm1c_hamming_%%T.out 2>&1
)
:wait
if not exist validation\logs\exp_hm1_teachers2.done ( timeout /t 60 /nobreak > nul & goto wait )
%PY% validation\exp_hm1c_hamming.py --table %D%\fastword_table_v6q.npz --shard "%SH%" --tokenizer "%TOK%" --n 300 --blends 0 --tag v6q > validation\logs\exp_hm1c_hamming_v6q.out 2>&1
echo done > validation\logs\exp_hm1c_hamming.done
