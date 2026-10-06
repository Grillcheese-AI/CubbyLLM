# H-E19: rerun the tau-1.405 beam with every finished path kept (--dump-paths), dev 80:200 first, then the 236.
# The path scorer's weight is chosen on the dev paths; the 236 paths are read once after. Resumable (jsonl per tag).
$root = 'C:\Users\grill\Documents\GitHub\CubbyLLM'
Set-Location $root
$py = 'C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe'
$common = @('validation\exp_he19_beam.py', '--mode', 'loop', '--taus=1.405', '--to-line', '1', '--max-steps', '10',
  '--chunk', '128', '--export', 'C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly',
  '--adapter', '"H:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_step_cont"',
  '--tokenizer', '"H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json"', '--dump-paths')
foreach ($run in @(@{tag='dev_paths'; extra=@('--dev', '80:200')}, @{tag='v1_paths'; extra=@()})) {
  $t = $run.tag
  $p = Start-Process -FilePath $py -ArgumentList ($common + $run.extra + @('--tag', $t)) -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput "$root\validation\logs\exp_he19_beam_loop_$t.out" -RedirectStandardError "$root\validation\logs\exp_he19_beam_loop_$t.err"
  "exit $($p.ExitCode)" | Set-Content "$root\validation\logs\exp_he19_beam_loop_$t.done"
}

