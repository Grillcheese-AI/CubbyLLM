# H-E19 beam: the 236-question loop at width 1 and at the dev-calibrated tau (resumable; run without a cmd window)
$root = 'C:\Users\grill\Documents\GitHub\CubbyLLM'
Set-Location $root
$py = 'C:\Users\grill\Documents\GitHub\grilly2\.venv\Scripts\python.exe'
$args_ = @('validation\exp_he19_beam.py', '--mode', 'loop', '--taus=-inf',
  '--tau-from', 'validation\logs\exp_he19_beam_calibrate_dev80.json', '--to-line', '1', '--max-steps', '10',
  '--chunk', '128', '--export', 'C:\CUBBY-TRAINED-MODELS\base450m_cont\grilly',
  '--adapter', 'H:\My Drive\cubbyllm\emitter\cubby450m_v12e_w_step_cont',
  '--tokenizer', 'H:\My Drive\cubbyllm\token_cache_base\tokenizer\grillcheese_bbpe128k.json', '--tag', 'v1')
$p = Start-Process -FilePath $py -ArgumentList $args_ -NoNewWindow -PassThru -Wait `
  -RedirectStandardOutput "$root\validation\logs\exp_he19_beam_loop_v1.out" -RedirectStandardError "$root\validation\logs\exp_he19_beam_loop_v1.err"
"exit $($p.ExitCode)" | Set-Content "$root\validation\logs\exp_he19_beam_run.done"
