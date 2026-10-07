# Make every training checkpoint loadable with from_pretrained, without copying weights:
# eval_models\stepN\ gets the run's small config/tokenizer files plus a hard link to the
# checkpoint's pytorch_model.bin (same drive, so no extra space and no admin rights).
#   powershell -File make_eval_models.ps1
$run = Join-Path $PSScriptRoot 'parvaaz_train'
$gen = Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--ai4bharat--indic-parler-tts\snapshots\*\generation_config.json" | Select-Object -First 1
Get-ChildItem $run -Directory -Filter 'checkpoint-*' | ForEach-Object {
    $step = ($_.Name -split '-')[1]
    $ev = Join-Path $PSScriptRoot "eval_models\step$step"
    New-Item -ItemType Directory -Force $ev | Out-Null
    foreach ($f in 'config.json', 'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json') {
        Copy-Item (Join-Path $run $f) $ev -Force
    }
    if (-not (Test-Path "$ev\generation_config.json")) { Copy-Item $gen.FullName $ev }
    if (-not (Test-Path "$ev\pytorch_model.bin")) {
        New-Item -ItemType HardLink -Path "$ev\pytorch_model.bin" -Target (Join-Path $_.FullName 'pytorch_model.bin') | Out-Null
    }
    "step$step ready"
}
