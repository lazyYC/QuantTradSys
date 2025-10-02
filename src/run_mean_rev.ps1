$workers     = 4
$studyName   = "mean_rev_sep27"
$storageUri  = "sqlite:///storage/optuna_mean_rev.db"
$workingDir  = "C:\Users\YCL\QuantTradSys"
$logDir      = "storage\\logs\\optuna_workers"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$env:PYTHONPATH = 'src'

$baseArgs = @(
    "scripts/pipeline_run_mean_rev.py",
    "--symbol", "BTC/USDT",
    "--timeframe", "5m",
    "--lookback-days", "400",
    "--n-trials", "5000",
    "--study-name", $studyName,
    "--storage", $storageUri,
    "--n-jobs", "2"
)

for ($i = 1; $i -le $workers; $i++) {
    $stdout = Join-Path $logDir ("worker_${i}.out.log")
    $stderr = Join-Path $logDir ("worker_${i}.err.log")
    Start-Process -FilePath "python.exe" -ArgumentList $baseArgs -WorkingDirectory $workingDir -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Start-Sleep -Seconds 1
}
