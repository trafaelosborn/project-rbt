$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$preflight = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'preflight.json') -Raw | ConvertFrom-Json
if (-not $preflight.complete -or -not $preflight.all_67_locks_available -or $preflight.pause_files.Count -ne 0) {
    throw 'Restart preflight did not pass.'
}
$stdoutPath = Join-Path $PSScriptRoot 'runner_stdout.log'
$stderrPath = Join-Path $PSScriptRoot 'runner_stderr.log'
$receiptPath = Join-Path $PSScriptRoot 'process_launch.json'
if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath) -or (Test-Path -LiteralPath $receiptPath)) {
    throw 'This launch attempt already has evidence. Use a new attempt directory instead of overwriting it.'
}
$pythonPath = (Get-Command python).Source
$runnerArguments = @('-u', 'scripts/run_h1_ablation_matrix.py', '--output-root', 'data/h1_ablation_production_20261001', '--workers', '2', '--session-proposals', '100', '--session-seconds', '1800')
$runnerProcess = Start-Process -FilePath $pythonPath -ArgumentList $runnerArguments -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$runnerProcess.PriorityClass = 'BelowNormal'
$receipt = [ordered]@{
    runner_pid = $runnerProcess.Id
    launched_utc = [DateTime]::UtcNow.ToString('o')
    executable = $pythonPath
    arguments = $runnerArguments
    priority = 'BelowNormal'
    hidden = $true
    parent_waits_for_runner = $true
    action = 'Resume original production registration after stale heartbeat and absent processes'
    starting_committed_proposals = $preflight.committed_proposal_total
    registration_sha256 = $preflight.registration_sha256
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$receipt | ConvertTo-Json -Depth 5 | Write-Output
$runnerProcess.WaitForExit()
$runnerProcess.Refresh()
[ordered]@{exited_utc=[DateTime]::UtcNow.ToString('o');runner_pid=$runnerProcess.Id;exit_code=$runnerProcess.ExitCode} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'process_exit.json') -Encoding utf8
exit $runnerProcess.ExitCode
