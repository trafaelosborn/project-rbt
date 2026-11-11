param(
    [Parameter(Mandatory=$true)][ValidateSet('Install','Start','Status','Disable','Remove')][string]$Action,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][ValidatePattern('^ProjectRBT-H1-[A-Za-z0-9-]+$')][string]$TaskName
)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$taskConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$expectedConfig = [IO.Path]::GetFullPath((Join-Path $taskConfig.root 'task_host/config.json'))
if ($configPath -ne $expectedConfig -or $taskConfig.schema -ne 'h1_scheduled_config_v1') {throw 'Invalid H1 host config path/schema'}
$pythonPath = (Get-Command python).Source
$hostScript = Join-Path $repoPath 'scripts/run_h1_scheduled.py'
$actionArguments = '-u "' + $hostScript + '" run --config "' + $configPath + '"'
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.TaskPath -ne '\' -or $existing.Actions.Count -ne 1 -or
        $existing.Actions[0].Execute -ne $pythonPath -or
        $existing.Actions[0].Arguments -ne $actionArguments -or
        $existing.Actions[0].WorkingDirectory -ne $repoPath) {
        throw 'Existing task does not match this experiment; refusing to alter it'
    }
}
switch ($Action) {
    'Install' {
        if ($existing) {throw 'Task already exists; inspect it instead of overwriting it'}
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        $taskAction = New-ScheduledTaskAction -Execute $pythonPath -Argument $actionArguments -WorkingDirectory $repoPath
        $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
        $logon = New-ScheduledTaskTrigger -AtLogOn -User $identity
        $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden -Priority 7
        Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Principal $principal -Trigger @($logon,$repeat) -Settings $settings -Description 'Registered Project RBT H1 experiment; bounded checkpoint recovery; no unattended override of errors or pauses.' | Out-Null
        Export-ScheduledTask -TaskName $TaskName | Set-Content -LiteralPath (Join-Path $taskConfig.root 'task_host/task.xml') -Encoding Unicode
        [ordered]@{installed_utc=[DateTime]::UtcNow.ToString('o');task_name=$TaskName;config=$configPath;user=$identity;logon_type='Interactive';run_level='Limited';time_limit='PT0S';restart_check_minutes=5;restart_at_user_logon=$true;single_instance=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskConfig.root 'task_host/install_receipt.json') -Encoding utf8
    }
    'Start' {if (-not $existing) {throw 'Task is not installed'}; Start-ScheduledTask -TaskName $TaskName}
    'Disable' {if (-not $existing) {throw 'Task is not installed'}; Disable-ScheduledTask -TaskName $TaskName | Out-Null}
    'Remove' {
        if (-not $existing) {throw 'Task is not installed'}
        if ($existing.State -eq 'Running') {throw 'Pause the experiment and wait for its workers to exit before removing its task'}
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        return
    }
}
$task = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName
$hostStatePath = Join-Path $taskConfig.root 'task_host/state.json'
$matrixStatePath = Join-Path $taskConfig.root 'matrix_status.json'
$hostState = if (Test-Path -LiteralPath $hostStatePath) {Get-Content -LiteralPath $hostStatePath -Raw | ConvertFrom-Json} else {$null}
$matrixState = if (Test-Path -LiteralPath $matrixStatePath) {Get-Content -LiteralPath $matrixStatePath -Raw | ConvertFrom-Json} else {$null}
$age = if ($matrixState) {([DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse($matrixState.updated_utc)).TotalSeconds} else {$null}
[ordered]@{checked_utc=[DateTime]::UtcNow.ToString('o');task_name=$TaskName;scheduler_state=[string]$task.State;last_result=$info.LastTaskResult;last_run=$info.LastRunTime.ToString('o');next_run=$info.NextRunTime.ToString('o');config=$configPath;host_state=$hostState.state;host_reason=$hostState.reason;interruption_restarts=$hostState.interruption_restarts;matrix_state=$matrixState.status;matrix_heartbeat_age_seconds=$age;matrix_heartbeat_fresh=($null -ne $age -and $age -lt 15 -and $age -ge 0);scheduler_running=($task.State -eq 'Running')} | ConvertTo-Json
