# Self-healing wrapper for train_JDD_pytorch.py: relaunches the training process whenever
# it exits (crash, kill, machine sleep/shutdown interrupting it -- observed repeatedly on
# this GPU-less, left-running-for-hours setup) until training_state.json reports the target
# epoch reached. Relies on train_JDD_pytorch.py's own --auto-resume (default on) to pick up
# from the last completed epoch each time -- this script only handles "did it finish, and if
# not, start it again," not the resume logic itself.
#
# Meant to be launched as a detached process (see the Start-Process call used to launch this
# script itself), independent of whatever process-tracking mechanism has been killing the
# plain backgrounded training command directly -- so it keeps retrying even if that mechanism
# fires again.

$workDir = 'C:\Users\douyuhan\Desktop\work\AIDenoise_research\PMRID'
$python = 'D:\softwares\Miniforge3\envs\AIISP_init\python.exe'
$ckpDir = Join-Path $workDir 'models\checkpoints_jdd'
$stateFile = Join-Path $ckpDir 'training_state.json'
$loopLog = Join-Path $ckpDir 'loop.log'
$maxEpochIndex = 14  # --num-epoch 15 -> valid completed-epoch indices are 0..14

New-Item -ItemType Directory -Force -Path $ckpDir | Out-Null

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $loopLog -Value $line
}

function Get-CompletedEpoch {
    if (Test-Path $stateFile) {
        try { return (Get-Content $stateFile -Raw | ConvertFrom-Json).epoch } catch { return -1 }
    }
    return -1
}

Write-Log "=== wrapper loop starting (PID $PID) ==="

while ($true) {
    $completed = Get-CompletedEpoch
    if ($completed -ge $maxEpochIndex) {
        Write-Log "training already complete (epoch $completed) -- exiting loop"
        break
    }

    Write-Log "launching training process (last completed epoch: $completed)..."

    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    $env:NUMEXPR_NUM_THREADS = '4'

    $argList = @(
        '-u', 'train_JDD_pytorch.py',
        '--real-data-dir', 'dataset/SID_Sony_raw',
        '--synth-image-dir', 'dataset/train_JDD',
        '--cpu-threads', '4',
        '--batch-size', '1',
        '--num-epoch', '15',
        '--log-every', '50',
        '--ckp-dir', 'models/checkpoints_jdd'
    )

    $proc = Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $workDir `
        -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $ckpDir 'train_stdout.log') `
        -RedirectStandardError (Join-Path $ckpDir 'train_stderr.log')

    Write-Log "training process exited with code $($proc.ExitCode)"

    $completed = Get-CompletedEpoch
    if ($completed -ge $maxEpochIndex) {
        Write-Log "training complete after this run (epoch $completed) -- exiting loop"
        break
    }

    Write-Log "not complete yet (last completed epoch: $completed) -- waiting 15s before relaunch"
    Start-Sleep -Seconds 15
}

Write-Log "=== wrapper loop finished: ALL DONE ==="
