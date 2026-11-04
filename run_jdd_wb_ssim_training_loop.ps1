# Self-healing wrapper for train_JDD_pytorch.py, trying the new --demosaic-ssim-epochs
# schedule (plan_phase1.md's deferred "loss函数升级" item -- see train_JDD_pytorch.py's
# module docstring and models/pytorch_ssim.py). Same mechanism as
# run_jdd_wb_training_loop.ps1 (kept as-is, not reused directly, so that script's own
# documented history stays accurate): relaunches the training process whenever it exits
# (crash, kill, machine sleep/shutdown) until training_state.json reports the target
# epoch reached, relying on train_JDD_pytorch.py's own --auto-resume (default on).
#
# Fresh run, warm-started from models/torch_pretrained.ckp (not epoch_34.ckp) -- isolates
# the SSIM-schedule variable from the demosaic_weight=2.0 continuation tried previously,
# which showed no measurable benefit (see CLAUDE.md's JDD_WB section). --demosaic-weight
# is kept at the original 1.0 for the same reason: only one new variable (the SSIM
# schedule) per run, matching this project's established practice.
#
# --demosaic-ssim-epochs 10 (DMSC_NestUnet's own CHANGE_CRI_EPOCH=10) over --num-epoch 20
# total: first 10 epochs use (1 - SSIM) as the demosaic loss, last 10 switch to plain L1 --
# matched in length to the first (pre-continuation) JDD_WB run's 20 epochs, for a fair
# before/after comparison against that run's epoch_best_14 (PSNR 32.50 / SSIM 0.903).
#
# Meant to be launched as a detached process, independent of whatever process-tracking
# mechanism has been killing the plain backgrounded training command directly -- so it
# keeps retrying even if that mechanism fires again.

$workDir = 'C:\Users\douyuhan\Desktop\work\AIDenoise_research\PMRID'
$python = 'D:\softwares\Miniforge3\envs\AIISP_init\python.exe'
$ckpDir = Join-Path $workDir 'models\checkpoints_jdd_wb_ssim'
$stateFile = Join-Path $ckpDir 'training_state.json'
$loopLog = Join-Path $ckpDir 'loop.log'
$maxEpochIndex = 19  # --num-epoch 20 -> valid completed-epoch indices are 0..19

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
        '--init-ckp', 'models/torch_pretrained.ckp',
        '--cpu-threads', '4',
        '--batch-size', '1',
        '--num-epoch', '20',
        '--learning-rate', '1e-5',
        '--demosaic-weight', '1.0',
        '--demosaic-ssim-epochs', '10',
        '--log-every', '50',
        '--ckp-dir', 'models/checkpoints_jdd_wb_ssim'
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
