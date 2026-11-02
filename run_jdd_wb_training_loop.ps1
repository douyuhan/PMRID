# Self-healing wrapper for train_JDD_pytorch.py, Phase 3 Option 5 run (WB'd-raw input
# convention -- see doc_draft/PMRID_JDD_plan_phase3_demosaic_wb.md 方案5 / CLAUDE.md's
# "JDD Phase 3, Option 5" section). Same mechanism as run_jdd_training_loop.ps1 (the
# Phase 1/2 run's launcher, kept as-is and not reused directly so that script's
# documented history stays accurate): relaunches the training process whenever it exits
# (crash, kill, machine sleep/shutdown) until training_state.json reports the target
# epoch reached, relying on train_JDD_pytorch.py's own --auto-resume (default on) to pick
# up from the last completed epoch each time.
#
# Continuation run (epoch 20-34, --auto-resume picks up from epoch_19.ckp): the first
# 20 epochs (this same ckp-dir) showed real/synth denoise loss plateauing by ~epoch 8-9
# while the demosaic loss kept trending down through epoch 18 -- so this continuation
# raises --demosaic-weight 1.0 -> 2.0 to push more of the synthetic-batch gradient
# toward the demosaic head specifically, while leaving --learning-rate/--batch-size
# unchanged (the QAT experiments showed escalating LR alongside other changes backfires,
# so only one knob -- demosaic-weight -- is changed here). training_state.json's
# best_loss/best_epoch (currently epoch 14) carry over automatically, so best-checkpoint
# tracking stays correct across the whole 0-34 run, not just this continuation.
#
# Meant to be launched as a detached process, independent of whatever process-tracking
# mechanism has been killing the plain backgrounded training command directly -- so it
# keeps retrying even if that mechanism fires again.

$workDir = 'C:\Users\douyuhan\Desktop\work\AIDenoise_research\PMRID'
$python = 'D:\softwares\Miniforge3\envs\AIISP_init\python.exe'
$ckpDir = Join-Path $workDir 'models\checkpoints_jdd_wb'
$stateFile = Join-Path $ckpDir 'training_state.json'
$loopLog = Join-Path $ckpDir 'loop.log'
$maxEpochIndex = 34  # --num-epoch 35 -> valid completed-epoch indices are 0..34 (continuation from 0..19)

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
        '--num-epoch', '35',
        '--learning-rate', '1e-5',
        '--demosaic-weight', '2.0',
        '--log-every', '50',
        '--ckp-dir', 'models/checkpoints_jdd_wb'
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
