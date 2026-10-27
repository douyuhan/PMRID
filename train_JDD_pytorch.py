#!/usr/bin/env python3
"""JDD = Joint Denoising and Demosaicing. Fine-tunes models/net_torch.py::NetworkDemosaic
(denoise + demosaic, shared trunk -- i.e. a single network trained to do both jobs at
once, not a demosaic-only network) from an existing denoise-only checkpoint (default
models/torch_pretrained.ckp).

Phase 1 scope (see the plan this was built from): validate that the architecture and
training pipeline work end-to-end, not final quality. Two complementary data sources,
alternated every step, same as discussed in the design:
  - --real-data-dir: dataset/training_pytorch.py::CleanRawImages over real sensor raw
    (e.g. dataset/SID_Sony_raw) -- physically correct noise synthesis, denoise loss
    only (no real demosaic ground truth exists for real raw at this phase).
  - --synth-image-dir: dataset/synthetic_mosaic.py::SyntheticMosaicImages over ordinary
    sRGB photos (e.g. Kodak24/McMaster) -- approximate noise synthesis (inverse-gamma
    linearization is only an approximation of the real sensor domain), but an exact
    demosaic ground truth (the source image itself), so both the denoise and demosaic
    losses are computed on these batches.

Shares add_data_aug_args/build_data_aug_options/get_loss_l1 with train_pytorch.py (same
Reno 10x sensor-calibration defaults), and the best-checkpoint tracking pattern
(epoch_best_{epoch}.ckp, previous best deleted) from train_qat_pytorch.py.
"""
import argparse
import json
import platform
from itertools import cycle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.synthetic_mosaic import SyntheticMosaicImages
from dataset.training_pytorch import CleanRawImages, DataAug
from models.net_torch import NetworkDemosaic
from train_pytorch import add_data_aug_args, build_data_aug_options, get_loss_l1

# Windows priority-class constants (kernel32 SetPriorityClass) -- used by --low-priority
# below so a long, unattended CPU-only training run doesn't compete with the user's own
# foreground work on this no-GPU machine. Not meaningful on non-Windows platforms.
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IDLE_PRIORITY_CLASS = 0x00000040

# Written/updated after every epoch, alongside the regular epoch_{N}.ckp checkpoints, so an
# interrupted run (e.g. the machine sleeping overnight -- observed twice in practice on this
# GPU-less, left-running-for-hours setup) can pick back up without redoing already-completed
# epochs. Deliberately does NOT persist optimizer state (Adam's moment estimates reset on
# resume) -- an acceptable simplification given the small LR and thousands of steps per epoch
# make the resulting transient negligible and self-correcting within a few steps.
TRAINING_STATE_FILENAME = 'training_state.json'


def load_resume_state(ckp_dir: Path):
    """If ckp_dir has a training_state.json from a previous (interrupted) run of this
    script, return (net, start_epoch, best_loss, best_epoch) to continue from; else None."""
    state_path = ckp_dir / TRAINING_STATE_FILENAME
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    ckp_path = ckp_dir / f"epoch_{state['epoch']}.ckp"
    if not ckp_path.exists():
        print(f"warning: {state_path} refers to epoch {state['epoch']} but {ckp_path} is "
              f"missing -- ignoring resume state and starting fresh from --init-ckp")
        return None
    net = NetworkDemosaic()
    net.load_state_dict(torch.load(str(ckp_path), map_location='cpu'))
    print(f"resuming from {ckp_path} (epoch {state['epoch']} complete, "
          f"best_loss={state['best_loss']:.4f} at epoch {state['best_epoch']})")
    return net, state['epoch'] + 1, state['best_loss'], state['best_epoch']


def save_resume_state(ckp_dir: Path, epoch: int, best_loss: float, best_epoch):
    state_path = ckp_dir / TRAINING_STATE_FILENAME
    state_path.write_text(json.dumps({'epoch': epoch, 'best_loss': best_loss, 'best_epoch': best_epoch}))


# Append-only per-epoch loss history, separate from training_state.json (which only ever
# holds the *current* resume point, overwritten every epoch -- by design, since it's meant
# to answer "where do I resume from," not "what did every epoch's loss look like"). Opened
# in append mode and written with a plain file write (not through PowerShell's
# -RedirectStandardOutput, which truncates train_stdout.log on every wrapper relaunch --
# confirmed losing every prior epoch's "epoch N mean loss" line across a resume) so the full
# per-epoch curve survives any number of interruptions/relaunches, independent of how the
# process is wrapped.
EPOCH_HISTORY_FILENAME = 'epoch_history.csv'
EPOCH_HISTORY_HEADER = 'epoch,mean_loss,mean_real_denoise,mean_synth_denoise,mean_demosaic\n'


def append_epoch_history(ckp_dir: Path, epoch: int, mean_loss: float,
                          mean_real_denoise: float, mean_synth_denoise: float, mean_demosaic: float):
    path = ckp_dir / EPOCH_HISTORY_FILENAME
    write_header = not path.exists()
    with path.open('a') as f:
        if write_header:
            f.write(EPOCH_HISTORY_HEADER)
        f.write(f'{epoch},{mean_loss},{mean_real_denoise},{mean_synth_denoise},{mean_demosaic}\n')


def limit_cpu_usage(num_threads: int, low_priority: bool):
    """Caps how much of this (GPU-less) machine's CPU the training run can take, so it
    can be left running for hours without blocking the user's own foreground work:
      - num_threads bounds PyTorch's intra-op thread pool (the actual conv/matmul compute,
        the dominant cost here) and OpenCV's thread pool (used by SyntheticMosaicImages'
        cv2.imread, which runs in the main process since DataLoader uses num_workers=0).
        NumPy/BLAS threading (used by DataAug's numpy-side steps) is controlled by
        OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS env vars instead, since those
        are read at import time -- set them in the shell before launching this script for
        full coverage; this function alone only covers the torch/cv2 side.
      - low_priority lowers the OS scheduling priority of this process itself (Windows
        only), so the OS scheduler automatically favors the user's foreground applications
        over this training run even within whatever thread budget num_threads allows --
        this is what actually keeps the machine feeling responsive, independent of the
        exact thread count chosen.
    """
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)  # this script is a single sequential pipeline; no benefit to more
    cv2.setNumThreads(num_threads)

    if low_priority and platform.system() == 'Windows':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # ctypes' default restype/argtypes are 32-bit c_int, which mis-marshals the 64-bit
        # HANDLE GetCurrentProcess()'s pseudo-handle (-1) needs on 64-bit Windows -- without
        # this, SetPriorityClass silently fails with ERROR_INVALID_HANDLE (confirmed via
        # GetLastError while debugging this). Must declare these explicitly as pointer-sized.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.GetCurrentProcess()
        ok = kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        if not ok:
            print(f'warning: SetPriorityClass failed (GetLastError={kernel32.GetLastError()}), '
                  f'continuing at normal priority')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-data-dir', type=Path, required=True,
                         help='CleanRawImages-format directory with an index.json, e.g. dataset/SID_Sony_raw')
    parser.add_argument('--synth-image-dir', type=Path, required=True,
                         help='directory (searched recursively) of ordinary sRGB images, e.g. a Kodak24/test folder')
    add_data_aug_args(parser)
    parser.add_argument('--init-ckp', type=Path, default=Path('models/torch_pretrained.ckp'),
                         help='denoise-only checkpoint to initialize the shared trunk from')
    parser.add_argument('--demosaic-weight', type=float, default=1.0,
                         help='weight of the demosaic L1 loss relative to the denoise L1 loss, on synthetic batches')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--ckp-dir', default=Path('models/checkpoints_demosaic'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-5, type=float,
                         help='fine-tuning LR -- small, since this adapts an already-converged '
                              'denoise checkpoint rather than training from scratch')
    parser.add_argument('--num-epoch', default=10, type=int)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--log-every', default=20, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                         help='defaults to cuda if available, else cpu')
    parser.add_argument('--cpu-threads', type=int, default=4,
                         help='caps PyTorch/OpenCV CPU thread usage so a long run on a GPU-less '
                              'machine does not fully occupy it -- see limit_cpu_usage(). Also set '
                              'OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS to the same '
                              'value in the launching shell for full coverage (numpy/BLAS side).')
    parser.add_argument('--low-priority', action=argparse.BooleanOptionalAction, default=True,
                         help='run this process at Windows BELOW_NORMAL priority (default on) so '
                              'the OS scheduler favors the user\'s foreground work; pass '
                              '--no-low-priority to disable')
    parser.add_argument('--auto-resume', action=argparse.BooleanOptionalAction, default=True,
                         help='if --ckp-dir already has a training_state.json from a previous '
                              '(interrupted) run, continue from there instead of --init-ckp; '
                              'pass --no-auto-resume to always start fresh from --init-ckp')
    args = parser.parse_args()

    limit_cpu_usage(args.cpu_threads, args.low_priority)

    args.ckp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    resumed = load_resume_state(args.ckp_dir) if args.auto_resume else None
    if resumed is not None:
        net, start_epoch, best_loss, best_epoch = resumed
        net = net.to(device)
    else:
        net = NetworkDemosaic.from_pretrained(args.init_ckp).to(device)
        start_epoch, best_loss, best_epoch = 0, float('inf'), None
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    aug_opts = build_data_aug_options(args)
    train_aug = DataAug(aug_opts, device=device)

    real_ds = CleanRawImages(data_dir=args.real_data_dir, opts=aug_opts)
    synth_ds = SyntheticMosaicImages(args.synth_image_dir, opts=aug_opts)
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    global_step = 0
    for epoch in range(start_epoch, args.num_epoch):
        net.train()
        # synth_loader (e.g. 42 Kodak24/McMaster images) is far smaller than real_loader
        # (e.g. 231 SID_Sony_raw images), so it determines how many steps make up one
        # epoch here; real_loader is cycled to keep supplying batches alongside it.
        real_iter = cycle(real_loader)
        epoch_losses = []
        epoch_real_denoise, epoch_synth_denoise, epoch_demosaic = [], [], []
        for synth_batch in tqdm(synth_loader, dynamic_ncols=True):
            # --- real raw: denoise loss only (no real demosaic ground truth yet) ---
            imgs, g_means = next(real_iter)
            noisy, gt_rggb, norm_k = train_aug.transform(imgs, g_means)
            optimizer.zero_grad()
            pred_rggb, _pred_rgb = net(noisy)
            loss_real = get_loss_l1(pred_rggb, gt_rggb, norm_k)
            loss_real.backward()
            optimizer.step()

            # --- synthetic sRGB: denoise loss + demosaic loss ---
            imgs, g_means, rgb01 = synth_batch
            noisy, gt_rggb, norm_k, gt_rgb = train_aug.transform(imgs, g_means, rgb01)
            optimizer.zero_grad()
            pred_rggb, pred_rgb = net(noisy)
            loss_denoise = get_loss_l1(pred_rggb, gt_rggb, norm_k)
            loss_demosaic = (pred_rgb - gt_rgb).abs().mean()
            loss_synth = loss_denoise + args.demosaic_weight * loss_demosaic
            loss_synth.backward()
            optimizer.step()

            epoch_losses.append(loss_real.item() + loss_synth.item())
            epoch_real_denoise.append(loss_real.item())
            epoch_synth_denoise.append(loss_denoise.item())
            epoch_demosaic.append(loss_demosaic.item())
            if global_step % args.log_every == 0:
                tqdm.write(
                    f"epoch {epoch} step {global_step}: real_denoise={loss_real.item():.4f} "
                    f"synth_denoise={loss_denoise.item():.4f} demosaic={loss_demosaic.item():.4f}"
                )
            global_step += 1

        # Per-step loss is noisy (same reasoning as train_qat_pytorch.py) -- the epoch
        # mean below is what should actually be watched and is what "best checkpoint"
        # is defined against, since there's no held-out validation split here either.
        epoch_loss = float(np.mean(epoch_losses))
        tqdm.write(f"epoch {epoch} mean loss: {epoch_loss:.4f}")
        append_epoch_history(
            args.ckp_dir, epoch, epoch_loss,
            float(np.mean(epoch_real_denoise)), float(np.mean(epoch_synth_denoise)), float(np.mean(epoch_demosaic)),
        )

        save_regular = (epoch % args.save_every == 0)
        is_new_best = epoch_loss < best_loss
        if save_regular or is_new_best:
            net.eval()
            state_dict = net.state_dict()

            if save_regular:
                torch.save(state_dict, args.ckp_dir / f"epoch_{epoch}.ckp")

            if is_new_best:
                if best_epoch is not None:
                    old_best_path = args.ckp_dir / f"epoch_best_{best_epoch}.ckp"
                    if old_best_path.exists():
                        old_best_path.unlink()
                torch.save(state_dict, args.ckp_dir / f"epoch_best_{epoch}.ckp")
                best_loss, best_epoch = epoch_loss, epoch
                tqdm.write(f"new best checkpoint: epoch {epoch}, mean loss {epoch_loss:.4f}")

            if save_regular:
                # only meaningful to resume from a regular epoch_{N}.ckp (always written when
                # save_regular), not from an epoch that only produced an epoch_best_*.ckp
                save_resume_state(args.ckp_dir, epoch, best_loss, best_epoch)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
