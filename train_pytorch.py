#!/usr/bin/env python3
"""PyTorch port of train_meg.py.

Same training flow (Adam + a piecewise/cyclic LR schedule + per-ISO-normalized L1 loss),
just swapping MegEngine's GradManager/optimizer/DataLoader for PyTorch's autograd/optimizer/
DataLoader. Uses models/net_torch.py::Network (the PyTorch port of models/net_mge.py::Network)
and dataset/training_pytorch.py (the PyTorch port of dataset/training.py) -- see that file's
docstring for the handful of bugs found and fixed while porting (missing channel permute,
missing cvt_k/cvt_b broadcast reshape, CleanRawImages.__init__ dropping its own `opts` arg,
the missing final inp_scale rescale in DataAug.transform).
The dead LR-schedule branch already removed from train_meg.py is not re-added here either.

DataAugOptions is built from individual CLI options (add_data_aug_args/build_data_aug_options
below) rather than a `--data-aug-config` JSON file -- defaults are the Reno 10x calibration,
matching run_benchmark_pytorch.py's KSigma(...) exactly, since that's this repo's only actual
target sensor. train_qat_pytorch.py reuses these same two functions.
"""
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.net_torch import Network
from dataset.training_pytorch import CleanRawImages, DataAug, DataAugOptions, NoiseProfile

# Reno 10x KSigma calibration, matching run_benchmark_pytorch.py's KSigma(...) construction
# exactly -- the default target sensor for every script in this repo.
RENO10X_K_COEFF = (0.0005995267, 0.00868861)
RENO10X_B_COEFF = (7.11772e-7, 6.514934e-4, 0.11492713)
RENO10X_ANCHOR_ISO = 1600.0
RENO10X_VALUE_SCALE = 959.0  # camera_value_scale / noise_profile.value_scale / KSigma.V
RENO10X_INP_SCALE = 256.0    # matches run_benchmark_pytorch.py's Denoiser/export_onnx.py default


def add_data_aug_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """CLI options for every dataset/training_pytorch.py::DataAugOptions field. Defaults are
    the Reno 10x calibration (matching run_benchmark_pytorch.py's KSigma) for --k-coeff/
    --b-coeff/--noise-value-scale/--camera-value-scale/--anchor-iso/--inp-scale -- these six
    are sensor-calibration constants, fixed for a given sensor, not training choices. The
    other three (--iso-range/--output-shape/--target-brightness-range) are pure training
    strategy knobs unrelated to sensor calibration -- --iso-range in particular is the cheap
    lever for widening high-ISO coverage discussed for QAT fine-tuning, so it stays a
    separate, easily-overridden option rather than being folded into the sensor constants.
    """
    group = parser.add_argument_group('data augmentation / KSigma calibration')
    group.add_argument('--k-coeff', nargs=2, type=float, default=list(RENO10X_K_COEFF),
                        help='noise model K(iso) polynomial coefficients')
    group.add_argument('--b-coeff', nargs=3, type=float, default=list(RENO10X_B_COEFF),
                        help='noise model B(iso)/Sigma(iso) polynomial coefficients')
    group.add_argument('--noise-value-scale', type=float, default=RENO10X_VALUE_SCALE,
                        help='the scale --k-coeff/--b-coeff were calibrated at')
    group.add_argument('--camera-value-scale', type=float, default=RENO10X_VALUE_SCALE,
                        help="scale applied to the clean [0,1] image before noise synthesis "
                             "-- same role as inference's KSigma.V")
    group.add_argument('--anchor-iso', type=float, default=RENO10X_ANCHOR_ISO,
                        help="ISO noise level everything is normalized to -- same as "
                             "inference's KSigma(anchor=...)")
    group.add_argument('--inp-scale', type=float, default=RENO10X_INP_SCALE,
                        help="final rescale before the network sees the data -- must match "
                             "the inference side's Denoiser/export_onnx.py inp_scale")
    
    group.add_argument('--iso-range', nargs=2, type=float, default=[800.0, 25600.0],
                        help='ISO range randomly sampled per training sample to synthesize '
                             'noise at')
    group.add_argument('--output-shape', nargs=2, type=int, default=[512, 512],
                        help='random-crop size in raw bayer-pixel space (network input ends '
                             'up half that per side, 4x channels)')
    group.add_argument('--target-brightness-range', nargs=2, type=float, default=[0.02, 0.5],
                        help='brightness-darkening augmentation target range (never brightens)')
    return group


def build_data_aug_options(args: argparse.Namespace) -> DataAugOptions:
    return DataAugOptions(
        noise_profile=NoiseProfile(
            K=tuple(args.k_coeff), B=tuple(args.b_coeff), value_scale=args.noise_value_scale,
        ),
        iso_range=tuple(args.iso_range),
        camera_value_scale=args.camera_value_scale,
        anchor_iso=args.anchor_iso,
        output_shape=tuple(args.output_shape),
        target_brighness_range=tuple(args.target_brightness_range),
        inp_scale=args.inp_scale,
    )


def get_loss_l1(pred: torch.Tensor, label: torch.Tensor, norm_k: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    L1 = (pred - label).abs().reshape(B, -1).mean(dim=1)
    L1 = L1 / norm_k.reshape(B)
    return L1.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    add_data_aug_args(parser)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ckp-dir', default=Path('./checkpoints'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=4000, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                         help='defaults to cuda if available, else cpu')
    args = parser.parse_args()

    args.ckp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Create model
    net = Network().to(device)
    # Create optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    aug_opts = build_data_aug_options(args)
    train_aug = DataAug(aug_opts, device=device)
    train_ds = CleanRawImages(data_dir=args.data_dir, opts=aug_opts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # learning rate scheduler
    def adjust_learning_rate(opt, epoch, step):
        M = len(train_ds) // args.batch_size
        T = M * 100
        Th = T // 2

        if epoch < 3000:
            f = 1 - step / (M*3000)
        elif epoch < 5000:
            f = 0.2
        else:
            f = 0.1

        t = step % T
        if t < Th:
            f2 = t / Th
        else:
            f2 = 2 - (t/Th)

        lr = f * f2 * args.lr

        for pgroup in opt.param_groups:
            pgroup["lr"] = lr

        return lr

    # train step
    def train_step(img, gt, norm_k):
        optimizer.zero_grad()
        pred = net(img)
        loss = get_loss_l1(pred, gt, norm_k)
        loss.backward()
        optimizer.step()
        return loss

    # train loop
    global_step = 0
    for epoch in range(args.num_epoch):
        for bidx, (imgs, g_means) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            imgs, gt, norm_k = train_aug.transform(imgs, g_means)
            lr = adjust_learning_rate(optimizer, epoch, global_step)
            loss = train_step(imgs, gt, norm_k)

            if global_step % 100 == 0:
                tqdm.write(f"clock: {epoch},{bidx}, loss: {loss.item()}, lr: {lr}")

            global_step += 1

        # save checkpoint
        if epoch % 100 == 0:
            torch.save(net.state_dict(), args.ckp_dir / f"epoch_{epoch}.ckp")


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
