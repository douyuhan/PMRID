#!/usr/bin/env python3
"""PyTorch port of train_meg.py.

Same training flow (Adam + a piecewise/cyclic LR schedule + per-ISO-normalized L1 loss),
just swapping MegEngine's GradManager/optimizer/DataLoader for PyTorch's autograd/optimizer/
DataLoader. Uses models/net_torch.py::Network (the PyTorch port of models/net_mge.py::Network)
and dataset/training_pytorch.py (the PyTorch port of dataset/training.py) -- see that file's
docstring for the handful of bugs found and fixed while porting (missing channel permute,
missing cvt_k/cvt_b broadcast reshape, CleanRawImages.__init__ dropping its own `opts` arg).
The dead LR-schedule branch already removed from train_meg.py is not re-added here either.
"""
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.net_torch import Network
from dataset.training_pytorch import CleanRawImages, DataAug, DataAugOptions


def get_loss_l1(pred: torch.Tensor, label: torch.Tensor, norm_k: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    L1 = (pred - label).abs().reshape(B, -1).mean(dim=1)
    L1 = L1 / norm_k.reshape(B)
    return L1.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-aug-config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
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

    aug_opts = DataAugOptions.parse_file(args.data_aug_config)
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
