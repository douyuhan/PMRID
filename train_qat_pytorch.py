#!/usr/bin/env python3
"""QAT (quantization-aware training) fine-tuning for models/net_torch.py::Network.

Starts from the pretrained fp32 checkpoint and fine-tunes with fake-quantization
inserted via `prepare_qat_fx` (per-channel symmetric int8 weight fake-quant, matching
quantize_onnx.py's own `--per-channel` default), so the weights this saves are already
shaped to tolerate that quantization scheme -- narrowing the calibration-coverage gap
quantize_onnx.py alone can't close, by adapting the weight distributions themselves
instead of only the calibrated scale/zero-point.

`convert_fx()` is deliberately never called here. PyTorch's own quantization backend
refuses per-channel quantization for ConvTranspose2d ("Per Channel Quantization is
currently disabled for transposed conv") -- and every DecoderStage in this network uses
one for its 2x upsample -- so producing a deployable int8 model straight from this
script isn't an option (confirmed empirically; see CLAUDE.md's int8 section). Instead,
after fake-quant fine-tuning, the underlying fp32 weights are extracted (fake-quant
only affects the forward computation, never the stored parameter dtype) and saved as a
plain Network()-compatible state_dict -- loadable exactly like models/torch_pretrained.ckp,
and meant to be fed through the existing export_onnx.py (fp32 export) ->
quantize_onnx.py (onnxruntime PTQ, which has no ConvTranspose2d per-channel restriction)
pipeline, same as the vanilla checkpoint, just calibrated from QAT-adapted weights.

Extraction relies on this architecture having no BatchNorm to fuse: prepare_qat_fx swaps
each Conv2d/ConvTranspose2d for its nn.qat.* counterpart in place (same qualified name,
same real fp32 .weight/.bias parameters, just with extra weight_fake_quant/
activation_post_process buffers alongside) rather than restructuring the module graph --
so a plain Network()'s state_dict keys are a strict subset of the QAT-prepared model's,
and can be pulled out by name.

Data augmentation / KSigma calibration options (--k-coeff/--b-coeff/--iso-range/...) are
shared with train_pytorch.py via add_data_aug_args/build_data_aug_options -- defaults are
the Reno 10x calibration, matching this repo's only actual target sensor.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.training_pytorch import CleanRawImages, DataAug
from models.net_torch import Network
from train_pytorch import add_data_aug_args, build_data_aug_options, get_loss_l1


def extract_float_state_dict(prepared: torch.nn.Module) -> dict:
    plain_keys = set(Network().state_dict().keys())
    prepared_sd = prepared.state_dict()
    missing = plain_keys - prepared_sd.keys()
    if missing:
        raise RuntimeError(
            f"QAT-prepared model is missing {len(missing)} key(s) a plain Network() "
            f"expects (e.g. {sorted(missing)[:5]}) -- prepare_qat_fx must have fused or "
            f"renamed a submodule, so this extraction's no-BatchNorm assumption no "
            f"longer holds and needs revisiting."
        )
    return {k: prepared_sd[k] for k in plain_keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    add_data_aug_args(parser)
    parser.add_argument('--init-ckp', type=Path, default=Path('models/torch_pretrained.ckp'),
                         help='fp32 checkpoint to start QAT fine-tuning from')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--ckp-dir', default=Path('models/checkpoints_qat'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-5, type=float,
                         help='QAT fine-tuning LR -- much smaller than train_pytorch.py\'s '
                              'from-scratch default, since this only lightly adapts '
                              'already-converged weights to fake-quant noise')
    parser.add_argument('--num-epoch', default=10, type=int,
                         help='QAT fine-tuning is meant to be brief -- a handful of epochs '
                              'of adaptation, not full retraining')
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--backend', default='onednn',
                         help='quantization backend whose default QAT qconfig to fake-quantize '
                              'with -- onednn gives per-channel symmetric int8 weights, '
                              'matching quantize_onnx.py\'s own --per-channel default')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                         help='defaults to cuda if available, else cpu')
    args = parser.parse_args()

    args.ckp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    net = Network()
    net.load_state_dict(torch.load(args.init_ckp, map_location='cpu'))
    net.to(device)

    qconfig = get_default_qat_qconfig(args.backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_inputs = (torch.randn(1, 4, 64, 64, device=device),)
    prepared = prepare_qat_fx(net, qconfig_mapping, example_inputs)
    prepared.to(device)

    optimizer = optim.Adam(prepared.parameters(), lr=args.lr)

    aug_opts = build_data_aug_options(args)
    train_aug = DataAug(aug_opts, device=device)
    train_ds = CleanRawImages(data_dir=args.data_dir, opts=aug_opts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    global_step = 0
    best_loss = float('inf')
    best_epoch = None
    for epoch in range(args.num_epoch):
        prepared.train()
        epoch_losses = []
        for imgs, g_means in tqdm(train_loader, dynamic_ncols=True):
            imgs, gt, norm_k = train_aug.transform(imgs, g_means)

            optimizer.zero_grad()
            pred = prepared(imgs)
            loss = get_loss_l1(pred, gt, norm_k)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if global_step % 100 == 0:
                tqdm.write(f"clock: {epoch}, loss: {loss.item()}")
            global_step += 1

        # Per-step loss (just printed above) is too noisy to judge convergence from --
        # even at batch_size=4 the per-step coefficient of variation is still ~0.57
        # (checked empirically against this dataset/config), so a single value can be
        # several times the true average. The epoch mean below is what should actually
        # be watched, and is also what "the checkpoint with minimum loss" is defined
        # against, since there's no held-out validation split in this pipeline.
        epoch_loss = float(np.mean(epoch_losses))
        tqdm.write(f"epoch {epoch} mean loss: {epoch_loss:.4f}")

        save_regular = (epoch % args.save_every == 0)
        is_new_best = epoch_loss < best_loss
        if save_regular or is_new_best:
            prepared.eval()
            state_dict = extract_float_state_dict(prepared)

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


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
