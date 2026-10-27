#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
import skimage.metrics
from tqdm import tqdm

from models.net_torch import Network, NetworkDemosaic
from utils import RawUtils, round_half_up
from dataset.benchmark import BenchmarkLoader, RawMeta


class KSigma:

    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


class Denoiser:

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0, device='cpu'):
        net = Network()
        state_dict = torch.load(str(model_path), map_location=device)
        net.load_state_dict(state_dict)
        net.eval()

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray):
        H, W = bayer_01.shape
        ph, pw = (32-((H//2) % 32))//2, (32-((W//2) % 32))//2

        # reflect-pad the RAW bayer mosaic (before packing into RGGB planes), not the
        # planes themselves: zero-padding creates an artificial black edge the network
        # never saw in training, which visibly hurts denoising quality right at the
        # image border. Padding by 2*ph/2*pw raw pixels (always even, since ph/pw are
        # multiplied by 2) with numpy's default 'reflect' (no edge-pixel duplication)
        # keeps the bayer color at each mosaic position correct across the seam --
        # e.g. row "R0 G0 R1 G1" padded by 2 columns each side becomes
        # "R1 G0 | R0 G0 R1 G1 | R1 G0", not "G0 R0 | ... | G1 R1" (what an
        # edge-duplicating reflect, e.g. cv2.BORDER_REFLECT/mode='symmetric', would
        # give -- it flips the R/G parity right at the seam).
        bayer_01 = np.pad(bayer_01, [(2*ph, 2*ph), (2*pw, 2*pw)], mode='reflect')

        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale

        inp = np.ascontiguousarray(inp_rggb, dtype=np.float32)
        with torch.no_grad():
            inp_t = torch.from_numpy(inp).to(self.device)
            pred_t = self.net(inp_t)[0] / self.inp_scale
            pred = pred_t.cpu().numpy().transpose(1, 2, 0)
        pred = self.ksigma(pred, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred = pred[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred)


def demosaic_rgb_to_display(rgb_01: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Converts NetworkDemosaic's direct full-resolution rgb output to a displayable
    image. Unlike RawUtils.bayer2rgb's real-sensor pipeline, this does NOT apply white
    balance or a CCM: the demosaic head (see models/net_torch.py::NetworkDemosaic,
    train_JDD_pytorch.py) was trained only on Phase-1 synthetic data -- ordinary public
    sRGB photos, approximately inverse-gamma'd into "linear" domain (see
    dataset/synthetic_mosaic.py) -- which are already "a normal photo's colors," not raw
    sensor response needing WB/CCM correction. Applying bayer2rgb's WB/CCM step to this
    output would be wrong, not just redundant. Feeding this to genuinely real sensor raw
    will visibly mis-color the result -- a known, documented Phase 1 limitation (the
    demosaic head has never seen real sensor raw), not a bug in this function."""
    return np.clip(rgb_01, 0, 1) ** (1 / gamma)


class DemosaicDenoiser:
    """Like Denoiser, but wraps NetworkDemosaic and also returns its direct
    full-resolution rgb demosaic output (KSigma-denormalized, [0,1]-ish linear domain,
    not yet gamma'd for display -- see demosaic_rgb_to_display)."""

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0, device='cpu'):
        net = NetworkDemosaic()
        state_dict = torch.load(str(model_path), map_location=device)
        net.load_state_dict(state_dict)  # strict=True: a fully-trained JDD checkpoint
        # already has every key (unlike NetworkDemosaic.from_pretrained, which is for
        # loading a plain denoise-only checkpoint into a freshly-initialized demosaic head)
        net.eval()

        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray):
        # identical to Denoiser.pre_process -- see that method's comments for why
        # reflect-padding (not zero-padding) by an even raw-pixel amount is needed
        H, W = bayer_01.shape
        ph, pw = (32-((H//2) % 32))//2, (32-((W//2) % 32))//2
        bayer_01 = np.pad(bayer_01, [(2*ph, 2*ph), (2*pw, 2*pw)], mode='reflect')

        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale

        inp = np.ascontiguousarray(inp_rggb, dtype=np.float32)
        with torch.no_grad():
            inp_t = torch.from_numpy(inp).to(self.device)
            pred_rggb_t, pred_rgb_t = self.net(inp_t)
            pred_rggb = (pred_rggb_t[0] / self.inp_scale).cpu().numpy().transpose(1, 2, 0)
            pred_rgb_full = (pred_rgb_t[0] / self.inp_scale).cpu().numpy().transpose(1, 2, 0)
        # same per-frame KSigma affine denormalizes both outputs -- DataAug.transform
        # applied the identical forward affine to both the RGGB and RGB ground truth
        # during training, so the inverse here must match for both too
        pred_rggb = self.ksigma(pred_rggb, iso, inverse=True)
        pred_rgb_full = self.ksigma(pred_rgb_full, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred_rggb = pred_rggb[ph:-ph, pw:-pw]
        # rgb is at 2x the RGGB-plane resolution, i.e. exactly the padded raw-pixel
        # resolution -- so it needs the same 2*ph/2*pw crop pre_process padded by,
        # not the ph/pw crop the (half-resolution) RGGB output uses
        pred_rgb_full = pred_rgb_full[2*ph:-2*ph, 2*pw:-2*pw]
        return RawUtils.rggb2bayer(pred_rggb), pred_rgb_full


def save_rgb_png(path: Path, rgb_01: np.ndarray):
    bgr_uint8 = round_half_up(rgb_01.clip(0, 1)[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(path), bgr_uint8)


def save_bayer_raw(path: Path, bayer_01: np.ndarray, bitWidth: int):
    bayer_uint16 = round_half_up(bayer_01.clip(0, 1) * (2 ** bitWidth))
    max_val = 2 ** bitWidth - 1
    bayer_uint16 = bayer_uint16.clip(0, max_val).astype(np.uint16)
    bayer_uint16.tofile(str(path))


def run_benchmark(model_path, bm_loader: BenchmarkLoader, save_dir: Path = None, jdd: bool = False):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = DemosaicDenoiser(model_path, ksigma) if jdd else Denoiser(model_path, ksigma)
    if jdd:
        print(
            "--jdd: additionally saving {name}_pred_demosaic.png (the network's direct "
            "demosaic output). No PSNR/SSIM is computed for it -- it isn't in the same "
            "color space as this benchmark's WB+CCM'd GT (see demosaic_rgb_to_display's "
            "docstring), and on real sensor raw it's expected to look visibly off-color "
            "(known Phase 1 limitation, not a bug). The PSNR/SSIM below is still the "
            "denoise branch only, computed exactly as without --jdd."
        )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    PSNRs, SSIMs = [], []

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, meta in bar:
        bar.set_description(meta.name)
        pattern = meta.bayer_pattern
        # convert whatever bayer_pattern this sample uses to the canonical
        # RGGB layout that bayer2rggb/rggb2bayer/bayer2rgb assume; this is
        # an involution, so applying it again with the same pattern later
        # converts back (canonical -> original)
        input_bayer, gt_bayer = RawUtils.to_canonical_rggb(input_bayer, gt_bayer, pattern=pattern)

        if jdd:
            pred_bayer, pred_rgb_demosaic = denoiser.run(input_bayer, iso=meta.ISO)
        else:
            pred_bayer = denoiser.run(input_bayer, iso=meta.ISO)

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
            input_bayer, pred_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )
        if jdd:
            inp_rgb, pred_rgb, gt_rgb, pred_rgb_demosaic = RawUtils.to_canonical_rggb(
                inp_rgb, pred_rgb, gt_rgb, pred_rgb_demosaic, pattern=pattern,
            )
        else:
            inp_rgb, pred_rgb, gt_rgb = RawUtils.to_canonical_rggb(inp_rgb, pred_rgb, gt_rgb, pattern=pattern)
        bar.set_description(meta.name+' ✓')

        if save_dir is not None:
            # full-image RGB visualizations (not cropped to ROIs)
            save_rgb_png(save_dir / f'{meta.name}_input.png', inp_rgb)
            save_rgb_png(save_dir / f'{meta.name}_pred.png', pred_rgb)
            save_rgb_png(save_dir / f'{meta.name}_gt.png', gt_rgb)
            if jdd:
                save_rgb_png(save_dir / f'{meta.name}_pred_demosaic.png',
                             demosaic_rgb_to_display(pred_rgb_demosaic))

            # denoised bayer raw, flipped back to the sample's original bayer_pattern
            # and rescaled to the same raw_bitWidth range as the input/gt files
            pred_bayer_orig_pattern = RawUtils.to_canonical_rggb(pred_bayer, pattern=pattern)
            save_bayer_raw(save_dir / f'{meta.name}_pred.raw', pred_bayer_orig_pattern, meta.raw_bitWidth)

        psnrs = []
        ssims = []

        for x0, y0, x1, y1 in meta.ROIs:
            pred_patch = pred_rgb[y0:y1, x0:x1]
            gt_patch = gt_rgb[y0:y1, x0:x1]

            psnr = skimage.metrics.peak_signal_noise_ratio(gt_patch, pred_patch, data_range=1.0)
            ssim = skimage.metrics.structural_similarity(gt_patch, pred_patch, channel_axis=-1, data_range=1.0)
            psnrs.append(float(psnr))
            ssims.append(float(ssim))

        bar.set_description(meta.name+' ✓✓')

        PSNRs = PSNRs + psnrs   # list append
        SSIMs = SSIMs + ssims

    mean_psnr = np.mean(PSNRs)
    mean_ssim = np.mean(SSIMs)
    print("mean PSNR:", mean_psnr)
    print("mean SSIM:", mean_ssim)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument(
        '--save-dir', type=Path, default=None,
        help='directory to save input/pred/gt RGB visualizations (png, full image) '
             'and the denoised bayer output (.raw, restored to the original bayer_pattern)',
    )
    parser.add_argument(
        '--jdd', action='store_true',
        help="model is a models/net_torch.py::NetworkDemosaic checkpoint (e.g. from "
             "train_JDD_pytorch.py), not a plain denoise-only Network. Denoise PSNR/SSIM "
             "is computed exactly as without this flag; additionally saves "
             "{name}_pred_demosaic.png from the network's own demosaic output (see "
             "demosaic_rgb_to_display for why this has no PSNR/SSIM and no WB/CCM).",
    )

    args = parser.parse_args()

    bm_loader = BenchmarkLoader(args.benchmark.resolve())
    run_benchmark(args.model, bm_loader, save_dir=args.save_dir, jdd=args.jdd)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
