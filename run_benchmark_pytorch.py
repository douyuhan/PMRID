#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
import skimage.metrics
from tqdm import tqdm

from models.net_torch import Network
from utils import RawUtils
from benchmark import BenchmarkLoader, RawMeta


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


def save_rgb_png(path: Path, rgb_01: np.ndarray):
    bgr_uint8 = np.round(rgb_01.clip(0, 1)[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(path), bgr_uint8)


def save_bayer_raw(path: Path, bayer_01: np.ndarray, max_val: int):
    bayer_uint16 = np.round(bayer_01.clip(0, 1) * max_val).astype(np.uint16)
    bayer_uint16.tofile(str(path))


def run_benchmark(model_path, bm_loader: BenchmarkLoader, save_dir: Path = None):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(model_path, ksigma)

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

        pred_bayer = denoiser.run(input_bayer, iso=meta.ISO)

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
            input_bayer, pred_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )
        inp_rgb, pred_rgb, gt_rgb = RawUtils.to_canonical_rggb(inp_rgb, pred_rgb, gt_rgb, pattern=pattern)
        bar.set_description(meta.name+' ✓')

        if save_dir is not None:
            # full-image RGB visualizations (not cropped to ROIs)
            save_rgb_png(save_dir / f'{meta.name}_input.png', inp_rgb)
            save_rgb_png(save_dir / f'{meta.name}_pred.png', pred_rgb)
            save_rgb_png(save_dir / f'{meta.name}_gt.png', gt_rgb)

            # denoised bayer raw, flipped back to the sample's original bayer_pattern
            # and rescaled to the same raw_bitWidth range as the input/gt files
            pred_bayer_orig_pattern = RawUtils.to_canonical_rggb(pred_bayer, pattern=pattern)
            max_val = 2 ** meta.raw_bitWidth - 1
            save_bayer_raw(save_dir / f'{meta.name}_pred.raw', pred_bayer_orig_pattern, max_val)

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

    args = parser.parse_args()

    bm_loader = BenchmarkLoader(args.benchmark.resolve())
    run_benchmark(args.model, bm_loader, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
