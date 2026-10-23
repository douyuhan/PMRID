#!/usr/bin/env python3
"""Synthetic training data source for the demosaic head (see models/net_torch.py::
NetworkDemosaic): plain sRGB photos, approximately linearized and mosaicked, used
alongside dataset/training_pytorch.py::CleanRawImages' real sensor raw data.

The two sources are deliberately complementary, not redundant:
  - CleanRawImages: real sensor raw, so noise synthesis is physically correct, but
    there is no literal full-resolution RGB ground truth (a real CFA only ever
    samples one channel per pixel) -- used here for the denoise loss only.
  - SyntheticMosaicImages (this file): the "clean" image is an ordinary sRGB photo,
    so mosaicking it and treating the pre-mosaic image as ground truth gives an
    *exact* demosaic target with no teacher algorithm needed, at the cost of only
    approximating the real sensor's linear domain (inverse-gamma, CCM =~ identity).

Unlike CleanRawImages.random_flip_and_crop, no bayer-phase offset bookkeeping is
needed here: that trick exists because CleanRawImages crops/flips an array that has
*already* been mosaicked (so flipping shifts which physical site lands at an even/odd
position). Here the crop/flip happens on the full RGB image, before mosaicking, so
RawUtils.rgb_to_mosaic_rggb always samples at a fixed phase (top-left = R) and the
result is canonical RGGB regardless of which flip was applied.
"""
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils import RawUtils


class SyntheticMosaicImages(Dataset):

    IMAGE_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, image_dir, opts, gamma: float = 2.2):
        """
        Args:
            - image_dir: directory (searched recursively) of ordinary sRGB images
            - opts: dataset/training_pytorch.py::DataAugOptions (reuses .output_shape
              for the crop size, in the same "raw bayer-pixel space" units as
              CleanRawImages -- here that's just the RGB pixel grid size, since every
              pixel already has all 3 channels before mosaicking)
            - gamma: fixed gamma assumed to invert the sRGB images back to an
              approximately linear domain (matching this repo's other fixed-gamma=2.2
              conventions, e.g. RawUtils.bayer2rgb)
        """
        self.paths = sorted(
            p for p in Path(image_dir).rglob('*') if p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self.paths:
            raise ValueError(f'no images found under {image_dir}')
        self.opts = opts
        self.gamma = gamma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f'failed to read image: {path}')
        img01 = img_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0, 1]
        linear01 = img01 ** self.gamma  # approximate inverse sRGB gamma -> linear domain

        tH, tW = self.opts.output_shape
        H0, W0 = linear01.shape[:2]
        if H0 < tH or W0 < tW:
            raise ValueError(f'{path} ({H0}x{W0}) is smaller than the crop size {tH}x{tW}')

        # even crop origin keeps the 2x2 mosaic phase aligned after cropping
        y0 = np.random.randint(0, H0 - tH + 1) // 2 * 2
        x0 = np.random.randint(0, W0 - tW + 1) // 2 * 2
        crop = linear01[y0:y0 + tH, x0:x0 + tW]

        if np.random.rand() > 0.5:
            crop = np.flip(crop, axis=1)
        if np.random.rand() > 0.5:
            crop = np.flip(crop, axis=0)
        crop = np.ascontiguousarray(crop)

        rggb01 = RawUtils.rgb_to_mosaic_rggb(crop)
        rggb01_chw = np.ascontiguousarray(rggb01.transpose(2, 0, 1)).astype(np.float32)
        rgb01_chw = np.ascontiguousarray(crop.transpose(2, 0, 1)).astype(np.float32)
        g_mean_01 = float(crop[..., 1].mean())
        return rggb01_chw, g_mean_01, rgb01_chw

# vim: ts=4 sw=4 sts=4 expandtab
