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

Phase 3 Option 5 (doc_draft/PMRID_JDD_plan_phase3_demosaic_wb.md 方案5, opts.wb_gain_range
set -- see dataset/training_pytorch.py's module docstring): these source photos are
*already* "implicitly WB'd" (ordinary processed sRGB images, channels roughly balanced --
see that doc's root-cause section), unlike real sensor raw, which is genuinely unWB'd.
To retrain the network on a unified "native, unWB'd raw" input convention, this class
mosaics a deliberately un-white-balanced version of each crop (dividing R/B by a randomly
sampled per-sample gain before mosaicking -- G is untouched, gain=1), simulating what this
same scene would have looked like straight off a real sensor before WB. The *un-divided*
crop is still returned as-is for the demosaic RGB ground truth: for the clean signal this
round trip is an exact no-op ((L/gain)*gain == L), so the GT doesn't need touching -- it's
only the *noisy* path (noise added downstream, in DataAug.transform, between this inv-WB
and the matching WB-back) that actually differs from a plain mosaic, because noise
synthesized at the smaller per-channel native magnitude ends up proportionally larger once
WB'd back, reproducing a real sensor's "channels needing more WB gain end up relatively
noisier" behavior. The sampled gain is returned alongside the mosaic so
DataAug.transform can apply the *same* value for its WB-back step (it can't resample
independently -- see that function's batch_wb_gain parameter).
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

        # Phase 3 Option 5 (see module docstring): mosaic a deliberately un-white-balanced
        # version of crop. wb_gain is always returned (identity [1, 1] when
        # opts.wb_gain_range is None), so DataAug.transform's batch_wb_gain parameter
        # always receives a real value and the whole round trip collapses to a no-op
        # exactly reproducing pre-Option-5 behavior when the option is disabled.
        if self.opts.wb_gain_range is not None:
            (r_lo, r_hi), (b_lo, b_hi) = self.opts.wb_gain_range
            r_gain = float(np.random.uniform(r_lo, r_hi))
            b_gain = float(np.random.uniform(b_lo, b_hi))
        else:
            r_gain, b_gain = 1.0, 1.0
        wb_gain = np.array([r_gain, b_gain], dtype=np.float32)

        crop_native = crop.copy()
        crop_native[..., 0] /= r_gain
        crop_native[..., 2] /= b_gain

        rggb01 = RawUtils.rgb_to_mosaic_rggb(crop_native)
        rggb01_chw = np.ascontiguousarray(rggb01.transpose(2, 0, 1)).astype(np.float32)
        # the demosaic RGB ground truth is the ORIGINAL (not inv-WB'd) crop -- see
        # module docstring for why this round trip must not touch it
        rgb01_chw = np.ascontiguousarray(crop.transpose(2, 0, 1)).astype(np.float32)
        g_mean_01 = float(crop[..., 1].mean())
        return rggb01_chw, g_mean_01, rgb01_chw, wb_gain

# vim: ts=4 sw=4 sts=4 expandtab
