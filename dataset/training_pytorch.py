#!/usr/bin/env python3
"""PyTorch port of dataset/training.py (the MegEngine training data pipeline).

Ports the framework-specific pieces (megengine.data.Dataset -> torch.utils.data.Dataset,
megengine random/round ops -> torch equivalents) and drops two dependencies not installed
in this repo's AIISP_init env (pydantic, megfile) in favor of dataclasses + plain
pathlib/open, consistent with how the rest of this repo (e.g. dataset/benchmark.py's
RawMeta) already does config/metadata parsing.

While porting, a few more bugs turned up in the MegEngine reference beyond the ones found
first (BayerPattern(Enum, str) order, add_noise's missing return value, the dead
LR-schedule branch in train_meg.py) -- these later ones have since been fixed in both
dataset/training.py and this file, so the two stay in sync:
  - CleanRawImages.__init__ did `self.opts = DataAugOptions` (the class, not the `opts`
    argument actually passed in) -- fixed to `self.opts = opts`.
  - bayer2rggb packing there produces channel-LAST (H, W, 4); nothing in that pipeline
    ever permuted to channel-first before calling the network, which expects (N, 4, H, W)
    (matching models/net_torch.py::Network / models/net_mge.py::Network's first conv).
    Fixed by permuting to (4, H, W) right in __getitem__, so the default DataLoader
    collate stacks batches straight into (N, 4, H, W).
  - DataAug.k_sigma's cvt_k/cvt_b are per-sample values (shape (N,)) but were applied to
    the (N, H, W, 4)-shaped batch with a bare `* cvt_k` -- with no reshape, that only
    broadcasts correctly by accident (would require N == 4). Fixed by reshaping
    cvt_k/cvt_b to (-1, 1, 1, 1), the same way add_noise already reshapes k/b.
  - DataAug.transform ended right after `* cvt_k + cvt_b`, leaving the network input in
    raw `camera_value_scale`-scaled units. Inference (run_benchmark_pytorch.py) instead
    ends KSigma normalization with `/ V` (KSigma.__call__) followed by a separate
    `* inp_scale` (Denoiser.run) before the network ever sees the data -- a step this
    training pipeline had no equivalent of at all. Confirmed numerically (fixed camera
    parameters, varying iso) that this made every training-time network input exactly
    `camera_value_scale / inp_scale` larger in magnitude than what the same model sees at
    inference (~3.75x for the Reno 10x defaults, 959/256, constant across iso). Fixed by
    adding an `inp_scale` field to DataAugOptions (default 256.0, matching
    export_onnx.py/run_benchmark_pytorch.py's own default) and applying the matching
    `/ camera_value_scale * inp_scale` step at the end of DataAug.transform.

Found later, and (per the sync-policy change noted above) fixed in this file only, not
dataset/training.py: DataAug.add_noise's `shot_noisy = torch.poisson((img / k_t).clamp(0,
1)) * k_t` clamped the Poisson rate's *upper* bound to 1. img lives in
[0, camera_value_scale] (~0-959 for the Reno 10x defaults -- confirmed by both add_noise's
own docstring and by k_sigma() operating on the exact same camera_value_scale-scaled
tensors), while k_t is only ~0.5-15 over the calibrated ISO range, so img/k_t commonly
reaches into the hundreds for any non-trivial brightness -- the upper clamp was crushing
the Poisson rate to ~1 for virtually every non-near-black pixel, making shot_noisy
~= Poisson(1)*k_t, i.e. nearly independent of img's actual value rather than noise added
on top of it. Caught by directly visualizing a synthesized noisy mosaic next to its clean
source (something no prior validation here had done -- only aggregate loss curves/shapes
had been checked): the "noisy" image was visually indistinguishable from pure static.
Ruled out a camera_value_scale/[0,1] domain mismatch in transform()'s call site first
(the read-noise term sqrt(b) -- unaffected by this clamp -- would be 100%-2200% of full
scale if img were [0,1], vs. a believable 0.1%-2.3% of camera_value_scale; img_add_noise
and k_sigma already use the identical camera_value_scale-scaled tensors). Fixed by keeping
only the lower bound (`.clamp(min=0)`, still needed to keep the Poisson rate
non-negative): confirmed the same mosaic's structure becomes visible again, with mean abs
diff vs. the clean mosaic dropping ~13x (1.47e-1 -> 1.14e-2 on a [0,1] scale, ISO 3200).
This clamp is present identically in dataset/training.py (`.clip(0, 1)`, same line) --
confirmed to predate this port, not introduced by it.
"""
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BayerPattern(str, Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


@dataclass
class RawImageItem:
    path: str
    width: int
    height: int
    black_level: int
    bayer_pattern: BayerPattern
    g_mean_01: float
    white_level: int = 65535


@dataclass
class NoiseProfile:
    K: Tuple[float, float]
    B: Tuple[float, float, float]
    value_scale: float = 959.0


@dataclass
class DataAugOptions:
    noise_profile: NoiseProfile
    iso_range: Tuple[float, float]
    camera_value_scale: float = 959.0
    anchor_iso: float = 1600.0
    output_shape: Tuple[int, int] = (512, 512)   # 512x512x4
    target_brighness_range: Tuple[float, float] = (0.02, 0.5)
    inp_scale: float = 256.0

    @staticmethod
    def parse_file(path) -> "DataAugOptions":
        data = dict(json.loads(Path(path).read_text()))
        data['noise_profile'] = NoiseProfile(**data['noise_profile'])
        data['iso_range'] = tuple(data['iso_range'])
        if 'output_shape' in data:
            data['output_shape'] = tuple(data['output_shape'])
        if 'target_brighness_range' in data:
            data['target_brighness_range'] = tuple(data['target_brighness_range'])
        return DataAugOptions(**data)


class CleanRawImages(Dataset):

    def __init__(self, *, index_file: Optional[str] = None, data_dir: Optional[Path] = None, opts: DataAugOptions):
        """
        Args:
            - data_dir: a directory that contains "index.json" and raw images
            - index_file: the absolute path to the index file
        """
        super().__init__()

        assert not (index_file is None and data_dir is None)

        if data_dir is None:
            index_file = Path(index_file)
        else:
            assert index_file is None
            data_dir = Path(data_dir)
            index_file = data_dir / "index.json"

        self.opts = opts
        self.filelist: List[RawImageItem] = []
        with index_file.open() as f:
            raw_items = json.load(f)
        for raw in raw_items:
            raw = dict(raw)
            raw['bayer_pattern'] = BayerPattern(raw['bayer_pattern'])
            item = RawImageItem(**raw)
            if data_dir is not None:
                item.path = str(data_dir / item.path)
            self.filelist.append(item)

    def __len__(self):
        return len(self.filelist)

    def random_flip_and_crop(self, img: np.ndarray, src_bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Random flip and crop a bayer-patterned image, and normalize the bayer pattern to RGGB.
        """

        flip_ud = np.random.rand() > 0.5
        flip_lr = np.random.rand() > 0.5

        if src_bayer_pattern == BayerPattern.RGGB:
            crop_x_offset, crop_y_offset = 0, 0
        elif src_bayer_pattern == BayerPattern.GBRG:
            crop_x_offset, crop_y_offset = 0, 1
        elif src_bayer_pattern == BayerPattern.GRBG:
            crop_x_offset, crop_y_offset = 1, 0
        elif src_bayer_pattern == BayerPattern.BGGR:
            crop_x_offset, crop_y_offset = 1, 1

        if flip_lr:
            crop_x_offset = (crop_x_offset + 1) % 2
        if flip_ud:
            crop_y_offset = (crop_y_offset + 1) % 2

        H0, W0 = img.shape
        tH, tW = self.opts.output_shape

        x0, y0 = np.random.randint(0, W0 - tW), np.random.randint(0, H0 - tH)
        x0, y0 = x0 // 2 * 2 + crop_x_offset, y0 // 2 * 2 + crop_y_offset

        img_crop = img[y0:y0+tH, x0:x0+tW]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)

        return img_crop

    def __getitem__(self, index: int):
        item = self.filelist[index]
        rawimg = np.fromfile(item.path, dtype=np.uint16).reshape((item.height, item.width))
        # random crop to output size
        rawimg = self.random_flip_and_crop(rawimg, item.bayer_pattern).astype(np.float32)

        raw01 = (rawimg - item.black_level) / (item.white_level - item.black_level)
        H, W = raw01.shape
        # pixel shuffle to RGGB image, then channel-first (4, H, W) for the network's NCHW
        # convention -- the MegEngine reference never does this permute (see module docstring)
        rggb01 = raw01.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
        rggb01 = np.ascontiguousarray(rggb01.transpose(2, 0, 1))
        return rggb01, np.float32(item.g_mean_01)


class NoiseProfileFunc:

    def __init__(self, noise_profile: NoiseProfile):
        self.polyK = np.poly1d(noise_profile.K)
        self.polyB = np.poly1d(noise_profile.B)
        self.value_scale = noise_profile.value_scale

    def __call__(self, iso, value_scale=959.0):
        r = value_scale / self.value_scale
        k = self.polyK(iso) * r
        b = self.polyB(iso) * r * r

        return k, b


class DataAug:

    def __init__(self, opts: DataAugOptions, device='cpu'):
        self.opts = opts
        self.noise_func = NoiseProfileFunc(opts.noise_profile)
        self.device = torch.device(device)

    def transform(self, batch_img01: torch.Tensor, batch_g_mean, batch_rgb01: torch.Tensor = None):
        """
        Args:
            - batch_img01: (N, 4, H, W) float32 tensor, clean RGGB in [0, 1] (as returned
              by CleanRawImages, batched by the DataLoader's default collate)
            - batch_g_mean: (N,) per-sample mean green-channel brightness in [0, 1]
            - batch_rgb01: optional (N, 3, 2H, 2W) float32 tensor, an exact full-resolution
              RGB ground truth for the same scene as batch_img01 (as returned by
              dataset/synthetic_mosaic.py::SyntheticMosaicImages, whose RGGB is a direct
              mosaic of this RGB -- unlike CleanRawImages, which has no RGB counterpart at
              all). When given, it is carried through the *same* brightness/KSigma/
              inp_scale affine chain as batch_gt (so a demosaic loss against it is
              consistent with the denoise loss's domain) but never has noise added, since
              it's the demosaic target, not a noisy input.

        Returns:
            - noisy_img (network input)
            - gt (network target)
            - norm_k (per-sample loss normalization factor, for get_loss_l1)
            - gt_rgb (only when batch_rgb01 is given)
        """
        batch_imgs = batch_img01.to(self.device, dtype=torch.float32) * self.opts.camera_value_scale
        batch_gt, brightness_scale = self.brightness_aug(batch_imgs, batch_g_mean)
        batch_imgs, batch_iso = self.add_noise(batch_gt)
        cvt_k, cvt_b = self.k_sigma(batch_iso)

        batch_imgs = batch_imgs * cvt_k + cvt_b
        batch_gt = batch_gt * cvt_k + cvt_b

        # Match inference's final normalization exactly (run_benchmark_pytorch.py's
        # KSigma.__call__ ends with `/ V`, then Denoiser.run separately multiplies by
        # `inp_scale` before feeding the network) -- without this, the network here was
        # trained on inputs a fixed `camera_value_scale / inp_scale` (~3.75x for the
        # Reno 10x defaults) larger in magnitude than what it actually sees at inference.
        batch_imgs = batch_imgs / self.opts.camera_value_scale * self.opts.inp_scale
        batch_gt = batch_gt / self.opts.camera_value_scale * self.opts.inp_scale

        if batch_rgb01 is None:
            return batch_imgs, batch_gt, cvt_k

        batch_rgb = batch_rgb01.to(self.device, dtype=torch.float32) * self.opts.camera_value_scale
        batch_rgb = batch_rgb * brightness_scale
        batch_rgb = batch_rgb * cvt_k + cvt_b
        batch_rgb = batch_rgb / self.opts.camera_value_scale * self.opts.inp_scale
        return batch_imgs, batch_gt, cvt_k, batch_rgb

    def k_sigma(self, iso: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        k, sigma = self.noise_func(iso, value_scale=self.opts.camera_value_scale)
        k_a, sigma_a = self.noise_func(self.opts.anchor_iso, value_scale=self.opts.camera_value_scale)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        cvt_k = torch.as_tensor(cvt_k, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        cvt_b = torch.as_tensor(cvt_b, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        return cvt_k, cvt_b

    def brightness_aug(self, img_batch: torch.Tensor, orig_gmean) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (darkened_batch, scale) -- the scale is exposed so transform() can
        apply the identical per-sample darkening to a parallel RGB ground truth tensor
        without redrawing a different random scale for it."""
        low, high = self.opts.target_brighness_range
        orig_gmean = np.asarray(orig_gmean)
        N = len(orig_gmean)
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N,)))
        s = np.clip(btarget / orig_gmean, 0.01, 1.0)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        return img_batch * s, s

    def add_noise(self, img: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Args:
            - img: [-black, camera_value_scale]

        Returns:
            - noisy_img
            - iso
        """

        N = img.shape[0]
        isos = np.random.uniform(*self.opts.iso_range, size=(N,))
        k, b = self.noise_func(isos, value_scale=self.opts.camera_value_scale)
        k_t = torch.as_tensor(k, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        b_t = torch.as_tensor(b, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)

        # The Poisson rate img/k_t is NOT generally <=1: img lives in [0, camera_value_scale]
        # (~0-959 for the Reno 10x defaults, confirmed by both this function's own docstring
        # and by k_sigma() using the exact same camera_value_scale-scaled tensors), while k_t
        # is only ~0.5-15 over the calibrated ISO range -- so img/k_t commonly reaches into the
        # hundreds for any non-trivial brightness. An upper clamp of 1 here (as this line
        # previously read: `.clamp(0, 1)`) was confirmed to crush the Poisson rate to ~1 for
        # virtually every non-near-black pixel, making shot_noisy ~= Poisson(1)*k_t --
        # essentially independent of img's actual value, i.e. not noise added to the image but
        # a near-total replacement of it with noise. Verified directly: with the old clamp, a
        # synthesized "noisy" mosaic was visually indistinguishable from pure static; with only
        # the (still necessary, to keep the Poisson rate non-negative) lower bound kept, the
        # same mosaic's underlying structure is clearly visible again, with mean abs diff vs.
        # the clean mosaic dropping ~13x (1.47e-1 -> 1.14e-2 on a [0,1] scale, ISO 3200). This
        # clamp(0, 1) bug is inherited from the original dataset/training.py (MegEngine) --
        # confirmed present there too (same `.clip(0, 1)`, line ~205) -- so it predates this
        # port and is not re-synced there per this repo's policy of no longer porting fixes to
        # the MegEngine training code (see this file's module docstring).
        shot_noisy = torch.poisson((img / k_t).clamp(min=0)) * k_t
        read_noisy = torch.randn(img.shape, device=self.device) * torch.sqrt(b_t)
        noisy = shot_noisy + read_noisy
        noisy = torch.round(noisy)

        return noisy, isos

# vim: ts=4 sw=4 sts=4 expandtab
