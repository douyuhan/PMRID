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

    def transform(self, batch_img01: torch.Tensor, batch_g_mean) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            - batch_img01: (N, 4, H, W) float32 tensor, clean RGGB in [0, 1] (as returned
              by CleanRawImages, batched by the DataLoader's default collate)
            - batch_g_mean: (N,) per-sample mean green-channel brightness in [0, 1]

        Returns:
            - noisy_img (network input)
            - gt (network target)
            - norm_k (per-sample loss normalization factor, for get_loss_l1)
        """
        batch_imgs = batch_img01.to(self.device, dtype=torch.float32) * self.opts.camera_value_scale
        batch_gt = self.brightness_aug(batch_imgs, batch_g_mean)
        batch_imgs, batch_iso = self.add_noise(batch_gt)
        cvt_k, cvt_b = self.k_sigma(batch_iso)

        batch_imgs = batch_imgs * cvt_k + cvt_b
        batch_gt = batch_gt * cvt_k + cvt_b
        return batch_imgs, batch_gt, cvt_k

    def k_sigma(self, iso: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        k, sigma = self.noise_func(iso, value_scale=self.opts.camera_value_scale)
        k_a, sigma_a = self.noise_func(self.opts.anchor_iso, value_scale=self.opts.camera_value_scale)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        cvt_k = torch.as_tensor(cvt_k, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        cvt_b = torch.as_tensor(cvt_b, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        return cvt_k, cvt_b

    def brightness_aug(self, img_batch: torch.Tensor, orig_gmean) -> torch.Tensor:
        low, high = self.opts.target_brighness_range
        orig_gmean = np.asarray(orig_gmean)
        N = len(orig_gmean)
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N,)))
        s = np.clip(btarget / orig_gmean, 0.01, 1.0)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
        return img_batch * s

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

        shot_noisy = torch.poisson((img / k_t).clamp(0, 1)) * k_t
        read_noisy = torch.randn(img.shape, device=self.device) * torch.sqrt(b_t)
        noisy = shot_noisy + read_noisy
        noisy = torch.round(noisy)

        return noisy, isos

# vim: ts=4 sw=4 sts=4 expandtab
