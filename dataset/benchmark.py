#!/usr/bin/env python3
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from utils import RawUtils

# applied when a sample's meta dict has no 'wb_gain'/'CCM' key at all -- e.g. a dataset
# whose sensor calibration isn't known yet -- so RawMeta can still be constructed instead
# of raising a KeyError/TypeError. (1.0, 1.0, 1.0) is a no-op WB gain (RawUtils.bayer2rgb's
# wb_gain[[0,1,1,2]] expansion multiplies every channel by 1.0); np.eye(3) is a no-op CCM
# (same "assume identity when no calibrated matrix exists" convention already used by
# dataset/convert_arw_to_raw.py for the SID Sony sensor).
DEFAULT_WB_GAIN = (1.0, 1.0, 1.0)
DEFAULT_CCM = tuple(tuple(row) for row in np.eye(3).tolist())


def read_array(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint16)


@dataclass
class RawMeta:
    name: str
    scene_id: str
    light: str
    ISO: int
    exp_time: float
    bayer_pattern: str
    shape: Tuple[int, int]
    ROIs: List[Tuple[int, int, int, int]]
    # Optional, not because a missing calibration is fine to use silently (it isn't --
    # see __post_init__'s warning), but because dataclass field ordering requires
    # defaulted fields to come after non-defaulted ones; None is just the "not provided"
    # sentinel that __post_init__ replaces with DEFAULT_WB_GAIN/DEFAULT_CCM.
    wb_gain: Optional[Tuple[float, float, float]] = None
    CCM: Optional[Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]] = None
    raw_bitWidth: int = 16

    def __post_init__(self):
        if self.wb_gain is None:
            warnings.warn(
                f"{self.name}: meta has no 'wb_gain' -- defaulting to {DEFAULT_WB_GAIN} "
                f"(no-op WB). PSNR/SSIM and saved visualizations for this sample will be "
                f"computed without any real white-balance correction."
            )
            self.wb_gain = DEFAULT_WB_GAIN
        if self.CCM is None:
            warnings.warn(
                f"{self.name}: meta has no 'CCM' -- defaulting to the identity matrix "
                f"(no-op CCM). PSNR/SSIM and saved visualizations for this sample will be "
                f"computed without any real color correction."
            )
            self.CCM = DEFAULT_CCM


class BenchmarkLoader:

    def __init__(self, dataset_info_json: Path, base_path=None):
        with dataset_info_json.open() as f:
            self._dataset = [
                {
                    'input': x['input'],
                    'gt': x['gt'],
                    'meta': RawMeta(**x['meta'])
                }
                for x in json.load(f)
            ]
        if base_path is None:
            self.base_path = dataset_info_json.parent
        else:
            self.base_path = Path(base_path)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, RawMeta]:
        if self._idx >= len(self):
            raise StopIteration

        input_bayer, gt_bayer, meta = self._load_idx(self._idx)

        self._idx += 1
        return input_bayer, gt_bayer, meta

    def _load_idx(self, idx: int):
        item = self._dataset[idx]

        img_files = item['input'], item['gt']
        meta = item['meta']
        bayers = []
        for img_file in img_files:
            if not Path(img_file).is_absolute():
                img_file = self.base_path / img_file
            bayer = read_array(img_file)
            bayer = bayer.reshape(*meta.shape)
            assert meta.bayer_pattern in RawUtils.BAYER_PATTERNS, \
                f'unsupported bayer_pattern: {meta.bayer_pattern!r}'
            # hardware normalizes via a bit-shift (divide by 2**bit), not by the
            # true max representable value (2**bit - 1) -- see round_half_up's
            # docstring in utils.py for the matching quantize-side convention
            bayer = bayer.astype(np.float32) / (2 ** meta.raw_bitWidth)
            bayers.append(bayer)

        input_bayer, gt_bayer = bayers
        return input_bayer, gt_bayer, meta

# vim: ts=4 sw=4 sts=4 expandtab
