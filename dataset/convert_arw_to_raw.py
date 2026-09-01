#!/usr/bin/env python3
"""Convert Sony .ARW raw images under INPUT_DIR to plain uint16 .raw bayer files under
OUTPUT_DIR, preserving the directory structure, with each output filename suffixed
_H_<height>_W_<width> (the raw bayer mosaic's pixel shape). Also runs a simple ISP on
each image (white balance from the camera's own as-shot metadata, demosaic, gamma -- no
color-correction matrix, since this sensor has none calibrated in this repo) and saves a
PNG under PNG_DIR, purely as a visual sanity check that the .raw conversion looks right --
not a colorimetrically accurate render (same caveat as this repo's other visualization
code). Finally writes an `index.json` under OUTPUT_DIR listing every converted image in
the RawImageItem format dataset/training_pytorch.py::CleanRawImages expects, so
OUTPUT_DIR can be passed directly as train_pytorch.py's/train_qat_pytorch.py's --data-dir.

Reads each file's raw bayer mosaic via rawpy/LibRaw's `raw_image_visible` (the active
sensor area, with the optical-black border rows/columns already excluded) -- not a
demosaiced/processed image -- matching this repo's convention elsewhere (dataset/
benchmark.py, dataset/training_pytorch.py) of storing single-channel bayer arrays as
plain uint16 binary files.

ISP-relevant metadata actually available per .ARW file (via rawpy/LibRaw; checked against
a real file from this dataset): `black_level_per_channel` (per-CFA-position black level,
e.g. [512,512,512,512] here -- uniform across channels for this sensor, so collapsed to
one scalar to match RawImageItem.black_level's schema, which has no per-channel field;
`main()` warns if a file's channels actually disagree, rather than silently averaging
over a real difference), `white_level`/`camera_white_level_per_channel` (16383 here, i.e.
this is a 14-bit sensor stored in a uint16 container -- also collapsed to the file's
black_level/white_level fields), `camera_whitebalance`/`daylight_whitebalance` (as-shot
and daylight-preset WB gains, `[R,G,B,G2]`), `raw_pattern`/`color_desc` (the 2x2 CFA
layout -- RGGB for this camera, decoded below into one of RawUtils.BAYER_PATTERNS),
`rgb_xyz_matrix` (sensor-RGB-to-XYZ color matrix -- no equivalent calibrated CCM exists
elsewhere in this repo for this sensor, so the simple ISP below uses an identity CCM
instead of this), and `tone_curve` (linear/identity here, i.e. this file has no baked-in
tone mapping). None of this needs a separate EXIF library -- ISO/exposure-time/aperture/
etc. are regular EXIF tags rawpy doesn't expose and would need e.g. `exifread` (not
installed here) to read, but nothing in RawImageItem's schema needs them.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import RawUtils, round_half_up

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "SID_Sony"
OUTPUT_DIR = SCRIPT_DIR / "SID_Sony_raw"
PNG_DIR = SCRIPT_DIR / "SID_Sony_png"
INDEX_JSON_PATH = OUTPUT_DIR / "index.json"


def save_rgb_png(path: Path, rgb_01: np.ndarray):
    bgr_uint8 = round_half_up(rgb_01.clip(0, 1)[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(path), bgr_uint8)


def bayer_pattern_from_rawpy(raw) -> str:
    """Derive one of RawUtils.BAYER_PATTERNS ('RGGB'/'BGGR'/'GRBG'/'GBRG') from rawpy's
    own CFA description: `raw_pattern` is a 2x2 tile of color indices, `color_desc` says
    which letter each index maps to. Reading the tile top-left/top-right/bottom-left/
    bottom-right gives exactly the 4-letter name these pattern strings encode.
    """
    pattern = raw.raw_pattern
    desc = raw.color_desc.decode()
    tile = ''.join(desc[pattern[r, c]] for r in range(2) for c in range(2))
    if tile not in RawUtils.BAYER_PATTERNS:
        raise ValueError(
            f"unrecognized/unsupported CFA layout {tile!r} "
            f"(raw_pattern={pattern.tolist()}, color_desc={desc!r})"
        )
    return tile


def convert_one(src: Path, raw_dst_dir: Path, png_dst_dir: Path) -> dict:
    with rawpy.imread(str(src)) as raw:
        bayer = raw.raw_image_visible.copy()
        black_levels = np.array(raw.black_level_per_channel, dtype=np.float64)
        if black_levels.std() > 1.0:
            print(f"WARNING: {src} has non-uniform black_level_per_channel "
                  f"{raw.black_level_per_channel} -- averaging anyway, since "
                  f"RawImageItem only has one scalar black_level field")
        black_level = float(black_levels.mean())
        white_level = float(raw.white_level)
        # camera_whitebalance is [R, G, B, G2] read from the file's own as-shot
        # metadata; normalize by G so it matches this repo's wb_gain convention
        # (G == 1.0, e.g. dataset/benchmark/benchmark.json's per-sample wb_gain)
        cam_wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        wb_gain = cam_wb / cam_wb[1]
        bayer_pattern = bayer_pattern_from_rawpy(raw)

    H, W = bayer.shape
    raw_dst_dir.mkdir(parents=True, exist_ok=True)
    raw_dst_path = raw_dst_dir / f"{src.stem}_H_{H}_W_{W}.raw"
    bayer.astype(np.uint16).tofile(raw_dst_path)

    bayer_01 = ((bayer.astype(np.float32) - black_level) / (white_level - black_level)).clip(0, 1)
    # flip to the canonical RGGB layout RawUtils.bayer2rggb/bayer2rgb assume (a no-op for
    # this camera, since its own CFA already is RGGB, but doing it explicitly -- rather
    # than relying on that coincidence -- matches how every other script in this repo
    # handles a sample's bayer_pattern, e.g. run_benchmark_pytorch.py's run_benchmark)
    canonical_01 = RawUtils.to_canonical_rggb(bayer_01, pattern=bayer_pattern)
    g_mean_01 = float(RawUtils.bayer2rggb(canonical_01)[..., [1, 2]].mean())

    rgb = RawUtils.bayer2rgb(canonical_01, wb_gain=wb_gain, CCM=np.eye(3), gamma=2.2)
    rgb = RawUtils.to_canonical_rggb(rgb, pattern=bayer_pattern)

    png_dst_dir.mkdir(parents=True, exist_ok=True)
    png_dst_path = png_dst_dir / f"{src.stem}_H_{H}_W_{W}.png"
    save_rgb_png(png_dst_path, rgb)

    print(f"{src} -> {raw_dst_path}, {png_dst_path}  ({H}x{W}, pattern={bayer_pattern})")

    return {
        "path": raw_dst_path.relative_to(OUTPUT_DIR).as_posix(),
        "width": W,
        "height": H,
        "black_level": round(black_level),
        "white_level": round(white_level),
        "bayer_pattern": bayer_pattern,
        "g_mean_01": g_mean_01,
    }


def main():
    arw_paths = sorted(INPUT_DIR.rglob("*.ARW"))
    print(f"found {len(arw_paths)} .ARW files under {INPUT_DIR}")

    index = []
    for src in arw_paths:
        rel_dir = src.parent.relative_to(INPUT_DIR)
        index.append(convert_one(src, OUTPUT_DIR / rel_dir, PNG_DIR / rel_dir))

    INDEX_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_JSON_PATH, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"wrote {len(index)}-entry index.json to {INDEX_JSON_PATH}")


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
