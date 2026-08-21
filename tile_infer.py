#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from utils import RawUtils
from dataset.benchmark import BenchmarkLoader
from run_benchmark_pytorch import KSigma, Denoiser, save_rgb_png, save_bayer_raw

# --dtype describes the ONNX graph's own declared EXTERNAL input/output tensor type --
# i.e. what numpy dtype must be fed in / comes back out -- not whether the graph is
# quantized internally. These two things are independent:
#
# - fp32 (default): the graph's 'input'/'output' are declared tensor(float). This covers
#   BOTH a plain export_onnx.py export AND an int8-quantized model from quantize_onnx.py:
#   quantize_onnx.py's default QuantFormat.QDQ wraps only the *internal* weights/activations
#   in QuantizeLinear/DequantizeLinear pairs (confirmed empirically -- sess.get_inputs()[0].type
#   is still 'tensor(float)' after quantization) and deliberately leaves the graph's outer
#   boundary as float32, so an int8-quantized model is fed/read exactly like a plain fp32
#   one from this script's point of view. Use --dtype fp32 for both.
# - fp16: export_onnx.py --dtype fp16 does a *graph-level* cast (onnxconverter_common,
#   keep_io_types=False) that changes the declared external tensor type itself to
#   float16, not just internal weights -- so this is the one case where the fed/read
#   numpy dtype genuinely has to change.
# - int8: NOT a real choice for anything this repo currently produces. No export path
#   here ever declares a genuinely int8 external input/output (quantize_onnx.py's QDQ
#   int8 models still declare tensor(float), same as fp32 -- see above), so there's no
#   dtype to actually feed. Kept as a rejected CLI choice (see main()) only to mirror
#   export_onnx.py's --dtype options and fail loudly instead of silently mishandling a
#   model this script doesn't know how to talk to, should one ever show up.
NP_DTYPE = {'fp32': np.float32, 'fp16': np.float16}
ONNX_TYPE = {'fp32': 'tensor(float)', 'fp16': 'tensor(float16)'}


def tile_positions(length, tile, stride):
    """1D tile start positions covering [0, length). The last tile is shifted back
    to end exactly at `length` instead of running past the edge."""
    if length <= tile:
        return [0]
    positions = [0]
    while positions[-1] + tile < length:
        nxt = positions[-1] + stride
        if nxt + tile >= length:
            nxt = length - tile
        positions.append(nxt)
        if nxt == length - tile:
            break
    return positions


def ownership_bounds(positions, tile, length):
    """Split the full [0, length) range into one non-overlapping [start, end) slice
    per tile, cut at the midpoint of every pair of neighboring tiles' overlap."""
    bounds = [0]
    for p_prev, p_next in zip(positions, positions[1:]):
        bounds.append((p_prev + tile + p_next) // 2)
    bounds.append(length)
    return bounds


def tiled_infer_rggb(sess, input_name, rggb, tile_h, tile_w, margin, np_dtype, extra_feed=None):
    """Run a full-size (H, W, 4) RGGB-plane image through a fixed tile_h x tile_w
    ONNX PMRID graph, splitting it into overlapping tiles and stitching the
    non-overlapping "ownership" region of each tile back together.

    This is the same tile_positions/ownership_bounds scheme used in
    DnCNN-PyTorch_Saoyan/tile_infer.py, generalized from a single grayscale
    channel to PMRID's 4-channel RGGB planes: each tile is (tile_h, tile_w, 4),
    transposed to (1, 4, tile_h, tile_w) for the ONNX input, and the network's
    (1, 4, tile_h, tile_w) output is transposed back before stitching.

    Note tile_h/tile_w are sizes in RGGB-plane space, i.e. *half* the resolution
    of the original bayer image in each dimension (bayer2rggb packs each 2x2
    bayer block into one RGGB pixel) -- a 256x256 ONNX tile therefore covers a
    512x512 region of the original raw image.

    `extra_feed`, if given, is merged into every tile's onnxruntime feed dict --
    used to pass a per-image, per-tile-invariant input (e.g. `iso`, for a
    --bake-ksigma graph) alongside the tile itself.
    """
    H, W = rggb.shape[:2]

    # if the image is smaller than one tile in either dimension, reflect-pad up to
    # tile size; this content is never cropped away by a neighboring tile's margin
    # (there is none), so it does leak into the receptive field near the true edge
    pad_h, pad_w = max(0, tile_h - H), max(0, tile_w - W)
    if pad_h or pad_w:
        rggb = np.pad(rggb, [(0, pad_h), (0, pad_w), (0, 0)], mode='reflect')
    Hp, Wp = rggb.shape[:2]

    stride_h = max(tile_h - 2 * margin, 1)
    stride_w = max(tile_w - 2 * margin, 1)
    ys = tile_positions(Hp, tile_h, stride_h)
    xs = tile_positions(Wp, tile_w, stride_w)
    y_bounds = ownership_bounds(ys, tile_h, Hp)
    x_bounds = ownership_bounds(xs, tile_w, Wp)

    out = np.zeros((Hp, Wp, 4), np.float32)  # accumulate in float32 regardless of --dtype; bookkeeping only, not model compute
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            tile = rggb[y0:y0 + tile_h, x0:x0 + tile_w].transpose(2, 0, 1)[np.newaxis].astype(np_dtype)
            feed = {input_name: tile, **(extra_feed or {})}
            tile_out = sess.run(None, feed)[0][0].transpose(1, 2, 0).astype(np.float32)

            oy0, oy1 = y_bounds[i], y_bounds[i + 1]
            ox0, ox1 = x_bounds[j], x_bounds[j + 1]
            out[oy0:oy1, ox0:ox1] = tile_out[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]

    grid_info = f'{len(ys)}x{len(xs)} tiles (tile {tile_h}x{tile_w}, margin {margin}, stride {stride_h}x{stride_w})'
    return out[:H, :W], grid_info  # crop back off any reflect-padding


class TiledOnnxDenoiser:
    """ONNX + tiling equivalent of run_benchmark_pytorch.py's Denoiser: same
    KSigma normalization / RGGB packing / inp_scale, but the network forward
    pass is replaced by tiled_infer_rggb over a fixed-shape onnxruntime session
    instead of one single-shot PyTorch forward over the whole (32-padded) image.

    If `bake_ksigma` is True, the ONNX graph in `sess` was exported with
    export_onnx.py --bake-ksigma: KSigma normalize/denormalize (and the
    inp_scale multiply/divide) live *inside* the graph, so this class must NOT
    also apply them externally -- doing so would double-normalize. Instead the
    raw [0,1] RGGB tile is fed straight into the graph, alongside a second
    `iso` input (the graph's second declared input, by convention named 'iso'
    -- see export_onnx.py), and the graph's output is already the final
    denoised RGGB, no un-scaling needed.
    """

    def __init__(self, sess: ort.InferenceSession, margin: int, ksigma: KSigma, dtype: str, inp_scale=256.0, bake_ksigma=False):
        self.sess = sess
        self.input_name = sess.get_inputs()[0].name
        self.tile_h, self.tile_w = sess.get_inputs()[0].shape[2:]
        self.margin = margin
        self.ksigma = ksigma
        self.np_dtype = NP_DTYPE[dtype]
        self.inp_scale = inp_scale
        self.bake_ksigma = bake_ksigma
        if bake_ksigma:
            self.iso_name = sess.get_inputs()[1].name

    def run(self, bayer_01: np.ndarray, iso: float):
        # reflect-pad the RAW bayer mosaic by `margin` RGGB-plane pixels (= 2*margin
        # raw pixels, always even so bayer2rggb's color assignment stays correct --
        # see Denoiser.pre_process in run_benchmark_pytorch.py for why 'reflect'
        # specifically, not zero or edge-duplicating reflect) so the outermost tile's
        # true-image-edge side also gets real, mirrored context instead of relying
        # solely on the network's own implicit zero-padding right at the true edge.
        pad = self.margin
        bayer_01 = np.pad(bayer_01, [(2*pad, 2*pad), (2*pad, 2*pad)], mode='reflect')

        rggb = RawUtils.bayer2rggb(bayer_01).clip(0, 1)

        if self.bake_ksigma:
            iso_arr = np.array([iso], dtype=self.np_dtype)
            pred_rggb, grid_info = tiled_infer_rggb(
                self.sess, self.input_name, rggb.astype(np.float32),
                self.tile_h, self.tile_w, self.margin, self.np_dtype,
                extra_feed={self.iso_name: iso_arr},
            )
        else:
            rggb = self.ksigma(rggb, iso) * self.inp_scale
            pred_rggb, grid_info = tiled_infer_rggb(
                self.sess, self.input_name, rggb.astype(np.float32),
                self.tile_h, self.tile_w, self.margin, self.np_dtype,
            )
            pred_rggb = pred_rggb / self.inp_scale
            pred_rggb = self.ksigma(pred_rggb, iso, inverse=True)

        pred_rggb = pred_rggb[pad:-pad, pad:-pad]  # crop the edge buffer back off
        return RawUtils.rggb2bayer(pred_rggb), grid_info


def main():
    parser = argparse.ArgumentParser(description="Tiled inference through a fixed-shape PMRID ONNX export, over real (arbitrary-size) raw images")
    parser.add_argument('--onnx', type=Path, required=True, help='fixed-shape ONNX model exported by export_onnx.py (without a dynamic H/W)')
    parser.add_argument('--benchmark', type=Path, required=True, help='benchmark.json/meta_info.json-style dataset description (same format as run_benchmark_pytorch.py)')
    parser.add_argument('--save-dir', type=Path, required=True, help='directory to save the denoised RGB visualization (png, full image) and bayer output (.raw) per sample')
    parser.add_argument(
        '--margin', type=int, default=32,
        help='pixels (in RGGB-plane space, i.e. half the bayer resolution) discarded from each '
             'tile edge before stitching neighboring tiles together. PMRID is a multi-scale U-Net '
             '(4 downsample/upsample stages), so unlike a plain CNN there is no single exact '
             'receptive-field number to set this to -- start around here and tune based on '
             '--compare-full.',
    )
    parser.add_argument(
        '--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'],
        help="the ONNX graph's own declared EXTERNAL input/output tensor dtype (what numpy "
             "dtype must be fed in/read back), NOT whether it's quantized internally -- pass "
             "'fp32' for both a plain export_onnx.py export AND an int8-quantized "
             "quantize_onnx.py model (its QDQ format keeps external I/O as float32; see the "
             "NP_DTYPE/ONNX_TYPE comment above main() for why). 'fp16' is for an "
             "export_onnx.py --dtype fp16 export specifically (a real graph-level type "
             "change). 'int8' is rejected -- no export in this repo ever declares a "
             "genuinely int8 external input/output.",
    )
    parser.add_argument(
        '--compare-full', type=Path, default=None, metavar='TORCH_MODEL',
        help='also run each sample through the full (untiled) PyTorch model at this checkpoint '
             'path (e.g. models/torch_pretrained.ckp) and report the difference vs the tiled ONNX '
             'result, to help tune --margin',
    )
    parser.add_argument(
        '--bake-ksigma', action='store_true',
        help='must be set if (and only if) the ONNX model in --onnx was exported with '
             'export_onnx.py --bake-ksigma. When set, tile_infer.py skips its own external KSigma '
             'step and instead feeds each sample\'s ISO straight into the graph as its second '
             'input, per tile, since normalize/denormalize already happen inside the graph.',
    )
    args = parser.parse_args()

    if args.dtype == 'int8':
        raise NotImplementedError(
            "--dtype int8 is rejected: no ONNX export this repo produces ever declares a "
            "genuinely int8 external input/output -- an int8-quantized quantize_onnx.py model "
            "still declares tensor(float) I/O (QDQ format), so pass --dtype fp32 for it too "
            "(see the NP_DTYPE/ONNX_TYPE comment above main() for the full explanation)."
        )

    sess = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    tile_h, tile_w = sess.get_inputs()[0].shape[2:]
    if not isinstance(tile_h, int) or not isinstance(tile_w, int):
        raise ValueError(
            f'tile_infer.py needs a fixed-shape ONNX export (export_onnx.py bakes in '
            f'--height/--width); got dynamic shape {tile_h!r} x {tile_w!r}'
        )
    onnx_input_type = sess.get_inputs()[0].type
    if onnx_input_type != ONNX_TYPE[args.dtype]:
        raise ValueError(
            f"--dtype {args.dtype} doesn't match the ONNX model's actual input type "
            f"{onnx_input_type} -- pass the --dtype it was exported with"
        )
    num_inputs = len(sess.get_inputs())
    if args.bake_ksigma and num_inputs != 2:
        raise ValueError(
            f'--bake-ksigma was passed but the ONNX model in --onnx only declares {num_inputs} '
            f'input(s) -- it was not exported with export_onnx.py --bake-ksigma'
        )
    if not args.bake_ksigma and num_inputs != 1:
        raise ValueError(
            f'the ONNX model in --onnx declares {num_inputs} inputs, which looks like it was '
            f'exported with export_onnx.py --bake-ksigma -- pass --bake-ksigma here too'
        )

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = TiledOnnxDenoiser(sess, args.margin, ksigma, args.dtype, bake_ksigma=args.bake_ksigma)

    full_denoiser = None
    if args.compare_full is not None:
        full_denoiser = Denoiser(args.compare_full, ksigma)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    bm_loader = BenchmarkLoader(args.benchmark.resolve())

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, meta in bar:
        bar.set_description(meta.name)
        pattern = meta.bayer_pattern
        input_bayer, gt_bayer = RawUtils.to_canonical_rggb(input_bayer, gt_bayer, pattern=pattern)

        pred_bayer, grid_info = denoiser.run(input_bayer, iso=meta.ISO)

        msg = f'{meta.name}  [{grid_info}]'
        if full_denoiser is not None:
            full_pred_bayer = full_denoiser.run(input_bayer, iso=meta.ISO)
            diff = np.abs(pred_bayer.astype(np.float64) - full_pred_bayer.astype(np.float64))
            msg += f'  vs full (untiled) PyTorch: max abs diff {diff.max():.3e}, mean abs diff {diff.mean():.3e} (0~1 scale)'
        tqdm.write(msg)

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
            input_bayer, pred_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )
        inp_rgb, pred_rgb, gt_rgb = RawUtils.to_canonical_rggb(inp_rgb, pred_rgb, gt_rgb, pattern=pattern)

        save_rgb_png(args.save_dir / f'{meta.name}_input.png', inp_rgb)
        save_rgb_png(args.save_dir / f'{meta.name}_pred.png', pred_rgb)
        save_rgb_png(args.save_dir / f'{meta.name}_gt.png', gt_rgb)

        pred_bayer_orig_pattern = RawUtils.to_canonical_rggb(pred_bayer, pattern=pattern)
        save_bayer_raw(args.save_dir / f'{meta.name}_pred.raw', pred_bayer_orig_pattern, meta.raw_bitWidth)


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
