#!/usr/bin/env python3
"""Fixed-point ("hardware") variant of tile_infer.py's pre/post-processing.

Motivation: everywhere else in this repo, the denoise network's boundary is
float32 in [0,1] (or an inp_scale-scaled variant of it) -- fine for a CPU/
onnxruntime deployment, but not for slotting PMRID into an otherwise fully
fixed-point ISP pipeline as a drop-in denoise module, where the upstream/
downstream modules hand off RAW pixel data as `bit_module`-bit integers
(default 12, i.e. RAW12) and expect the same back, with no floating point
anywhere in between.

This script reimplements exactly that boundary: the RAW bit-depth conversion,
KSigma normalize/denormalize, and the requantize back to RAW pixel data are
all integer-only fixed-point arithmetic. The network itself still goes
through onnxruntime (this repo's actual deployment target has its own
existing int8 inference software stack, reached the same way as in
tile_infer.py -- see quantize_onnx.py), whose graph declares tensor(float)
input/output regardless of internal int8 quantization (see tile_infer.py's
--dtype comment). So there is exactly ONE unavoidable float32 crossing: right
before/after the onnxruntime call. This is NOT a hidden floating-point
computation step -- it's a lossless re-interpretation of an already-computed
fixed-point value into the container type the graph requires
(val_fp = val_fxp / (1<<scale), and the reverse coming back), never fresh
arithmetic.

Visualization (PNG) and metrics are left exactly as in run_benchmark_pytorch.py
/ tile_infer.py, operating on float32 [0,1] as before, and `.raw` is still
saved at each sample's own `raw_bitWidth` (via the existing save_bayer_raw) --
only the network's own pre/post-processing is fixed-point; its result is
converted back to float32 [0,1] before handing off to that existing code.

Fixed-point convention throughout, exactly as specified: given `bit_int`
integer bits and `bit_frac` fractional bits,
    unsigned: val_fxp = clip(round(val_fp * (1<<bit_frac)), 0, (1<<(bit_int+bit_frac))-1)
    signed:   val_fxp = clip(round(val_fp * (1<<bit_frac)), -(1<<(bit_int+bit_frac)), (1<<(bit_int+bit_frac))-1)
    val_fp  = val_fxp * 1.0 / (1<<bit_frac)
`round` is round-half-away-from-zero (this repo's round_half_up, and C's own
round() semantics), for consistency with every other fixed-point conversion
already in this repo (see utils.py::round_half_up).
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from utils import RawUtils, round_half_up
from dataset.benchmark import BenchmarkLoader
from run_benchmark_pytorch import KSigma, Denoiser, save_rgb_png, save_bayer_raw
from tile_infer import tiled_infer_rggb, NP_DTYPE, ONNX_TYPE


# ---------------------------------------------------------------------------
# generic fixed-point helpers
# ---------------------------------------------------------------------------

def to_fixed(val_fp, bit_int, bit_frac, signed=False):
    """float -> fixed-point integer, per the convention in the module docstring.
    Always returned as int64 regardless of bit_int/bit_frac -- the clip range
    (not the container dtype) is what actually models a specific hardware
    register's width and its saturation behavior."""
    scaled = round_half_up(np.asarray(val_fp, dtype=np.float64) * (1 << bit_frac))
    if signed:
        lo, hi = -(1 << (bit_int + bit_frac)), (1 << (bit_int + bit_frac)) - 1
    else:
        lo, hi = 0, (1 << (bit_int + bit_frac)) - 1
    return np.clip(scaled, lo, hi).astype(np.int64)


def from_fixed(val_fxp, bit_frac):
    """fixed-point integer -> float64. A lossless re-interpretation of an
    already-computed fixed-point value (see module docstring) -- used both for
    the one unavoidable float32 handoff to onnxruntime, and to convert the
    final denoised result back to float32 [0,1] for the existing
    visualization/save/metrics code."""
    return np.asarray(val_fxp, dtype=np.float64) / (1 << bit_frac)


def round_shift(x_int, shift):
    """Right-shift an integer array by `shift` bits (>=0) with round-half-
    away-from-zero -- NOT truncation, and NOT numpy/Python's native `>>`
    (which floors toward -infinity, i.e. rounds negative halves toward zero
    instead of away from it -- the same banker's-rounding-style bias
    round_half_up was written to avoid, just in the integer/shift domain)."""
    x_int = np.asarray(x_int, dtype=np.int64)
    if shift == 0:
        return x_int
    half = 1 << (shift - 1)
    return np.sign(x_int) * ((np.abs(x_int) + half) >> shift)


def fxp_mul(a_fxp, b_fxp, a_frac, b_frac, out_frac):
    """Fixed-point multiply of two integers (a_fxp with a_frac fractional
    bits, b_fxp with b_frac fractional bits), rescaled down to out_frac
    fractional bits with rounding. Computed in int64 -- comfortably wide
    enough for every bit width this script uses; a real hardware MAC's
    accumulator just needs to be at least as wide, same idea."""
    product = np.asarray(a_fxp, dtype=np.int64) * np.int64(b_fxp)
    return round_shift(product, a_frac + b_frac - out_frac)


def rescale_bitdepth(value_int, from_bits, to_bits):
    """Linearly rescale an unsigned integer RAW image from one bit depth to
    another, entirely in integer arithmetic. from_bits/to_bits are always
    plain bit-depths, so the ratio between their full-scales, (1<<from_bits)
    and (1<<to_bits), is always exactly a power of two -- this is therefore
    always an exact shift, never a non-power-of-two fixed-point multiply/
    divide: a lossless left-shift when increasing bit depth, or a round-half-
    away-from-zero right-shift (the genuine, expected precision loss) when
    decreasing it."""
    shift = from_bits - to_bits
    if shift > 0:
        rescaled = round_shift(value_int, shift)
    elif shift < 0:
        rescaled = np.asarray(value_int, dtype=np.int64) << (-shift)
    else:
        rescaled = np.asarray(value_int, dtype=np.int64)
    return np.clip(rescaled, 0, (1 << to_bits) - 1)


# ---------------------------------------------------------------------------
# KSigma, collapsed into a single per-frame affine transform
# ---------------------------------------------------------------------------
# run_benchmark_pytorch.py's KSigma.__call__, combined with the inp_scale
# multiply (forward) / divide (inverse) that always brackets it, algebraically
# collapses into ONE affine transform each way -- V cancels out completely:
#   forward: net_input = img_01     * A(iso) + B(iso)
#   inverse: img_01     = net_output * C(iso) + D(iso)
# iso is a per-frame scalar (not per-pixel), so A/B/C/D are each computed once
# per frame, at full float64 precision -- this IS the "LUT": a per-frame pair
# of pre-computed constants, not a per-pixel polynomial-plus-division. Only
# these four scalars get quantized to fixed-point; every per-pixel operation
# after that is a single fixed-point multiply-add, no division anywhere.

def compute_forward_affine(ksigma: KSigma, iso: float, inp_scale: float):
    k, sigma = ksigma.K(iso), ksigma.Sigma(iso)
    k_a, sigma_a = ksigma.K(ksigma.anchor), ksigma.Sigma(ksigma.anchor)
    cvt_k = k_a / k
    cvt_b = (sigma / k ** 2 - sigma_a / k_a ** 2) * k_a
    A = cvt_k * inp_scale
    B = cvt_b * inp_scale / ksigma.V
    return A, B


def compute_inverse_affine(ksigma: KSigma, iso: float, inp_scale: float):
    k, sigma = ksigma.K(iso), ksigma.Sigma(iso)
    k_a, sigma_a = ksigma.K(ksigma.anchor), ksigma.Sigma(ksigma.anchor)
    cvt_k = k_a / k
    cvt_b = (sigma / k ** 2 - sigma_a / k_a ** 2) * k_a
    C = 1.0 / (inp_scale * cvt_k)
    D = -cvt_b / (ksigma.V * cvt_k)
    return C, D


def run_fixed_point(sess, input_name, tile_h, tile_w, margin, np_dtype,
                     input_bayer_int, iso, ksigma, bit_module, net_bit_int, net_bit_frac):
    """Fixed-point equivalent of tile_infer.py's TiledOnnxDenoiser.run: same
    reflect-pad -> bayer2rggb -> normalize -> tiled inference -> un-normalize
    -> rggb2bayer flow, but every pre/post-processing step is integer-only
    fixed-point arithmetic instead of float32.

    `input_bayer_int` is the RAW image, already in canonical RGGB bayer
    orientation, as an unsigned Q(0.bit_module) integer array (already
    rescaled to bit_module via rescale_bitdepth). Returns the denoised result,
    ALSO as an unsigned Q(0.bit_module) integer bayer array -- exactly what a
    real downstream fixed-point ISP module would receive.
    """
    pad = margin
    bayer_padded = np.pad(input_bayer_int, [(2 * pad, 2 * pad), (2 * pad, 2 * pad)], mode='reflect')
    rggb_int = RawUtils.bayer2rggb(bayer_padded)  # reshape only; stays unsigned Q(0.bit_module)

    net_int_lo, net_int_hi = -(1 << (net_bit_int + net_bit_frac)), (1 << (net_bit_int + net_bit_frac)) - 1

    A_fp, B_fp = compute_forward_affine(ksigma, iso, inp_scale=256.0)
    A_fxp = to_fixed(A_fp, net_bit_int, net_bit_frac, signed=True)
    B_fxp = to_fixed(B_fp, net_bit_int, net_bit_frac, signed=True)

    # net_input = rggb_int (Q0.bit_module, unsigned) * A_fxp (Q net_bit_int.net_bit_frac, signed) + B_fxp
    net_input_fxp = fxp_mul(rggb_int, A_fxp, bit_module, net_bit_frac, net_bit_frac) + B_fxp
    net_input_fxp = np.clip(net_input_fxp, net_int_lo, net_int_hi)  # model this bit width's own saturation

    # the one unavoidable float32 handoff -- see module docstring
    net_input_f32 = from_fixed(net_input_fxp, net_bit_frac).astype(np_dtype)
    pred_f32, grid_info = tiled_infer_rggb(sess, input_name, net_input_f32, tile_h, tile_w, margin, np_dtype)
    net_output_fxp = to_fixed(pred_f32.astype(np.float64), net_bit_int, net_bit_frac, signed=True)

    C_fp, D_fp = compute_inverse_affine(ksigma, iso, inp_scale=256.0)
    C_fxp = to_fixed(C_fp, net_bit_int, net_bit_frac, signed=True)
    D_fxp = to_fixed(D_fp, net_bit_int, net_bit_frac, signed=True)

    img01_fxp = fxp_mul(net_output_fxp, C_fxp, net_bit_frac, net_bit_frac, net_bit_frac) + D_fxp
    img01_fxp = np.clip(img01_fxp, net_int_lo, net_int_hi)

    # requantize from signed Q(net_bit_int.net_bit_frac) back down to the
    # unsigned Q(0.bit_module) RAW domain, clipping any negative/over-1.0
    # excursion from noise/residual -- the fixed-point analog of
    # save_bayer_raw's `bayer_01.clip(0, 1) * max_val`
    if net_bit_frac >= bit_module:
        rggb_out = round_shift(img01_fxp, net_bit_frac - bit_module)
    else:
        rggb_out = img01_fxp << (bit_module - net_bit_frac)
    rggb_out = np.clip(rggb_out, 0, (1 << bit_module) - 1)

    rggb_out = rggb_out[pad:-pad, pad:-pad]
    return RawUtils.rggb2bayer(rggb_out), grid_info


def main():
    parser = argparse.ArgumentParser(description="Fixed-point ('hardware') pre/post-processing variant of tile_infer.py")
    parser.add_argument('--onnx', type=Path, required=True, help='fixed-shape ONNX model exported WITHOUT --bake-ksigma -- this script applies KSigma itself, in fixed point; a baked-KSigma graph would double-apply it')
    parser.add_argument('--benchmark', type=Path, required=True, help='benchmark.json/meta_info.json-style dataset description (same format as the other scripts)')
    parser.add_argument('--save-dir', type=Path, required=True, help='directory to save the denoised RGB visualization (png, full image) and bayer output (.raw, at the sample\'s own raw_bitWidth) per sample')
    parser.add_argument(
        '--margin', type=int, default=32,
        help='pixels (in RGGB-plane space) discarded from each tile edge before stitching -- same meaning as tile_infer.py',
    )
    parser.add_argument(
        '--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'],
        help="the ONNX graph's own declared EXTERNAL tensor dtype -- same meaning as tile_infer.py's "
             "--dtype (fp32 covers both a plain export and a quantize_onnx.py int8/QDQ model, since "
             "QDQ format keeps external I/O as float32; see tile_infer.py for the full explanation). "
             "int8 is rejected for the same reason as there.",
    )
    parser.add_argument(
        '--bit-module', type=int, default=12,
        help="RAW module interface bit depth (default 12, i.e. RAW12) -- the width the upstream/"
             "downstream fixed-point ISP modules actually hand off/expect at the denoise island's "
             "boundary. If a sample's meta.raw_bitWidth differs from this, its data is linearly "
             "rescaled to this width first (an exact bit-shift, since both are plain bit-depths, "
             "hence always a power-of-two ratio -- never an approximate/lossy general rescale).",
    )
    parser.add_argument(
        '--net-bit-int', type=int, default=13,
        help="integer bits of the SIGNED fixed-point format used for the KSigma-normalized network "
             "input/output domain and its per-frame A/B/C/D coefficients (see compute_forward_affine/"
             "compute_inverse_affine). Must be wide enough to cover the largest |value| these take "
             "across your real deployment's ISO range -- e.g. at ISO 100 (near this model's low end), "
             "A can reach roughly ~3600, needing >=12 integer bits; the default of 13 leaves headroom. "
             "Tune this down once you know your real deployment's actual ISO range, to match your "
             "hardware's real register width.",
    )
    parser.add_argument(
        '--net-bit-frac', type=int, default=18,
        help='fractional bits of the same signed fixed-point format -- precision of the per-frame '
             'KSigma LUT coefficients and the network input/output value. Cheap to make generous: '
             'these are a handful of per-frame scalars plus one value per pixel, not a wide per-pixel '
             'weight tensor.',
    )
    parser.add_argument(
        '--compare-full', type=Path, default=None, metavar='TORCH_MODEL',
        help='also run each sample through the full (untiled) float32 PyTorch Denoiser and report the '
             'difference vs this fixed-point pipeline\'s result (on the same [0,1] float scale used '
             'elsewhere in this repo) -- use this to check the fixed-point implementation/bit widths '
             'against the float reference, the same way tile_infer.py uses it to check tiling.',
    )
    args = parser.parse_args()

    if args.dtype == 'int8':
        raise NotImplementedError("--dtype int8 is rejected (see tile_infer.py's --dtype comment for why)")

    sess = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError(
            f'{args.onnx} declares {len(inputs)} inputs -- this script assumes a plain (1-input) graph, '
            f'exported WITHOUT export_onnx.py --bake-ksigma, since it applies KSigma itself in fixed point.'
        )
    tile_h, tile_w = inputs[0].shape[2:]
    if not isinstance(tile_h, int) or not isinstance(tile_w, int):
        raise ValueError(f'needs a fixed-shape ONNX export; got dynamic shape {tile_h!r} x {tile_w!r}')
    onnx_input_type = inputs[0].type
    if onnx_input_type != ONNX_TYPE[args.dtype]:
        raise ValueError(
            f"--dtype {args.dtype} doesn't match the ONNX model's actual input type "
            f"{onnx_input_type} -- pass the --dtype it was exported with"
        )
    input_name = inputs[0].name
    np_dtype = NP_DTYPE[args.dtype]

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    full_denoiser = Denoiser(args.compare_full, ksigma) if args.compare_full is not None else None

    args.save_dir.mkdir(parents=True, exist_ok=True)
    bm_loader = BenchmarkLoader(args.benchmark.resolve())

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, meta in bar:
        bar.set_description(meta.name)
        pattern = meta.bayer_pattern
        input_bayer, gt_bayer = RawUtils.to_canonical_rggb(input_bayer, gt_bayer, pattern=pattern)

        # Recover the exact original raw_bitWidth integer from BenchmarkLoader's
        # float32 [0,1] normalization (which divides by 2**raw_bitWidth, the
        # same shift-friendly convention used throughout this script) --
        # lossless: float32's 24-bit mantissa comfortably exceeds any realistic
        # raw_bitWidth (12-16 bits), so this round-trip introduces no error
        # beyond what round_half_up already accounts for. This is specifically
        # un-doing BenchmarkLoader's own normalization constant, not applying
        # the (1<<bit_frac) fixed-point convention used everywhere else in this
        # script -- the genuinely fixed-point part of this pipeline starts once
        # we have this exact integer back, at the rescale_bitdepth call below.
        input_raw_int = round_half_up(input_bayer.astype(np.float64) * (2 ** meta.raw_bitWidth)).astype(np.int64)

        module_int = rescale_bitdepth(input_raw_int, meta.raw_bitWidth, args.bit_module)

        pred_module_int, grid_info = run_fixed_point(
            sess, input_name, tile_h, tile_w, args.margin, np_dtype,
            module_int, meta.ISO, ksigma, args.bit_module, args.net_bit_int, args.net_bit_frac,
        )

        # convert back to float32 [0,1] here -- everything from this point on
        # (visualization, saving, --compare-full) reuses the existing,
        # unchanged float32-based pipeline
        pred_bayer = from_fixed(pred_module_int, args.bit_module).astype(np.float32)

        msg = f'{meta.name}  [{grid_info}]'
        if full_denoiser is not None:
            # Denoiser.run() itself never clips its output (only save_bayer_raw/
            # bayer2rgb do, right before it's actually used for anything) -- so
            # near saturated highlights the network's residual can legitimately
            # overshoot past 1.0 there. This pipeline's own pred_bayer is always
            # clipped to [0,1] (RAW12 has no headroom above its max value), so
            # clip the reference the same way before diffing -- otherwise a
            # handful of saturated-highlight pixels dominate the max diff with a
            # difference that has nothing to do with fixed-point precision.
            full_pred_bayer = full_denoiser.run(input_bayer, iso=meta.ISO).clip(0, 1)
            diff = np.abs(pred_bayer.astype(np.float64) - full_pred_bayer.astype(np.float64))
            msg += f'  vs full (untiled) float32 PyTorch: max abs diff {diff.max():.3e}, mean abs diff {diff.mean():.3e} (0~1 scale)'
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
