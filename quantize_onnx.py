#!/usr/bin/env python3
"""Post-training static (int8) quantization for a fp32 PMRID ONNX export, targeting
onnxruntime's own CPU int8 execution. See export_onnx.py --dtype int8 for why this
needs a separate calibration-based step instead of a plain dtype cast (unlike --dtype
fp16), and AI_CMODEL_INTEGRATION.md for why the resulting int8 model is NOT guaranteed
to behave identically once deployed on other int8 runtimes/accelerators.
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static, quantize_dynamic

from utils import RawUtils
from dataset.benchmark import BenchmarkLoader
from run_benchmark_pytorch import KSigma
from tile_infer import tile_positions

QUANT_TYPE = {'int8': QuantType.QInt8, 'uint8': QuantType.QUInt8}
CALIBRATE_METHOD = {
    'minmax': CalibrationMethod.MinMax,
    'entropy': CalibrationMethod.Entropy,
    'percentile': CalibrationMethod.Percentile,
    'distribution': CalibrationMethod.Distribution,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Post-training static int8 quantization of a fp32 PMRID ONNX export")
    parser.add_argument('model', type=Path, help='fp32 ONNX model to quantize, exported by export_onnx.py WITHOUT --bake-ksigma (see below for why)')
    parser.add_argument('--benchmark', type=Path, default=None, help='benchmark.json/meta_info.json-style dataset to source calibration tiles from (real images, not synthetic data -- calibration quality depends on this being representative of real deployment inputs). Required unless --dynamic is set (dynamic quantization needs no calibration data at all)')
    parser.add_argument('--output', type=Path, default=None, help='output .onnx path; defaults to <model>_int8.onnx')
    parser.add_argument(
        '--dynamic', action='store_true',
        help="use onnxruntime's dynamic quantization (quantize_dynamic) instead of the "
             "default static quantization (quantize_static): activation scale/zero-point is "
             "computed fresh from the actual tensor on every inference call, instead of one "
             "constant baked in from --benchmark calibration data (which --dynamic doesn't "
             "need at all -- --num-tiles/--calibrate-method/--activation-type/--quant-format "
             "are ignored in this mode, they're static-only). Confirmed empirically to fix a "
             "real static-quantization failure mode on this network: KSigma normalizes "
             "different ISOs toward the same anchor noise level by compressing overall "
             "activation magnitude by up to ~31x between low and high ISO (see CLAUDE.md) -- "
             "one static scale, calibrated across that whole range, is dominated by the large "
             "low-ISO values and starves the much-smaller high-ISO activations of "
             "quantization levels. Dynamic quantization sidesteps this since there's no "
             "shared scale across calls to begin with -- verified: mean abs diff vs. fp32 on "
             "high-ISO dataset/test samples dropped from 0.012-0.043 (static) to "
             "0.0022-0.0034 (dynamic), no longer growing with ISO the way static's did. "
             "Trade-offs: this network's ConvTranspose2d layers (one per DecoderStage, 4 "
             "total) have no dynamic-quant kernel in onnxruntime and stay fp32 -- confirmed "
             "empirically, not a pure int8 graph -- and computing the scale on every call has "
             "real runtime cost vs. static quantization's baked-in constant; profile before "
             "assuming this is free on the actual deployment target, and note it only helps "
             "on runtimes that support computing quantization parameters at inference time. "
             "A fixed-function int8 accelerator needing pre-baked static scales cannot use "
             "this fix -- an external per-ISO rescale was tried as an alternative and found "
             "not to work (see CLAUDE.md): it necessarily undoes KSigma's own noise-level "
             "normalization along with the magnitude it was meant to only correct, since "
             "they're the same multiplication, not separable.")
    parser.add_argument('--num-tiles', type=int, default=200, help='target number of calibration tiles to collect (stops early once reached; prints a warning instead of silently returning fewer if the dataset runs out first)')
    parser.add_argument('--per-channel', action=argparse.BooleanOptionalAction, default=True, help='quantize conv weights with one scale per output channel instead of one scale for the whole tensor. Recommended on: this network is mostly depthwise-separable convs (models/net_torch.py), whose per-channel weight magnitudes can vary a lot, so per-channel quantization typically preserves accuracy much better here than per-tensor')
    parser.add_argument('--calibrate-method', type=str, default='minmax', choices=list(CALIBRATE_METHOD), help="activation calibration method (onnxruntime.quantization.CalibrationMethod). 'minmax' is onnxruntime's own default and simplest/fastest; 'entropy'/'percentile' are more robust to outlier activations at the cost of a slower calibration pass -- worth trying if minmax's PSNR/SSIM hit is too large")
    parser.add_argument('--activation-type', type=str, default='int8', choices=list(QUANT_TYPE), help='activation quantization dtype (onnxruntime default: int8/symmetric)')
    parser.add_argument('--weight-type', type=str, default='int8', choices=list(QUANT_TYPE), help='weight quantization dtype (onnxruntime default: int8/symmetric)')
    parser.add_argument('--reduce-range', action='store_true', help="quantize weights to 7-bit instead of 8-bit. onnxruntime's own docs frame this as a CPU-compatibility knob (avoids dot-product overflow on non-VNNI x86 CPUs, particularly with --per-channel), not a pure accuracy lever -- effect on this network's actual output quality is untested here, try empirically")
    parser.add_argument(
        '--quant-format', type=str, default='qdq', choices=['qdq', 'qoperator'],
        help="'qdq' (default) inserts QuantizeLinear/DequantizeLinear node pairs around quantized ops "
             "and keeps the graph's own external input/output tensors as float32 -- the most portable "
             "format, and what most vendor toolchains expect to import (see AI_CMODEL_INTEGRATION.md). "
             "'qoperator' fuses straight to int8 kernels (e.g. QLinearConv), which can be faster on CPU "
             "but is a less commonly supported import format elsewhere.",
    )
    return parser.parse_args()


class RGGBCalibrationDataReader(CalibrationDataReader):
    """Feeds pre-extracted (1,4,tile_h,tile_w) float32 RGGB tiles to onnxruntime's
    calibration pass, one at a time, already in the exact domain the network expects
    at inference (KSigma-normalized and *inp_scale -- see collect_calibration_tiles)."""

    def __init__(self, tiles, input_name):
        self.input_name = input_name
        self._iter = iter(tiles)

    def get_next(self):
        tile = next(self._iter, None)
        if tile is None:
            return None
        return {self.input_name: tile[np.newaxis]}


def collect_calibration_tiles(bm_loader: BenchmarkLoader, tile_h: int, tile_w: int, num_tiles: int):
    """Extract real (4, tile_h, tile_w) RGGB tiles from a benchmark dataset, in the same
    domain the plain (non-baked) ONNX graph's 'input' expects at inference: canonical
    RGGB layout, KSigma-normalized for that sample's real ISO, scaled by inp_scale.
    Only the noisy `input` image is used (never `gt`) since that's the only thing the
    network ever actually sees in real deployment.

    Non-overlapping tiles (no --margin) are enough for calibration -- we only need a
    representative sample of the activation value distribution, not a correctly
    stitched output.
    """
    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    inp_scale = 256.0

    tiles = []
    for input_bayer, _gt_bayer, meta in bm_loader:
        if len(tiles) >= num_tiles:
            break

        input_bayer = RawUtils.to_canonical_rggb(input_bayer, pattern=meta.bayer_pattern)
        rggb = RawUtils.bayer2rggb(input_bayer).clip(0, 1)
        rggb = ksigma(rggb, meta.ISO) * inp_scale
        rggb = rggb.astype(np.float32)

        H, W = rggb.shape[:2]
        if H < tile_h or W < tile_w:
            print(f'skipping {meta.name}: image ({H}x{W} RGGB-plane) is smaller than one tile ({tile_h}x{tile_w})')
            continue

        ys = tile_positions(H, tile_h, tile_h)  # stride == tile size: non-overlapping grid
        xs = tile_positions(W, tile_w, tile_w)
        for y0 in ys:
            for x0 in xs:
                if len(tiles) >= num_tiles:
                    break
                tile = rggb[y0:y0 + tile_h, x0:x0 + tile_w].transpose(2, 0, 1)
                tiles.append(tile)

    if len(tiles) < num_tiles:
        print(
            f'WARNING: only collected {len(tiles)} calibration tiles (requested {num_tiles}) -- '
            f'the --benchmark dataset ran out of images. Calibration will proceed with what was '
            f'collected, but consider pointing --benchmark at a larger dataset or lowering --num-tiles.'
        )
    else:
        print(f'collected {len(tiles)} calibration tiles')
    return tiles


def main():
    opt = parse_args()

    if not opt.dynamic and opt.benchmark is None:
        raise SystemExit('--benchmark is required unless --dynamic is set')

    sess = ort.InferenceSession(str(opt.model), providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError(
            f'{opt.model} declares {len(inputs)} inputs -- this looks like it was exported with '
            f'export_onnx.py --bake-ksigma. Quantize the plain (1-input) graph instead: KSigma has '
            f'no weights to quantize (it is pure elementwise math), and its `iso` input ranges over '
            f'a very wide dynamic range (real ISOs span roughly 100-25600) that a single int8 scale '
            f'would represent extremely coarsely. Re-run export_onnx.py without --bake-ksigma.'
        )
    if inputs[0].type != 'tensor(float)':
        raise ValueError(
            f"{opt.model}'s input is {inputs[0].type}, not tensor(float) -- quantize_static/"
            f"quantize_dynamic expect a float32 source graph. Re-run export_onnx.py with "
            f"--dtype fp32 (not fp16) first."
        )

    output_path = opt.output or opt.model.with_name(f'{opt.model.stem}_int8.onnx')

    if opt.dynamic:
        quantize_dynamic(
            model_input=str(opt.model),
            model_output=str(output_path),
            per_channel=opt.per_channel,
            weight_type=QUANT_TYPE[opt.weight_type],
            reduce_range=opt.reduce_range,
        )
    else:
        input_name = inputs[0].name
        tile_h, tile_w = inputs[0].shape[2:]

        bm_loader = BenchmarkLoader(opt.benchmark.resolve())
        tiles = collect_calibration_tiles(bm_loader, tile_h, tile_w, opt.num_tiles)
        reader = RGGBCalibrationDataReader(tiles, input_name)

        quantize_static(
            model_input=str(opt.model),
            model_output=str(output_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ if opt.quant_format == 'qdq' else QuantFormat.QOperator,
            per_channel=opt.per_channel,
            activation_type=QUANT_TYPE[opt.activation_type],
            weight_type=QUANT_TYPE[opt.weight_type],
            calibrate_method=CALIBRATE_METHOD[opt.calibrate_method],
            reduce_range=opt.reduce_range,
        )

    fp32_size = opt.model.stat().st_size
    int8_size = output_path.stat().st_size
    print(f'Exported to {output_path} ({int8_size/1e6:.2f} MB, vs {fp32_size/1e6:.2f} MB fp32 -- {fp32_size/int8_size:.2f}x smaller)')
    if opt.dynamic:
        print(
            'Dynamic quantization: this network\'s 4 ConvTranspose2d layers (one per '
            'DecoderStage) have no dynamic-quant kernel in onnxruntime and were left fp32 -- '
            'not a pure int8 graph. Still not automatically trusted -- re-run tile_infer.py '
            '(or the full PSNR/SSIM benchmark) against the fp32 baseline before using this '
            'for anything real, and profile inference speed on the actual target runtime '
            '(computing the scale on every call has real cost vs. static quantization).'
        )
    else:
        print(
            'This is only calibrated, not validated -- re-run the PSNR/SSIM benchmark against this '
            'int8 model (e.g. via tile_infer.py) and compare to the fp32 baseline before trusting it. '
            'A meaningful PSNR/SSIM drop means try --calibrate-method entropy/percentile, more '
            '--num-tiles, or --per-channel if it was off -- or, if the drop is specifically worse at '
            'high ISO, try --dynamic instead (see its --help for why).'
        )


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
