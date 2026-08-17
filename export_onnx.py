#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.net_torch import Network

# same defaults as run_benchmark_pytorch.py/tile_infer.py's KSigma(...) instantiation
# (Reno 10x noise calibration); overridable via --k-coeff/--b-coeff/--anchor/--v for a
# different sensor's calibration.
DEFAULT_K_COEFF = [0.0005995267, 0.00868861]
DEFAULT_B_COEFF = [7.11772e-7, 6.514934e-4, 0.11492713]
DEFAULT_ANCHOR = 1600.0
DEFAULT_V = 959.0


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained PMRID PyTorch checkpoint to a fixed-shape ONNX graph")
    parser.add_argument('model', type=Path, help='path to the pytorch checkpoint (state_dict), e.g. models/torch_pretrained.ckp')
    parser.add_argument('--output', type=Path, default=None, help='output .onnx path; defaults to <model>_<dtype>_<height>x<width>[_ksigma].onnx next to the checkpoint')
    parser.add_argument('--height', type=int, default=256, help='network input height (in RGGB-plane space) baked into the exported graph; must be a multiple of 32 (the network downsamples 4x by 2x each stage)')
    parser.add_argument('--width', type=int, default=256, help='network input width (in RGGB-plane space) baked into the exported graph; must be a multiple of 32')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument(
        '--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'],
        help="data type for the exported graph's weights and input/output tensors. fp32 is the "
             "normal export. fp16 exports fp32 first, then casts the *exported graph* (weights + "
             "tensor dtypes) to float16 as a separate graph-rewrite step -- CPU PyTorch has no fast "
             "fp16 Conv2d kernel, so the model is never actually run in fp16 during export/tracing. "
             "int8 is a reserved placeholder, see below.",
    )
    parser.add_argument(
        '--bake-ksigma', action='store_true',
        help="also export the ISO-dependent KSigma normalize/denormalize affine transform "
             "(run_benchmark_pytorch.py's KSigma) into the graph, instead of leaving it to be "
             "applied externally in numpy. When set, the exported graph takes a *second* input "
             "'iso' (a 1-element tensor) alongside 'input', and internally does "
             "normalize(iso) -> network -> denormalize(iso). ISO stays a genuine runtime input "
             "either way (never baked in as a fixed constant) -- this flag only decides whether "
             "the affine math itself lives inside the graph or in the calling script.",
    )
    parser.add_argument('--k-coeff', type=float, nargs=2, default=DEFAULT_K_COEFF, metavar=('A', 'B'), help='KSigma K(iso) = A*iso + B (only used with --bake-ksigma)')
    parser.add_argument('--b-coeff', type=float, nargs=3, default=DEFAULT_B_COEFF, metavar=('A', 'B', 'C'), help='KSigma Sigma(iso) = A*iso^2 + B*iso + C (only used with --bake-ksigma)')
    parser.add_argument('--anchor', type=float, default=DEFAULT_ANCHOR, help='KSigma anchor ISO the network was trained at (only used with --bake-ksigma)')
    parser.add_argument('--v', type=float, default=DEFAULT_V, help='KSigma V normalization constant (only used with --bake-ksigma)')
    parser.add_argument('--dummy-iso', type=float, default=3200.0, help='ISO value used for the trace/verification dummy input (only used with --bake-ksigma); any value works since ISO is a real graph input, not baked in -- this just needs to be a realistic, non-anchor value so verification actually exercises the affine transform (at iso==anchor it collapses to the identity)')
    return parser.parse_args()


def load_net(model_path: Path) -> Network:
    net = Network()
    state_dict = torch.load(str(model_path), map_location='cpu')
    net.load_state_dict(state_dict)
    return net.eval()


class KSigmaModule(nn.Module):
    """Traceable torch port of run_benchmark_pytorch.py's KSigma. K(iso)/Sigma(iso) are
    evaluated with plain tensor ops (Horner form of the same low-degree polynomials
    np.poly1d would compute) so torch.onnx.export can trace them like any other op --
    there's no python-level branching on iso's *value*, only on `inverse` (fixed per
    call site at trace time, not a runtime input)."""

    def __init__(self, k_coeff, b_coeff, anchor: float, v: float):
        super().__init__()
        self.k_coeff = k_coeff
        self.b_coeff = b_coeff
        self.anchor = anchor
        self.v = v

    def _K(self, iso):
        a, b = self.k_coeff
        return a * iso + b

    def _Sigma(self, iso):
        a, b, c = self.b_coeff
        return a * iso ** 2 + b * iso + c

    def forward(self, img_01, iso, inverse: bool):
        k, sigma = self._K(iso), self._Sigma(iso)
        k_a, sigma_a = self._K(self.anchor), self._Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.v
        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k
        return img / self.v


class NetworkWithKSigma(nn.Module):
    """Wraps Network with KSigmaModule so the exported graph does the full
    normalize -> denoise -> denormalize pipeline, given the tile and its ISO."""

    def __init__(self, net: Network, ksigma: KSigmaModule, inp_scale=256.0):
        super().__init__()
        self.net = net
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def forward(self, rggb_01, iso):
        x = self.ksigma(rggb_01, iso, inverse=False) * self.inp_scale
        pred = self.net(x) / self.inp_scale
        return self.ksigma(pred, iso, inverse=True)


def main():
    opt = parse_args()

    if opt.dtype == 'int8':
        print(
            "--dtype int8 is not implemented yet: int8 needs calibration-based post-training "
            "quantization (representative raw RGGB tiles + a scale/zero-point calibration pass, "
            "e.g. via onnxruntime.quantization.quantize_static), not a plain dtype cast like fp16. "
            "Export fp32 or fp16 here first, then run that separate quantization step on it."
        )
        return

    for dim_name, dim in (('height', opt.height), ('width', opt.width)):
        if dim % 32 != 0:
            raise ValueError(f'--{dim_name} must be a multiple of 32 (network downsamples 4x by 2x each stage), got {dim}')

    net = load_net(opt.model)
    # PMRID's network operates on 4-channel RGGB planes, not raw bayer or 3-channel RGB
    dummy_rggb = torch.randn(1, 4, opt.height, opt.width, dtype=torch.float32)

    if opt.bake_ksigma:
        ksigma = KSigmaModule(opt.k_coeff, opt.b_coeff, opt.anchor, opt.v)
        model = NetworkWithKSigma(net, ksigma).eval()
        dummy_iso = torch.tensor([opt.dummy_iso], dtype=torch.float32)
        dummy_args = (dummy_rggb, dummy_iso)
        input_names = ['input', 'iso']
    else:
        model = net
        dummy_args = (dummy_rggb,)
        input_names = ['input']

    suffix = '_ksigma' if opt.bake_ksigma else ''
    output_path = opt.output or opt.model.with_name(f'{opt.model.stem}_{opt.dtype}_{opt.height}x{opt.width}{suffix}.onnx')

    # Always trace/export in float32 (see --dtype help above for why); for --dtype fp16 we
    # export to a temp fp32 file first, then convert that graph to fp16 and remove the temp file.
    export_path = output_path if opt.dtype == 'fp32' else output_path.with_suffix('.fp32tmp.onnx')
    torch.onnx.export(
        model, dummy_args, str(export_path),
        input_names=input_names, output_names=['output'],
        opset_version=opt.opset,
        dynamo=False,  # this graph is static (no control flow); skip the dynamo exporter to avoid the extra onnxscript dependency
    )

    if opt.dtype == 'fp16':
        import onnx
        from onnxconverter_common import float16 as onnx_float16
        model_fp16 = onnx_float16.convert_float_to_float16(onnx.load(str(export_path)), keep_io_types=False)
        onnx.save(model_fp16, str(output_path))
        export_path.unlink()

    ksigma_note = f', ksigma baked in (iso is a graph input)' if opt.bake_ksigma else ''
    print(f'Exported to {output_path} (dtype={opt.dtype}, shape=1x4x{opt.height}x{opt.width}{ksigma_note})')

    # verify the exported graph reproduces the PyTorch (float32) output, within the dtype's precision
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    onnx_dtype = np.float16 if opt.dtype == 'fp16' else np.float32
    feed = {'input': dummy_rggb.numpy().astype(onnx_dtype)}
    if opt.bake_ksigma:
        feed['iso'] = dummy_iso.numpy().astype(onnx_dtype)
    onnx_out = sess.run(None, feed)[0]
    with torch.no_grad():
        torch_out = model(*dummy_args).numpy()
    diff = np.abs(onnx_out.astype(np.float64) - torch_out.astype(np.float64)).max()
    print(f'ONNX ({opt.dtype}) vs PyTorch float32 max abs diff: {diff:.3e}')


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
