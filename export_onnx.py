#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from models.net_torch import Network


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained PMRID PyTorch checkpoint to a fixed-shape ONNX graph")
    parser.add_argument('model', type=Path, help='path to the pytorch checkpoint (state_dict), e.g. models/torch_pretrained.ckp')
    parser.add_argument('--output', type=Path, default=None, help='output .onnx path; defaults to <model>_<dtype>_<height>x<width>.onnx next to the checkpoint')
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
    return parser.parse_args()


def load_net(model_path: Path) -> Network:
    net = Network()
    state_dict = torch.load(str(model_path), map_location='cpu')
    net.load_state_dict(state_dict)
    return net.eval()


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
    dummy = torch.randn(1, 4, opt.height, opt.width, dtype=torch.float32)

    output_path = opt.output or opt.model.with_name(f'{opt.model.stem}_{opt.dtype}_{opt.height}x{opt.width}.onnx')

    # Always trace/export in float32 (see --dtype help above for why); for --dtype fp16 we
    # export to a temp fp32 file first, then convert that graph to fp16 and remove the temp file.
    export_path = output_path if opt.dtype == 'fp32' else output_path.with_suffix('.fp32tmp.onnx')
    torch.onnx.export(
        net, dummy, str(export_path),
        input_names=['input'], output_names=['output'],
        opset_version=opt.opset,
        dynamo=False,  # this graph is static (no control flow); skip the dynamo exporter to avoid the extra onnxscript dependency
    )

    if opt.dtype == 'fp16':
        import onnx
        from onnxconverter_common import float16 as onnx_float16
        model_fp16 = onnx_float16.convert_float_to_float16(onnx.load(str(export_path)), keep_io_types=False)
        onnx.save(model_fp16, str(output_path))
        export_path.unlink()

    print(f'Exported to {output_path} (dtype={opt.dtype}, shape=1x4x{opt.height}x{opt.width})')

    # verify the exported graph reproduces the PyTorch (float32) output, within the dtype's precision
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    input_np = dummy.numpy().astype(np.float16 if opt.dtype == 'fp16' else np.float32)
    onnx_out = sess.run(None, {'input': input_np})[0]
    with torch.no_grad():
        torch_out = net(dummy).numpy()
    diff = np.abs(onnx_out.astype(np.float64) - torch_out.astype(np.float64)).max()
    print(f'ONNX ({opt.dtype}) vs PyTorch float32 max abs diff: {diff:.3e}')


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
