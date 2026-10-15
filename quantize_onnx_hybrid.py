#!/usr/bin/env python3
"""Hybrid static PTQ: int8 for the deep middle of the network, fp16 for the input/output
boundary layers (and whatever a skip connection carries straight back to them) -- fully
static, no runtime scale computation (unlike quantize_onnx.py --dynamic) and no per-tile
blockiness (see CLAUDE.md's dynamic-quantization side-effect note), at the cost of needing
a deployment target that can mix int8 and fp16 in one graph, plus manual, architecture-
specific selection of which layers to protect.

This scheme was reverse-engineered from a real deployed model (models/from_JJ_torch_
pretrained_int8_fp16head_256x256.onnx), inspected node-by-node: it keeps `conv0`/`enc1`
(input-adjacent) and `dec3.proj_conv`/`dec4`/`out0`/`out1` (output-adjacent, plus whatever
consumes enc1's skip connection) in fp16, quantizing everything else (enc2/enc3/enc4/encdec/
dec1/dec2/dec3's main decode path -- the deep, low-resolution middle of the network) to
int8. Confirmed subjectively lossless by the user on dataset/test's high-ISO samples,
unlike plain full-network static PTQ (quantize_onnx.py's default mode).

Why this works: KSigma's cvt_k(iso) compresses activation magnitude by up to ~31x across
the ISO range this network sees (see quantize_onnx.py --dynamic's docs / CLAUDE.md) --
this hits hardest exactly at the layers that directly touch the raw, magnitude-varying
input (conv0) or must produce the small residual added back to it (out0/out1), and
anywhere a skip connection carries that same raw-magnitude-adjacent signal deep into the
decoder (dec3.proj_conv consumes enc1's output; dec4 consumes conv0's). Layers several
conv+ReLU stages removed from this boundary operate on activations shaped by the
network's own learned weights, not the input's raw scale, and tolerate one static
per-tensor/per-channel int8 scale much better.

Implementation: (1) onnxruntime.quantization.quantize_static as usual, but with every
node belonging to a --fp16-module passed via nodes_to_exclude, so those layers stay
float32 while everything else becomes int8 QDQ; (2) onnxconverter_common.float16.
convert_float_to_float16 with node_block_list covering every OTHER node, so only the
designated layers actually become float16 (keep_io_types=True, so the graph's own
external input/output stay float32 -- Cast nodes get inserted automatically at the
fp16<->int8 boundaries, matching the pattern seen in the JJ reference model).

Choosing --fp16-modules for a DIFFERENT network architecture: there's no universal
formula, but the same reasoning transfers -- (a) start from the layers structurally
closest to the input and output (the first and last few layers), plus any layer that
directly consumes a skip/residual connection originating near the input, since these are
the layers whose activations most directly reflect whatever external condition drives
your input's dynamic range (here, ISO; in general, whatever normalization scheme -- or
lack of one -- the input goes through). (b) Confirm quantitatively, not just by
architectural intuition: measure each layer's activation min/max across a calibration set
that spans the full range of the deployment condition (e.g. the whole ISO range here) --
layers whose range varies most across that set are the ones a single static scale will
represent worst, and are the real candidates. (c) Validate empirically per this repo's
established pattern -- quantize with a candidate --fp16-modules set, run tile_infer.py
across the full deployment range, and compare to the fp32 baseline; if the gap is still
too large, expand the fp16 set (prefer growing it along existing skip-connection paths,
not arbitrary unconnected layers, per the reasoning above) and re-test. There's a real
size/speed cost to keeping more of the network in fp16, so this is a tradeoff to tune,
not a one-shot answer.
"""
import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxconverter_common import float16 as onnx_float16
from onnxruntime.quantization import QuantFormat, quantize_static

from dataset.benchmark import BenchmarkLoader
from quantize_onnx import CALIBRATE_METHOD, QUANT_TYPE, RGGBCalibrationDataReader, collect_calibration_tiles

DEFAULT_FP16_MODULES = ['conv0', 'enc1', 'dec3.proj_conv', 'dec4', 'out0', 'out1']


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid int8 (deep middle) + fp16 (input/output boundary) static PTQ of a fp32 PMRID ONNX export")
    parser.add_argument('model', type=Path, help='fp32 ONNX model to quantize, exported by export_onnx.py WITHOUT --bake-ksigma')
    parser.add_argument('--benchmark', type=Path, required=True, help='benchmark.json/meta_info.json-style dataset to source calibration tiles from')
    parser.add_argument('--output', type=Path, default=None, help='output .onnx path; defaults to <model>_hybrid.onnx')
    parser.add_argument(
        '--fp16-modules', nargs='+', default=DEFAULT_FP16_MODULES,
        help="dotted PyTorch module-name prefixes (matching models/net_torch.py::Network's "
             "attribute names, e.g. 'dec3.proj_conv') to keep in fp16 instead of quantizing to "
             "int8 -- every ONNX node whose name falls under one of these prefixes is protected. "
             f"Default ({' '.join(DEFAULT_FP16_MODULES)}) matches the reference JJ model: the "
             "input-adjacent conv0/enc1, and the output-adjacent dec3.proj_conv/dec4/out0/out1 "
             "(the layers that consume conv0's/enc1's skip connections). See this script's "
             "module docstring for how to pick these for a different network.",
    )
    parser.add_argument('--num-tiles', type=int, default=200, help='target number of calibration tiles to collect')
    parser.add_argument('--per-channel', action=argparse.BooleanOptionalAction, default=True, help='quantize conv weights with one scale per output channel (see quantize_onnx.py)')
    parser.add_argument('--calibrate-method', type=str, default='minmax', choices=list(CALIBRATE_METHOD),
                         help="onnxruntime's own default is 'minmax'; measured empirically on this "
                              "network's hybrid scheme specifically (8 high-ISO dataset/test samples, "
                              "100-200 tiles): 'percentile' roughly halves mean abs diff vs. fp32 "
                              "again on top of the hybrid split itself (e.g. 0.0255 -> 0.0141 on the "
                              "worst sample) -- worth using over the 'minmax' default here.")
    parser.add_argument('--activation-type', type=str, default='int8', choices=list(QUANT_TYPE))
    parser.add_argument('--weight-type', type=str, default='int8', choices=list(QUANT_TYPE))
    parser.add_argument('--quant-format', type=str, default='qdq', choices=['qdq', 'qoperator'])
    return parser.parse_args()


def module_prefix_to_node_prefix(module: str) -> str:
    return '/' + module.replace('.', '/')


def node_in_module(node_name: str, module_prefix: str) -> bool:
    return node_name == module_prefix or node_name.startswith(module_prefix + '/')


def find_fp16_node_names(graph: onnx.GraphProto, fp16_modules: list) -> set:
    """Every node under one of `fp16_modules`'s prefixes, PLUS the node that directly
    produces the graph's own declared output. That last node is not optional: this
    architecture's `pred = inp + x` residual-add (models/net_torch.py::Network.forward)
    is a bare tensor op, not wrapped in any nn.Module, so no --fp16-modules prefix can
    ever match it by name -- yet it's exactly as boundary-adjacent as out0/out1, and
    leaving it out means the graph's final output silently ends up float16 instead of
    the float32 keep_io_types is supposed to guarantee (caught by running this for
    real: onnxruntime reported the output as tensor(float16), traced back to exactly
    this node having been left in the int8 bucket by accident). This generalizes past
    PMRID: any residual/skip combination performed directly in forward(), not inside a
    named submodule, needs the same treatment for any other architecture.
    """
    prefixes = [module_prefix_to_node_prefix(m) for m in fp16_modules]
    matched = {p: [] for p in prefixes}
    fp16_nodes = set()
    for n in graph.node:
        for p in prefixes:
            if node_in_module(n.name, p):
                matched[p].append(n.name)
                fp16_nodes.add(n.name)
                break

    for module, prefix in zip(fp16_modules, prefixes):
        if not matched[prefix]:
            raise ValueError(
                f"--fp16-modules {module!r} (looked for node names under {prefix!r}) matched no "
                f"nodes in this graph -- typo, or this module name doesn't exist in this network? "
                f"Check the actual node names (e.g. via Netron or onnx.load) and adjust."
            )
        print(f"  fp16 module {module!r}: {len(matched[prefix])} node(s)")

    out_name = graph.output[0].name
    out_producer = next((n for n in graph.node if out_name in n.output), None)
    if out_producer is None:
        raise ValueError(f"couldn't find the node producing this graph's own output tensor {out_name!r}")
    if out_producer.name not in fp16_nodes:
        print(f"  + final output-producing node {out_producer.name!r} ({out_producer.op_type}) -- "
              f"not covered by any --fp16-modules prefix, protected anyway (see this function's docstring)")
        fp16_nodes.add(out_producer.name)

    return fp16_nodes


def main():
    opt = parse_args()

    sess = ort.InferenceSession(str(opt.model), providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f'{opt.model} declares {len(inputs)} inputs -- looks like it was exported with --bake-ksigma; quantize the plain (1-input) graph instead')
    if inputs[0].type != 'tensor(float)':
        raise ValueError(f"{opt.model}'s input is {inputs[0].type}, not tensor(float) -- re-run export_onnx.py with --dtype fp32 first")
    input_name = inputs[0].name
    tile_h, tile_w = inputs[0].shape[2:]

    model_proto = onnx.load(str(opt.model))
    print(f"resolving --fp16-modules against {opt.model}:")
    fp16_node_names = find_fp16_node_names(model_proto.graph, opt.fp16_modules)
    print(f"total: {len(fp16_node_names)} node(s) will stay fp16; everything else is quantized to int8")

    bm_loader = BenchmarkLoader(opt.benchmark.resolve())
    tiles = collect_calibration_tiles(bm_loader, tile_h, tile_w, opt.num_tiles)
    reader = RGGBCalibrationDataReader(tiles, input_name)

    output_path = opt.output or opt.model.with_name(f'{opt.model.stem}_hybrid.onnx')
    quantized_path = output_path.with_name(f'{output_path.stem}_int8only.onnx')

    quantize_static(
        model_input=str(opt.model),
        model_output=str(quantized_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ if opt.quant_format == 'qdq' else QuantFormat.QOperator,
        per_channel=opt.per_channel,
        activation_type=QUANT_TYPE[opt.activation_type],
        weight_type=QUANT_TYPE[opt.weight_type],
        calibrate_method=CALIBRATE_METHOD[opt.calibrate_method],
        nodes_to_exclude=list(fp16_node_names),
    )

    quantized_proto = onnx.load(str(quantized_path))
    quantized_node_names = {n.name for n in quantized_proto.graph.node}
    missing = fp16_node_names - quantized_node_names
    if missing:
        raise RuntimeError(
            f"quantize_static renamed or dropped {len(missing)} node(s) we asked to exclude "
            f"(e.g. {sorted(missing)[:5]}) -- the node_block_list for the fp16 conversion step "
            f"below would be wrong. This means quantize_static's node-naming assumption this "
            f"script relies on no longer holds; needs investigating before trusting the output."
        )
    fp16_block_list = list(quantized_node_names - fp16_node_names)

    model_fp16 = onnx_float16.convert_float_to_float16(
        quantized_proto, keep_io_types=True, node_block_list=fp16_block_list,
    )
    onnx.save(model_fp16, str(output_path))
    quantized_path.unlink()

    fp32_size = opt.model.stat().st_size
    out_size = output_path.stat().st_size
    print(f'Exported to {output_path} ({out_size/1e6:.2f} MB, vs {fp32_size/1e6:.2f} MB fp32 -- {fp32_size/out_size:.2f}x smaller)')
    print(
        'This is only calibrated, not validated -- re-run tile_infer.py (or the full PSNR/SSIM '
        'benchmark) against the fp32 baseline, across the full range of deployment conditions '
        '(e.g. the whole ISO range), before trusting it. If it still degrades at the extremes, '
        'the --fp16-modules set may need to grow (see this script\'s module docstring).'
    )


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
