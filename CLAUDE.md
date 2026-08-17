# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Code and pre-trained models for the ECCV20 paper "Practical Deep Raw Image Denoising on Mobile Devices" (PMRID). This is a small research/benchmark repo, not a training codebase — there is no training loop here, only inference (`Denoiser`) and evaluation against a RAW image dataset.

There are two parallel benchmark scripts, kept in sync by hand:

- `run_benchmark_meg.py` — the original script, runs the **MegEngine** model (`models/net_mge.py` + `models/mge_pretrained.ckp`), loading weights via `pickle` rather than MegEngine's own checkpoint API. Only supports `BGGR` bayer input and doesn't save any images.
- `run_benchmark_pytorch.py` — runs the **PyTorch** model (`models/net_torch.py` + `models/torch_pretrained.ckp`) instead, loading weights via `torch.load`/`load_state_dict`. Adds two things the MegEngine script doesn't have: support for all four bayer patterns, and a `--save-dir` option to dump visualizations (see below). On this project's Windows dev machine, MegEngine has no installable wheel, so **prefer `run_benchmark_pytorch.py`** unless you specifically need to reproduce the MegEngine numbers.

**Run all python commands in this repo inside the `AIISP_init` conda env** (`conda run -n AIISP_init python ...`) — it has `torch`/`numpy`/`opencv-python`/`scikit-image` installed; MegEngine is not installed there (and has no Windows wheel).

```
conda run -n AIISP_init python run_benchmark_pytorch.py --benchmark /path/to/PMRID/benchmark.json models/torch_pretrained.ckp --save-dir /path/to/output
```

- Requires `Python >= 3.6`.
- There are no tests, lint config, or CI in this repo.
- The scikit-image version in `AIISP_init` (0.26) no longer accepts `structural_similarity(..., multichannel=True)` — use `channel_axis=-1` instead (already done in `run_benchmark_pytorch.py`).

## Dataset layout

`dataset/benchmark/` (gitignored data, not the code) mirrors the structure described in the README: `benchmark.json` holds metadata for every sample, and each entry points at `input.raw`/`gt.raw` — raw uint16 Bayer buffers — under `SceneN/{Bright,Dark}/exposure-caseN/`. `benchmark.py::BenchmarkLoader` parses `benchmark.json` into `RawMeta` dataclasses and iterates over `(input_bayer, gt_bayer, meta)` triples, resolving relative file paths against the JSON's parent directory (or an explicit `base_path`).

Per-sample metadata (`RawMeta`) carries everything needed to turn a raw Bayer image into a comparable RGB image: `bayer_pattern` (one of `RGGB`/`BGGR`/`GRBG`/`GBRG` — the original Reno 10x dataset is always `BGGR`, but the loader/utils accept any of the four), `ISO`/`exp_time`, `wb_gain`, `CCM` (3x3 color correction matrix), `ROIs` (patch regions used for PSNR/SSIM), and `raw_bitWidth` (bit depth of the raw `.raw` files, default `16` when a dataset's json omits it — used by `BenchmarkLoader._load_idx` to normalize to `[0,1]` via `2**raw_bitWidth - 1` instead of assuming 16-bit; the Reno 10x benchmark data is 16-bit, but `dataset/test/meta_info.json` is a user-supplied 12-bit dataset).

## Processing pipeline (run_benchmark_pytorch.py / run_benchmark_meg.py)

The eval pipeline for each sample:

1. **Bayer pattern → canonical RGGB** — `RawUtils.to_canonical_rggb(*, pattern)` flips each sample's raw array from its own `meta.bayer_pattern` into the canonical RGGB layout (top-left pixel = R) that the rest of the pipeline assumes; this is an involution (self-inverse), so calling it again with the same `pattern` later converts back. (The MegEngine script instead hardcodes this as `RawUtils.bggr2rggb`, which only handles `BGGR`.)
2. **KSigma noise normalization** — `KSigma` maps sensor ISO to a fixed "anchor" ISO's noise level (polynomial K/Sigma coefficients are hardcoded in the benchmark script), applied before inference and inverted after.
3. **Bayer → RGGB planes** — `RawUtils.bayer2rggb` packs the 2x2 canonical bayer array into 4-channel RGGB planes before feeding the network; `RawUtils.rggb2bayer` reverses this after inference.
4. **Padding** — `Denoiser.pre_process` pads the raw bayer array (before `bayer2rggb`) up to a multiple of 32 (in RGGB-plane space; the network downsamples 4x via 4 encoder stages) and crops symmetrically after. Uses `np.pad(..., mode='reflect')` (no edge-pixel duplication) rather than zero/`'constant'` padding, and the pad amount is always even (`2*ph`/`2*pw` raw pixels) — both required to keep the correct bayer color at each mosaic position across the seam (e.g. row `R0 G0 R1 G1` padded 2 cols each side must become `R1 G0 | R0 G0 R1 G1 | R1 G0`, not `G0 R0 | ... | G1 R1`, which is what edge-duplicating reflect — `mode='symmetric'`/`cv2.BORDER_REFLECT` — would produce). Zero-padding was the original approach here and measurably hurt denoising quality right at the image border (confirmed by diffing tiled vs. full-image inference: a ~10x error spike confined to the outermost ~4px, gone after switching to this reflect scheme). `tile_infer.py`'s `TiledOnnxDenoiser.run` applies the same scheme (padding by `margin` RGGB-plane pixels) before tiling, for the same reason.
5. **Inference** — `Denoiser.run` scales input by `inp_scale` (256.0), runs `Network`, rescales output back down. The result is still a canonical-RGGB-layout bayer array.
6. **RGB conversion** — `RawUtils.bayer2rgb` applies white balance, demosaics with OpenCV (`COLOR_BAYER_BG2RGB_EA`), applies the CCM, and gamma-corrects (2.2), then `to_canonical_rggb` flips the RGB image back to the original orientation. This runs on the **full image** for all three of input/pred/gt; PSNR/SSIM then crop to `meta.ROIs` afterward — cropping happens on separate patch variables, so the full-size RGB arrays are also what `--save-dir` writes out as PNGs.
7. PSNR/SSIM (via `skimage.metrics`) are computed per-ROI and averaged across the whole dataset.
8. **`--save-dir` (pytorch script only)** — per sample, writes `{name}_input.png`/`{name}_pred.png`/`{name}_gt.png` (full-image RGB, BGR uint8 for `cv2.imwrite`) and `{name}_pred.raw` (the denoised bayer result, flipped from canonical back to the sample's original `bayer_pattern` via `to_canonical_rggb` again, saved as raw uint16 — same format as the input `.raw` files).

`utils.py::RawUtils` is a set of stateless classmethods for all Bayer/RGGB/RGB conversions — reuse these rather than reimplementing raw-image reshaping.

## ONNX export & tiled inference (export_onnx.py, tile_infer.py)

- `export_onnx.py` exports `models/net_torch.py::Network` to a **fixed-shape** ONNX graph, input/output `(1, 4, height, width)`, `--height`/`--width` default `256`/`256` and must be multiples of 32. `--dtype fp32|fp16|int8`: `fp32` is a normal export; `fp16` exports fp32 first then casts the *graph* (weights + tensor dtypes) to float16 via `onnxconverter_common` — the model is never actually run in fp16 (this CPU PyTorch build has no fast fp16 conv kernel); `int8` is a reserved no-op that prints why it isn't implemented (needs calibration-based post-training quantization, not a plain cast) and returns without writing a file. After exporting, it round-trips a dummy input through both the ONNX graph (via onnxruntime) and the original PyTorch model and prints the max abs diff as a sanity check.
  - `--bake-ksigma` additionally exports the ISO-dependent KSigma normalize/denormalize affine transform *into* the graph (`KSigmaModule`/`NetworkWithKSigma`), rather than leaving it to be applied externally in numpy (the default, matching `Denoiser`/`TiledOnnxDenoiser`). This is a real design choice, not a hard constraint — KSigma is pure elementwise tensor math (`K(iso)`/`Sigma(iso)` are low-degree polynomials in `iso`, no branching on its runtime value), so it traces into ONNX ops fine; with the flag on, the graph takes a *second* input `iso` (a 1-element tensor, never baked in as a constant) alongside `input`. `--k-coeff`/`--b-coeff`/`--anchor`/`--v` override the (Reno 10x-calibrated) defaults for a different sensor. Keeping KSigma external (the default) makes swapping noise calibration a config change instead of a re-export, and keeps the graph a "pure" CNN for e.g. future int8 calibration; baking it in makes the graph self-contained for deployment targets that can't easily run custom pre/post-processing.
- `tile_infer.py` runs a fixed-shape ONNX export over real (arbitrary-size) raw images. It reuses `KSigma`/`Denoiser`/`save_rgb_png`/`save_bayer_raw` from `run_benchmark_pytorch.py` and `BenchmarkLoader`/`RawUtils` — same bayer-pattern/KSigma/RGB-conversion/raw-bitWidth handling as the other benchmark scripts, just with the network forward pass replaced:
  - `tile_positions`/`ownership_bounds` — 1D helpers computing overlapping tile start positions and non-overlapping "ownership" slices for stitching (same scheme as `../DnCNN-PyTorch_Saoyan/tile_infer.py`, generalized from 1-channel grayscale to PMRID's 4-channel RGGB planes).
  - `tiled_infer_rggb` — splits a full-size `(H, W, 4)` RGGB-plane array into `tile_h x tile_w` (the ONNX graph's baked-in shape) tiles with `--margin` pixels of overlap, runs each through the onnxruntime session, and stitches results back using `ownership_bounds` (each final pixel comes from whichever tile "owns" that region, cutting overlaps at the midpoint). Reflect-pads if the image is smaller than one tile. Note tile size is in **RGGB-plane space** — a 256x256 ONNX tile covers a 512x512 region of the original bayer image.
  - `TiledOnnxDenoiser` — same KSigma-normalize → `bayer2rggb` → (now tiled) inference → un-normalize → `rggb2bayer` flow as `Denoiser.run`, but backed by `tiled_infer_rggb` over onnxruntime instead of one single-shot padded-to-32 PyTorch forward pass. Its `bake_ksigma` flag must match whether `--onnx` was actually exported with `export_onnx.py --bake-ksigma`: when true, it skips the external KSigma/inp_scale step entirely and instead feeds the raw `[0,1]` RGGB tile plus a per-image `iso` value straight into the graph (as its second declared input) for every tile — `main()` cross-checks `--bake-ksigma` against the ONNX model's actual declared input count (1 vs 2) and raises a clear error on mismatch rather than silently double-normalizing or failing deep inside onnxruntime.
  - `--compare-full PATH` additionally runs the same sample through the untiled PyTorch `Denoiser` (loaded from a checkpoint at `PATH`) and prints the diff vs the tiled ONNX result, to help tune `--margin`. In testing, increasing `--margin` (32→96) didn't shrink the observed diff (~0.037 max, ~4e-4 mean on a `[0,1]` scale) — it appears dominated by ONNX-runtime-vs-PyTorch floating-point kernel differences rather than tile-seam artifacts, so don't assume a bigger margin alone will close this gap.

## Model architecture (models/net_mge.py, models/net_torch.py)

Both files define the **identical** encoder-decoder (U-Net-style) architecture — one in MegEngine (`megengine.module`), one in PyTorch (`torch.nn`) — kept in sync by hand. When changing the architecture, mirror the change in both files.

- `Conv2D(...)` — helper returning either a standard conv or a depthwise+pointwise separable conv (`is_seperable=True`), optionally with ReLU. Almost all convs in this network are separable (mobile-oriented design).
- `EncoderBlock` / `EncoderStage` — residual blocks with a projection shortcut when stride/channels change; 4 encoder stages (`enc1..enc4`) downsample by 2x each via stride-2 first block.
- `DecoderBlock` / `DecoderStage` — residual decode block, then `ConvTranspose2d`/`M.ConvTranspose2d` 2x upsample, fused with a projected skip connection from the matching encoder stage.
- `Network` — 4 encoder stages (16→64→128→256→512 channels) + a bottleneck (`encdec`) + 4 decoder stages back down to 16 channels, final conv predicts a **residual** added to the input (`pred = inp + x`), i.e. the network predicts the noise, not the clean image directly.
- Input/output is 4-channel RGGB, not 3-channel RGB.
