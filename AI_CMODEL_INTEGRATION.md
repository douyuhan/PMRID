# PMRID → ISP Cmodel Integration Notes

Reference info to carry into the Cmodel project session when wiring up the exported PMRID ONNX graph for inference there. Values/line references are current as of this repo's `export_onnx.py`/`tile_infer.py`/`utils.py`/`run_benchmark_pytorch.py`.

**Yes, `tile_infer.py` is directly useful as a reference implementation** — it's the only place in this repo that already does "arbitrary-size real image → tiles → per-tile ONNX inference → stitched full image", which is exactly the shape of the problem in the Cmodel. Its tiling/stitching math (`tile_positions`/`ownership_bounds`), the bayer-coherent reflect-padding scheme, and the external KSigma application are all things a C implementation needs to reproduce bit-for-bit (or close to it) to match this Python pipeline's output.

## 1. Exported ONNX artifact(s)

Whichever `.onnx` file you copy over, you need to know these four things about it (they're not otherwise recoverable from the file alone without checking):

| Property | Where it's decided | How to check on the file |
|---|---|---|
| `--height`/`--width` (tile size, RGGB-plane space) | `export_onnx.py --height/--width` (default 256x256) | `onnx model's input[0].shape[2:]` |
| `--dtype` (`fp32`/`fp16`) | `export_onnx.py --dtype` | `input[0].type` — `tensor(float)` vs `tensor(float16)` |
| `--bake-ksigma` (1 vs 2 graph inputs) | `export_onnx.py --bake-ksigma` | `len(model.get_inputs())` — 1 = KSigma external, 2 = KSigma baked in (`input`, `iso`) |
| Sensor noise calibration (if baked) | `export_onnx.py --k-coeff/--b-coeff/--anchor/--v` (defaults = Reno 10x, see §3) | not recoverable from the graph — record separately |

Current local exports in `models/` (regenerate as needed):
- `torch_pretrained_fp32_256x256.onnx` — fp32, 1 input, KSigma external
- `torch_pretrained_fp16_256x256.onnx` — fp16, 1 input, KSigma external
- `torch_pretrained_fp16_256x256_ksigma.onnx` — fp16, 2 inputs (`input`, `iso`), KSigma baked in

## 2. Network I/O contract

- Input tensor `input`: shape `(1, 4, tile_h, tile_w)`, dtype fp32 or fp16. This is **not** raw bayer and **not** 3-channel RGB — it's 4-channel RGGB planes (see §4 for the exact packing).
- If `--bake-ksigma`: second input `iso`, shape `(1,)`, same dtype as `input`. A genuine per-call runtime value, never a baked constant — different ISO values must produce different outputs.
- Output tensor: shape `(1, 4, tile_h, tile_w)`, same dtype. This is the **final denoised** RGGB tile — the network already adds its predicted residual to the input (`pred = inp + noise_estimate`) internally, and if KSigma is baked in, the output is already de-normalized back to the original (non-anchor-ISO) scale. No further "add back the input" step is needed on the caller side.
- Tile size is in **RGGB-plane space** = half the resolution of the original bayer image in each dimension. A 256x256 ONNX tile corresponds to a 512x512 region of the raw bayer image.

## 3. KSigma noise normalization (needed if the graph does NOT have `--bake-ksigma`)

If the ONNX graph is the plain (1-input) variant, the Cmodel must apply this affine transform itself, before feeding a tile in and after reading a tile out. Reference: `run_benchmark_pytorch.py`'s `KSigma` class, also mirrored as a traceable `KSigmaModule` in `export_onnx.py` (for the `--bake-ksigma` path — read that one if you want the exact same math expressed with basic ops instead of `np.poly1d`).

```
K(iso)     = k_coeff[0]*iso + k_coeff[1]                                  # linear
Sigma(iso) = b_coeff[0]*iso^2 + b_coeff[1]*iso + b_coeff[2]                # quadratic

k, sigma     = K(iso), Sigma(iso)
k_a, sigma_a = K(anchor), Sigma(anchor)

cvt_k = k_a / k
cvt_b = (sigma / k^2 - sigma_a / k_a^2) * k_a

# forward (before feeding the network):
normalized = (rggb_01 * V) * cvt_k + cvt_b
normalized = normalized / V
network_input = normalized * inp_scale     # inp_scale = 256.0

# inverse (after reading the network's output):
x = (network_output / inp_scale) * V
x = (x - cvt_b) / cvt_k
denormalized_rggb_01 = x / V
```

Default (Reno 10x) coefficients, hardcoded in `run_benchmark_pytorch.py`/`tile_infer.py`/`export_onnx.py`'s defaults:
- `k_coeff = [0.0005995267, 0.00868861]`
- `b_coeff = [7.11772e-7, 6.514934e-4, 0.11492713]`
- `anchor = 1600` (ISO the network was trained at — KSigma collapses to identity when `iso == anchor`)
- `V = 959.0`
- `inp_scale = 256.0`

**`iso` is per-sample, real camera ISO** (not a gain multiplier like "256x" — see earlier conversation in this repo's history if that distinction matters for your data), read from each image's metadata. It must vary per real input; don't hardcode it to `anchor`.

## 4. Bayer pattern → canonical RGGB, and RGGB packing

The network only ever sees the **canonical RGGB** layout (top-left pixel of every 2x2 block = R). Real sensor data can be in any of 4 bayer orders; convert before inference and convert back after (this conversion is its own inverse — same operation both ways):

| `bayer_pattern` | flip rows? | flip cols? |
|---|---|---|
| `RGGB` | no | no |
| `BGGR` | yes | yes |
| `GRBG` | no | yes |
| `GBRG` | yes | no |

Reference: `utils.py::RawUtils.to_canonical_rggb` / `_PATTERN_FLIPS`.

**RGGB channel packing** (`utils.py::RawUtils.bayer2rggb`): given a canonical-RGGB 2D bayer array of shape `(H, W)`, each non-overlapping 2x2 block maps to one pixel across 4 channels, in this exact order:
- channel 0 = top-left of the block (R)
- channel 1 = top-right (G)
- channel 2 = bottom-left (G)
- channel 3 = bottom-right (B)

Result shape `(H/2, W/2, 4)`, transposed to `(4, H/2, W/2)` (then batched to `(1,4,H/2,W/2)`) for the network. `rggb2bayer` is the exact inverse reshape.

## 5. Padding for real (non-tile-sized) images

Zero-padding at image/tile boundaries measurably hurts denoising quality right at the edge (verified in this repo by diffing tiled vs. full-image inference — zero-padding caused a ~10x error spike confined to the outermost ~4px). The fix used throughout this repo: **reflect-pad the raw bayer mosaic itself** (before packing into RGGB planes), not the RGGB planes after packing, and always by an **even** number of raw pixels — both needed to keep the bayer color at each mosaic position consistent across the seam.

Example: a bayer row `R0 G0 R1 G1`, reflect-padded by 2 columns each side, must become:
```
R1 G0 | R0 G0 R1 G1 | R1 G0
```
**not**
```
G0 R0 | R0 G0 R1 G1 | G1 R1     <- WRONG: this is edge-duplicating reflect (numpy 'symmetric' / cv2.BORDER_REFLECT), flips R/G parity at the seam
```
The correct one is numpy's default `mode='reflect'` (does *not* duplicate the edge pixel). If the Cmodel's padding primitive duplicates the edge pixel by default (many do, e.g. `cv2.BORDER_REFLECT`), you likely need its "reflect-101"/no-duplicate variant instead (e.g. `cv2.BORDER_REFLECT_101`), or implement the mirror index math directly.

Two places in this repo use this scheme, for two different reasons:
- `run_benchmark_pytorch.py::Denoiser.pre_process` — pads up to a multiple of 32 (in RGGB-plane space) so the whole image can go through the network in one shot (4 encoder downsample stages, so H/W must be /32).
- `tile_infer.py::TiledOnnxDenoiser.run` — pads by `margin` RGGB-plane pixels (= `2*margin` raw pixels) before tiling, so the outermost tile's true-image-edge side gets real mirrored context instead of relying solely on the network's own implicit zero-padding at the true edge. Cropped back off by the same amount after inference.

## 6. Tiling & stitching (only relevant if the Cmodel also needs to tile — i.e. real images bigger than one ONNX tile)

Reference: `tile_infer.py::tile_positions`, `ownership_bounds`, `tiled_infer_rggb`. Same scheme as the sibling `../DnCNN-PyTorch_Saoyan/tile_infer.py`, generalized to 4-channel RGGB.

- **`tile_positions(length, tile, stride)`**: 1D tile start positions covering `[0, length)`. Starts at 0, advances by `stride` each time, except the last tile is shifted back to end exactly at `length` (no overshoot past the true edge).
- **`stride = tile - 2*margin`** (clamped to >= 1).
- **`ownership_bounds(positions, tile, length)`**: splits `[0, length)` into one non-overlapping slice per tile — each pair of neighboring tiles' overlap region is cut at its midpoint. Every output pixel is written by exactly one tile (whichever "owns" that region), not blended/averaged.
- Applied independently in both H and W to get a 2D grid of tiles; each tile is `(tile_h, tile_w, 4)` RGGB, run through the graph, and only its "ownership" sub-region is copied into the final stitched output.
- Default `margin = 32` (RGGB-plane pixels). In testing here, increasing margin 32→96 did **not** shrink the residual diff vs. untiled full-image inference (~0.037 max, ~4e-4 mean on `[0,1]` scale) — that residual is dominated by onnxruntime-vs-PyTorch floating-point kernel differences, not tile-seam artifacts, so don't expect a bigger margin alone to close a numerical gap of that kind.

## 7. Output rounding (if the Cmodel quantizes the float result back to a fixed bit depth)

Use **round-half-away-from-zero** ("真正的四舍五入"), not banker's rounding (round-half-to-even, which e.g. C's `rint()`/numpy's `round()` may default to depending on rounding mode). Reference implementation (`utils.py::round_half_up`):
```
round_half_up(x) = sign(x) * floor(abs(x) + 0.5)
```
This repo hit a real bug from this: pixel values land on exact `.5` boundaries often enough (e.g. whenever upstream float math produces a clean fraction of `1/65535` etc.) that banker's rounding measurably biased the quantized output. All three fixed-point save paths in this repo (`RawUtils.bayer2rgb`'s internal demosaic quantization, `save_rgb_png`, `save_bayer_raw`) were fixed to use this instead of `np.round`.

## 8. Known non-issues (don't re-diagnose these if they show up again)

- Visualization PNGs (`save_rgb_png`, and this repo's `bayer2rgb`) have exact duplicate outermost 1-2 rows/cols — this is `cv2.cvtColor`'s edge-aware Bayer demosaic having no real neighborhood at the image border, confirmed to reproduce even on raw GT bayer with no network involved. It's a simplified/fast visualization ISP, not representative of a real ISP pipeline, and doesn't affect the saved bayer `.raw` output (which never goes through demosaic). Not necessarily relevant to the Cmodel (which presumably has a real ISP), but worth knowing this repo's own visualization has this quirk if you're eyeballing PNGs from here side-by-side with Cmodel output.
- A small residual float diff (~4e-4 mean, ~0.037 max on `[0,1]` scale) between tiled ONNX and untiled PyTorch inference is expected and is a kernel-implementation artifact, not a correctness bug — see §6.

## 9. Code pointers in this repo (for exact reference while porting)

- KSigma math: `run_benchmark_pytorch.py:17-39` (numpy) / `export_onnx.py`'s `KSigmaModule` (torch, traceable, basic ops)
- Bayer pattern flips / RGGB packing: `utils.py::RawUtils` (`to_canonical_rggb`, `bayer2rggb`, `rggb2bayer`)
- Reflect-padding scheme: `run_benchmark_pytorch.py::Denoiser.pre_process`, `tile_infer.py::TiledOnnxDenoiser.run`
- Tiling/stitching: `tile_infer.py::tile_positions`, `ownership_bounds`, `tiled_infer_rggb`
- Rounding: `utils.py::round_half_up`
- Export options/defaults: `export_onnx.py` (top of file for `DEFAULT_K_COEFF` etc., `parse_args` for all flags)
