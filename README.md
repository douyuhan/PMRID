# Practical Mobile Raw Image Denoising (PMRID)

Code and dataset for ECCV20 paper [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935).

## Dataset

### Downloads
- [OneDrive](https://megvii-my.sharepoint.cn/:f:/g/personal/wangyuzhi_megvii_com/Et4v2Z7CkRxHnbcFUq6RXZMBfXUrlm_Se5OVDvcdujVsMA?e=vcfJWs)
- [Kaggle](https://www.kaggle.com/dataset/1bdc5cd707cfbb3ee842eb3cbfe93495dbba88017d29f295f8edbcb8f8790556)

### Usage

The dataset includes two 7zip files:
- `reno10x_noise.7z` contains DNG raw images shot by an _OPPO Reno 10x_ phone for noise parameter estimation (refer Sec 3.1 and 5.1 in the paper)
- `PMRID.7z` is the benchmark dataset described in Sec 5.2 in the paper

The structure of `PMRID.7z` is
```
- benchmark.json  # meta info
- Scene1/
  \- Bright/
     \- exposure-case1/ 
         \- input.raw   # RAW data for noisy image in uint16
          - gt.raw      # RAW data for clean image in uint16
      + case2/
  + Dark/
+ Secne2/
```

All metadata for images are listed in `benchmark.json`:
```python
{
   "input": "path/to/noisy_input.raw",
   "gt": "path/to/clean_gt.raw",
   "meta": {
       "name": "case_name",
       "scene_id": "scene_name",
       "light": "light condition",
       "ISO": "ISO",
       "exp_time": "exposure time",
       "bayer_pattern": "BGGR",
       "shape": [3000, 4000],
       "wb_gain": [r_gain, g_gain, b_gain],
       "CCM": [   # 3x3 color correction matrix
           [c11, c12, c13], 
           [c21, c22, c23], 
           [c31, c32, c33]
       ],
       "ROIs": [  # patch ROIs to calculate PSNR and SSIM, x0 is topleft
           [topleft_w, topleft_h, bottomright_w, bottomright_h]
       ]
   }
}
```

## Pre-trained Models and Benchmark Script

Both [PyTorch](https://pytorch.org/) and [MegEngine](https://megengine.org.cn/) pre-trained models are provided in the `models` directory.
`Python >= 3.6` is required to run the benchmark scripts.

- `run_benchmark_meg.py` runs the MegEngine model (`models/mge_pretrained.ckp`):
  ```
  pip install -r requirements.txt
  python3 run_benchmark_meg.py --benchmark /path/to/PMRID/benchmark.json models/mge_pretrained.ckp
  ```
- `run_benchmark_pytorch.py` runs the PyTorch model (`models/torch_pretrained.ckp`) and additionally supports:
  - all four standard bayer patterns (`RGGB`/`BGGR`/`GRBG`/`GBRG`), read per-sample from `meta.bayer_pattern`
  - `--save-dir DIR` to dump, per sample, full-image RGB PNG visualizations of the input/prediction/ground-truth (after the same simple ISP used for the PSNR/SSIM metrics) plus the denoised bayer result as a `.raw` file (same `uint16` format as the input, restored to the sample's original `bayer_pattern`)
  ```
  pip install -r requirements.txt
  python3 run_benchmark_pytorch.py --benchmark /path/to/PMRID/benchmark.json models/torch_pretrained.ckp --save-dir /path/to/output
  ```

### Test notes
- PMRID can replace GIC, 2DNR, 3DNR, YNR, based on IMX415 dataset.


## Training

`train_pytorch.py` trains `models/net_torch.py::Network` (**the recommended path** — see below for the MegEngine original this was ported from):
```
python3 train_pytorch.py --data-dir /path/to/clean_images --ckp-dir ./checkpoints
```
Unlike the benchmark dataset (real paired noisy/gt captures), training data is just **clean** (well-exposed, low-noise) raw images — noise is synthesized on the fly each batch (Poisson shot noise + Gaussian read noise, then KSigma-normalized to an anchor ISO, then rescaled to the network's actual input magnitude) rather than captured. `--data-dir` must contain an `index.json` listing each image's `path`/`width`/`height`/`black_level`/`white_level`/`bayer_pattern`/`g_mean_01`.

The data-augmentation/KSigma calibration is a set of CLI options (not a JSON config file), defaulting to the Reno 10x calibration — this repo's only actual target sensor, matching `run_benchmark_pytorch.py`'s `KSigma(...)` exactly:

| option | meaning | default |
|---|---|---|
| `--k-coeff` / `--b-coeff` | sensor noise model polynomial coefficients (same calibration as `run_benchmark_pytorch.py`'s `KSigma(K_coeff=..., B_coeff=...)`) | Reno 10x calibration |
| `--noise-value-scale` | the scale `--k-coeff`/`--b-coeff` were calibrated at | `959.0` |
| `--camera-value-scale` | scale applied to the clean `[0,1]` image before noise synthesis — same role as `KSigma`'s `V` at inference | `959.0` |
| `--anchor-iso` | the ISO noise level everything gets normalized to (same as `KSigma`'s `anchor`) | `1600.0` |
| `--inp-scale` | final rescale before the network sees the data — must match whatever `inp_scale` the inference side uses | `256.0` (same as `run_benchmark_pytorch.py`'s `Denoiser`/`export_onnx.py`) |
| `--iso-range` | random ISO sampled per image per batch to synthesize noise at | `800 6400` |
| `--output-shape` | random-crop size in raw bayer-pixel space (network input ends up half that per side, 4x channels) | `512 512` |
| `--target-brightness-range` | brightness-darkening augmentation target range (never brightens) | `0.02 0.5` |

The first five are **sensor-calibration constants** — this is also where the `KSigma` constants used by the benchmark/ONNX/quantization scripts above should be calibrated from, so training and inference stay consistent for a given sensor; for a fixed target sensor these can just stay at their Reno 10x defaults. `--iso-range`/`--output-shape`/`--target-brightness-range` are **training-strategy choices** unrelated to sensor calibration — override them per training run without touching the sensor constants (e.g. widen `--iso-range` to cover ISOs beyond the benchmark set). Checkpoints save to `--ckp-dir/epoch_N.ckp`, `torch.load`-compatible with every other script in this repo that takes a `.ckp` path.

`train_meg.py`/`dataset/training.py` is the original MegEngine version this was ported from (added upstream after this repo's initial pure-inference release); kept for reference, but MegEngine has no installable wheel on Windows, so it isn't runnable here, and per the user it's intentionally left un-synced with fixes (like the `inp_scale` normalization step above) that were only found/verified by actually running the PyTorch side.

### QAT (quantization-aware training) fine-tuning

`train_qat_pytorch.py` fine-tunes an existing fp32 checkpoint with fake-quantization inserted (`prepare_qat_fx`, per-channel symmetric int8 weights — matching `quantize_onnx.py`'s own `--per-channel` default), so the exported weights are already shaped to tolerate that quantization scheme rather than only calibrating scale/zero-point after the fact. It shares the same data-augmentation CLI options as `train_pytorch.py` above (same Reno 10x defaults):
```
python3 train_qat_pytorch.py --init-ckp models/torch_pretrained.ckp --data-dir /path/to/clean_images --ckp-dir ./checkpoints_qat --num-epoch 10
```
Meant to be a brief fine-tune (small `--learning-rate`, few `--num-epoch`) adapting already-converged weights, not training from scratch. It never calls PyTorch's own `convert_fx()` — that raises `Per Channel Quantization is currently disabled for transposed conv`, and every decoder stage here uses a `ConvTranspose2d` for its 2x upsample — so the saved checkpoint is a plain fp32 `.ckp`, meant to be fed through the existing `export_onnx.py` → `quantize_onnx.py` pipeline exactly like `models/torch_pretrained.ckp`, just calibrate against `dataset/calibrate/calibrate.json` (a curated subset of `dataset/benchmark/`, same schema) rather than the full benchmark set:
```
python3 export_onnx.py checkpoints_qat/epoch_9.ckp --dtype fp32 --output qat_fp32.onnx
python3 quantize_onnx.py qat_fp32.onnx --benchmark dataset/calibrate/calibrate.json --num-tiles 200
```
This has only been verified as a plumbing check (trains without error, checkpoint loads cleanly, re-exports/re-quantizes successfully) — whether it actually improves int8 quality on real data is still untested; see CLAUDE.md.

## ONNX export & tiled inference

- `export_onnx.py` exports the PyTorch model to a fixed-shape ONNX graph (input/output `(1, 4, H, W)` RGGB planes):
  ```
  python3 export_onnx.py models/torch_pretrained.ckp --height 256 --width 256 --dtype fp32
  ```
  `--dtype` supports `fp32` (plain export), `fp16` (fp32 export, then the graph is cast to float16 — the model itself is never run in fp16), and `int8` (reserved placeholder; prints why it isn't implemented and exits without writing a file, since int8 needs calibration-based post-training quantization, not a plain cast).
  `--bake-ksigma` additionally exports the ISO-dependent KSigma noise normalization into the graph itself (as a second `iso` input) instead of leaving it to be applied externally in numpy — by default (flag off) the graph is a bare CNN and the caller (`Denoiser`/`TiledOnnxDenoiser`) applies KSigma before/after it, same as `run_benchmark_pytorch.py`:
  ```
  python3 export_onnx.py models/torch_pretrained.ckp --height 256 --width 256 --dtype fp32 --bake-ksigma
  ```
- `tile_infer.py` runs that fixed-shape ONNX graph over real (arbitrary-size) raw images by splitting each into overlapping tiles matching the graph's `H`/`W`, running each tile through the graph, and stitching the non-overlapping "ownership" region of each tile back together:
  ```
  python3 tile_infer.py --onnx models/torch_pretrained_fp32_256x256.onnx --benchmark /path/to/PMRID/benchmark.json --save-dir /path/to/output --compare-full models/torch_pretrained.ckp
  ```
  `--margin` controls how many edge pixels of each tile are discarded before stitching (larger = less tile-seam artifact, more compute); `--compare-full` additionally runs the untiled PyTorch model on the same samples and reports the difference, to help tune it. `--bake-ksigma` must be passed if (and only if) `--onnx` was exported with `export_onnx.py --bake-ksigma` — it makes `tile_infer.py` skip its own external KSigma step and feed each sample's ISO straight into the graph instead; passing it inconsistently with how the model was actually exported is a hard error (checked against the ONNX model's declared input count), not a silent wrong result.

  `--dtype` is about the ONNX graph's own declared **external** input/output tensor type (what numpy dtype gets fed in/read back), not whether the graph is quantized internally — pass `fp32` for both a plain export and an int8-quantized `quantize_onnx.py` model (its default QDQ format keeps external I/O as float32; only the internal weights/activations are int8), and `fp16` only for an `export_onnx.py --dtype fp16` export (a genuine external type change). `int8` is rejected — no export here ever has a genuinely int8 external interface.

## Post-training quantization (int8)

`quantize_onnx.py` calibrates and quantizes a **fp32** ONNX export (from `export_onnx.py`, without `--bake-ksigma`) down to int8, using real images from a benchmark dataset as calibration data:
```
python3 quantize_onnx.py models/torch_pretrained_fp32_256x256.onnx --benchmark /path/to/PMRID/benchmark.json --num-tiles 200
```
Calibration images don't need to match the ONNX graph's tile size — they only need to be at least as large as one tile. `quantize_onnx.py` crops a grid of representative tiles out of each real `input.raw` image (never `gt.raw`), already normalized (KSigma + `inp_scale`, using that image's own real ISO) exactly the way the network sees them at inference — this is what makes the calibration meaningful, not just "some float32 arrays". `--per-channel` (recommended, on by default), `--calibrate-method` (`minmax`/`entropy`/`percentile`/`distribution`), and `--quant-format` (`qdq`, the default and most broadly compatible) let you tune accuracy vs. compatibility.

The resulting int8 model still declares `float32` external input/output (standard for onnxruntime's QDQ quantization format), so it runs through `tile_infer.py` unmodified with `--dtype fp32` — but **quantization is never automatically trusted**: re-run `tile_infer.py --compare-full` (or the full PSNR/SSIM benchmark) against the fp32 baseline before using an int8 model for anything real. Passing a model exported with `--bake-ksigma`, or one that isn't fp32, is rejected with a clear error rather than silently mis-quantized. A model validated against onnxruntime's own CPU execution is also not guaranteed to behave identically on a different int8 runtime/accelerator.

## Fully fixed-point inference (tile_infer_hw.py)

`tile_infer.py` always crosses the network's boundary in float32, even with an int8-quantized model. `tile_infer_hw.py` is for the case where PMRID needs to be a drop-in denoise module inside an otherwise fully fixed-point ISP pipeline — the upstream/downstream modules hand off/expect plain integer RAW data (e.g. RAW12) with no floating point at all. It reimplements the RAW bit-depth conversion, KSigma normalize/denormalize, and the final requantize back to RAW pixel data entirely in integer-only fixed-point arithmetic (visualization/`.raw` saving/metrics are left exactly as-is, on float32, same as the other scripts):
```
python3 tile_infer_hw.py --onnx models/torch_pretrained_fp32_256x256.onnx --benchmark /path/to/PMRID/benchmark.json --save-dir /path/to/output --bit-module 12 --compare-full models/torch_pretrained.ckp
```
`--bit-module` (default 12) is the RAW module interface width; if a sample's `raw_bitWidth` differs, it's converted to/from `--bit-module` via an exact integer bit-shift (always possible since both are plain bit-depths). `--net-bit-int`/`--net-bit-frac` (default 13/18, signed) size the fixed-point format used for the (per-frame, not per-pixel) KSigma coefficients and the value crossing into/out of the network — tune these down once you know your real deployment's ISO range, to match your target hardware's actual register width. The only unavoidable floating-point step is the literal handoff to onnxruntime (its graph still declares `float32` I/O regardless — see `tile_infer.py`'s `--dtype`), which is a lossless container conversion, not new computation. `--compare-full` verifies the result against the untiled float32 PyTorch reference the same way `tile_infer.py` does.

Every int↔float conversion of a fixed-width pixel value across this repo (loading `.raw` files, `tile_infer_hw.py`'s own fixed-point math, etc.) normalizes via `2**bit` — a bit-shift, matching real hardware — not `2**bit - 1`.


## Citation
```
@inproceedings{wang2020,
	title={Practical Deep Raw Image Denoising on Mobile Devices},
	author={Wang, Yuzhi and Huang, Haibin and Xu, Qin and Liu, Jiaming and Liu, Yiqun and Wang, Jue},
	booktitle={European Conference on Computer Vision (ECCV)},
	year={2020},
	pages={1--16}
}
```
