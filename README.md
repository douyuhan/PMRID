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

### Preparing training data from a different sensor

`dataset/convert_arw_to_raw.py` converts Sony `.ARW` raw files (e.g. the SID "See in the Dark" Sony subset, placed under `dataset/SID_Sony/`) into this repo's `.raw` + `index.json` training format in one pass — reads each file's bayer mosaic via `rawpy`, writes `.raw` (uint16, filename suffixed `_H_{height}_W_{width}`), a quick-look PNG (simple ISP, not colorimetrically accurate), and an `index.json` with `bayer_pattern`/`black_level`/`white_level`/`g_mean_01` decoded straight from the file's own metadata. `INPUT_DIR`/`OUTPUT_DIR`/`PNG_DIR` are edited directly in the script (not CLI flags). Used to build `dataset/SID_Sony_raw/` for the QAT run below.

### QAT (quantization-aware training) fine-tuning

**⚠️ Every result below predates a noise-synthesis bug fix (see "Joint denoising + demosaicing (JDD)" below) and has not been re-run.** `dataset/training_pytorch.py::DataAug.add_noise` had a bug that made synthesized training noise nearly independent of the actual clean image; `train_qat_pytorch.py` needs no code change to pick up the fix, but re-running it for post-fix numbers is deferred for now -- treat every finding below as observed under the old, now-fixed noise model.

`train_qat_pytorch.py` fine-tunes an existing fp32 checkpoint with fake-quantization inserted (`prepare_qat_fx`, per-channel symmetric int8 weights — matching `quantize_onnx.py`'s own `--per-channel` default), so the exported weights are already shaped to tolerate that quantization scheme rather than only calibrating scale/zero-point after the fact. It shares the same data-augmentation CLI options as `train_pytorch.py` above (same Reno 10x defaults, `--iso-range` widened to `800 25600`):
```
python3 train_qat_pytorch.py --init-ckp models/torch_pretrained.ckp --data-dir /path/to/clean_images --ckp-dir ./checkpoints_qat --num-epoch 20
```
Meant to be a brief fine-tune (small `--learning-rate`, few `--num-epoch`) adapting already-converged weights, not training from scratch (`--batch-size` defaults to 4 — best per-sample CPU throughput measured for this fake-quant-wrapped model specifically). Each epoch's mean training loss is tracked; an improvement saves `epoch_best_{epoch}.ckp` (previous best deleted), alongside the regular `epoch_{epoch}.ckp` per `--save-every`. **Always evaluate the best checkpoint, not the last epoch** — training loss is not monotonic in practice. It never calls PyTorch's own `convert_fx()` — that raises `Per Channel Quantization is currently disabled for transposed conv`, and every decoder stage here uses a `ConvTranspose2d` for its 2x upsample — so the saved checkpoint is a plain fp32 `.ckp`, meant to be fed through the existing `export_onnx.py` → `quantize_onnx.py` pipeline exactly like `models/torch_pretrained.ckp`, calibrating against `dataset/calibrate/calibrate.json` (a curated subset of `dataset/benchmark/`, now spanning ISO 800–25600, same schema) rather than the full benchmark set:
```
python3 export_onnx.py checkpoints_qat/epoch_best_7.ckp --dtype fp32 --output qat_fp32.onnx
python3 quantize_onnx.py qat_fp32.onnx --benchmark dataset/calibrate/calibrate.json --num-tiles 200
```
**Tried end-to-end on real data (231 images, 20 epochs) — did not close the int8 quality gap.** The best-checkpoint's quantized result came out close to (not clearly better than) quantizing the vanilla non-QAT checkpoint; weights had only shifted ~2-5% in magnitude from the init checkpoint. See CLAUDE.md for the full result and — more importantly — the root cause this pointed to (a per-ISO activation magnitude mismatch that weight fine-tuning doesn't address; see the int8 quantization section below).

**A stronger re-run (`--learning-rate 1e-4`, 10x, `--num-epoch 50`) made things worse, not better** — weights shifted more (confirmed: 1.22x max per-channel ratio vs. the earlier run's 1.05x) but the checkpoint's own fp32 output had already visibly degraded from the original (noisier render, confirmed both numerically — 0.02-0.08 mean abs diff vs. vanilla fp32 — and by eye), before quantization even entered the picture. Its quantized result was correspondingly worse too, compounding on an already-degraded baseline. Keep QAT learning rates conservative for this checkpoint/dataset (`~1e-5`) — a naive "bigger LR, more epochs" escalation actively hurts here.

### Joint denoising + demosaicing (JDD) — experimental, Phase 1 (train_JDD_pytorch.py)

`train_JDD_pytorch.py` fine-tunes `models/net_torch.py::NetworkDemosaic` — `Network` plus a second head, sharing the same encoder/decoder trunk, that demosaics the already-denoised RGGB output into a full-resolution RGB image (a `Conv2D(16→12)` + `PixelShuffle(2)` tapping the same `out0` feature the denoise head uses). A single network does both jobs in one forward pass (`pred_rggb, pred_rgb = net(noisy_rggb)`), fine-tuned from the existing denoise-only `models/torch_pretrained.ckp` rather than trained from scratch:

```
python3 train_JDD_pytorch.py \
    --real-data-dir dataset/SID_Sony_raw \
    --synth-image-dir dataset/train_JDD \
    --init-ckp models/torch_pretrained.ckp \
    --ckp-dir ./checkpoints_jdd --num-epoch 15 \
    --cpu-threads 4 --low-priority
```

Two complementary training data sources, alternated every step:
- `--real-data-dir`: real sensor clean raw (`CleanRawImages` format, e.g. `dataset/SID_Sony_raw`) — physically correct noise synthesis, but no real demosaic ground truth exists for it yet, so it only supervises the denoise loss.
- `--synth-image-dir`: ordinary sRGB photos (`dataset/synthetic_mosaic.py::SyntheticMosaicImages`), approximately linearized (`img ** 2.2`) and mosaicked (`RawUtils.rgb_to_mosaic_rggb`) — the un-mosaicked image is an *exact* demosaic ground truth, so these batches supervise both the denoise and (`--demosaic-weight`-scaled) demosaic losses. Both sources run through the same noise-synthesis/KSigma pipeline as the rest of this repo's training (`DataAug.transform`, now also carrying a parallel RGB ground truth through the same brightness/KSigma/inp_scale chain).

**CPU-only considerations** (no GPU on this project's dev machine, and training is meant to run for hours without hogging the machine): `--cpu-threads` (default 4) caps PyTorch/OpenCV threading — also set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS` to the same value in the launching shell, since `torch.set_num_threads` alone doesn't reach numpy/BLAS. `--low-priority` (default on) runs at Windows `BELOW_NORMAL` priority so the OS scheduler favors foreground work regardless of thread count. `--auto-resume` (default on) picks up from `training_state.json` after an interruption (this machine's sleep/power settings killed the training process multiple times during the actual run below — a real, repeated occurrence, not a hypothetical) without redoing completed epochs; `epoch_history.csv` is an append-only per-epoch loss log that survives any number of such interruptions (unlike plain stdout redirection, which gets truncated on each relaunch). `run_jdd_training_loop.ps1` (project root) is a self-healing wrapper that keeps relaunching the training process until `--num-epoch` is reached, launched as a detached process specifically so it survives whatever killed the plain backgrounded command.

**Phase 1 training: completed.** `--synth-image-dir dataset/train_JDD` (DIV2K_train_HR 800 + Flickr2K_HR 2650 = 3450 images; `DIV2K_valid_HR`'s 100 images deliberately kept out, held back for future visual/quantitative checks) + `dataset/SID_Sony_raw` (231 images), 15 epochs, batch size 1. **Best checkpoint: `models/checkpoints_jdd/epoch_best_13.ckp`.** Verified along the way: the new head is zero-initialized so training starts from a sane bilinear-demosaic baseline; loading `models/torch_pretrained.ckp` reproduces the plain `Network`'s output exactly; an initial smoke-test run (42 Kodak24/McMaster images, 2 epochs) already showed a clean forward pass visually matching its ground truth closely. See CLAUDE.md for the full design rationale (in particular, why this stays in PMRID's linear raw domain rather than the gamma domain a standalone demosaic network would normally use), and for the CPU-courtesy/resume features' implementation details.

**A pre-existing noise-synthesis bug was found and fixed while building this**, in the training pipeline every noise-based training script here shares (`train_pytorch.py`, `train_qat_pytorch.py`, and this script): `dataset/training_pytorch.py::DataAug.add_noise` clamped its Poisson shot-noise rate to `[0, 1]`, but the image values it's computed from routinely reach into the hundreds (in `camera_value_scale` units) — so the clamp was crushing the synthesized noise to be nearly independent of the actual image for any non-near-black pixel (confirmed visually: a synthesized "noisy" mosaic was indistinguishable from pure static before the fix). Fixed by keeping only the lower bound (`.clamp(min=0)`). This predates and is unrelated to the JDD work itself — it's a bug in shared training infrastructure that this work happened to surface by being the first to visually compare a clean vs. noisy synthesized sample side by side. **Only `dataset/training_pytorch.py` was fixed** (`dataset/training.py`/`train_meg.py`, the MegEngine originals, have the identical bug but are no longer kept in sync — see above). `train_qat_pytorch.py` needs no code change to benefit from the fix, but re-running it for post-fix QAT results is being deferred for now (see the QAT section's warning above) — full details and evidence in CLAUDE.md.

#### Evaluating a JDD checkpoint (--jdd, export_onnx.py / tile_infer.py / run_benchmark_pytorch.py)

All three scripts below now take a `--jdd` flag to work with a `NetworkDemosaic` checkpoint (like `epoch_best_13.ckp` above) instead of a plain denoise-only `Network`:

```
python3 export_onnx.py models/checkpoints_jdd/epoch_best_13.ckp --jdd --dtype fp32
python3 tile_infer.py --onnx models/checkpoints_jdd/epoch_best_13_fp32_256x256_jdd.onnx \
    --jdd --benchmark /path/to/meta_info.json --save-dir /path/to/output \
    --compare-full models/checkpoints_jdd/epoch_best_13.ckp
python3 run_benchmark_pytorch.py models/checkpoints_jdd/epoch_best_13.ckp --jdd \
    --benchmark /path/to/meta_info.json --save-dir /path/to/output
```

`export_onnx.py --jdd` exports the model's `(denoised_rggb, rgb)` tuple as a 2-output graph (suffix `_jdd`); not combinable with `--bake-ksigma` (not implemented for `NetworkDemosaic`). `tile_infer.py --jdd` stitches `denoised_rggb` exactly as before and `rgb` with every tile coordinate doubled (it's at 2x the RGGB-plane resolution — same pattern as `../../AIDMSC_research/DMSC_NestUnet/tile_infer.py`'s own 2x-upsampling exit). Both `tile_infer.py --jdd` and `run_benchmark_pytorch.py --jdd` additionally save `{name}_pred_demosaic.png` — the network's own direct demosaic output — alongside the unchanged denoise outputs (`_pred.png`/`_pred.raw`, and `run_benchmark_pytorch.py`'s PSNR/SSIM, computed by exactly the same code path as without `--jdd`).

**No PSNR/SSIM is computed for `_pred_demosaic.png`** — `run_benchmark_pytorch.py::demosaic_rgb_to_display` always applies CCM+gamma (optionally WB, see below) rather than `RawUtils.bayer2rgb`'s raw-sensor pipeline, so it isn't guaranteed to be in the same color space as the benchmark's WB+CCM'd ground truth. On real sensor raw it's visibly green-tinted with no WB applied — the demosaic head was only trained on Phase-1 synthetic data (ordinary sRGB photos), which never needed WB correction, so real (un-white-balanced) raw is out-of-domain for it.

`--wb-mode {none,output,input}` (diagnostic only, `--jdd`-only) tries two quick fixes for this on an already-trained checkpoint, without retraining: `output` WBs the demosaic branch's own RGB output after the fact; `input` WBs the raw before it reaches the network (affecting both heads). Both visually fix the color cast, but each has a real cost on a different axis — confirmed by zoomed visual comparison on `dataset/test`: `output` visibly worsens edge false-color (purple/magenta fringing), while `input` visibly increases denoise noise in flat regions (and does *not* worsen edge color). Neither is a recommended long-term fix; see `doc_draft/PMRID_JDD_plan_phase3_demosaic_wb.md` and CLAUDE.md for the full root-cause analysis and the training-time alternatives considered.

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

**Known issue — severe quality loss at high ISO, root cause identified and fixed (for onnxruntime-CPU targets)**: quantizing and then running on high-ISO samples (ISO 12800/25600) produces a subjectively bad (over-exposed, noisy) result, even though fp32/fp16 look fine on the exact same images with the exact same KSigma calibration. This is **not** a calibration-coverage problem (confirmed by directly including those exact high-ISO images in the calibration set — no improvement) and **not** fixed by QAT (see above). The actual cause: KSigma's `cvt_k(iso)` compresses overall activation magnitude by ~31x between ISO 800 and 25600 (measured directly: `cvt_k` = 1.98 / 1.00 (anchor) / 0.063 at those three ISOs) — fp32 doesn't care (full relative precision at any magnitude), but `quantize_static`'s one fixed per-tensor scale, calibrated across that whole range, is necessarily dominated by the large low-ISO values, starving the ~31x-smaller high-ISO activations of quantization levels.

**Fix: `quantize_onnx.py --dynamic`** — uses `onnxruntime.quantization.quantize_dynamic` instead of `quantize_static`; no `--benchmark` calibration set needed at all (activation scale is computed fresh from the actual tensor on every inference call, so it automatically adapts to whatever magnitude a given ISO produces):
```
python3 quantize_onnx.py models/torch_pretrained_fp32_256x256.onnx --dynamic --output models/torch_pretrained_fp32_256x256_dynamic_int8.onnx
```
Confirmed on the same 8 high-ISO `dataset/test` images: mean abs diff vs. fp32 dropped from 0.012–0.043 (static) to **0.0022–0.0034 (dynamic)** — a ~10-15x improvement, no longer growing with ISO the way static quantization's error did, and visually indistinguishable from the fp32 render. Declared external I/O is still plain `tensor(float)`, so it drops into `tile_infer.py --dtype fp32` unmodified. One caveat: the 4 `ConvTranspose` ops (one per decoder stage) have no dynamic-quant kernel support and stay fp32 — not a "pure" int8 graph — and computing the scale on every call has real runtime cost vs. static quantization's baked-in constant, worth profiling on the actual target hardware. `--weight-type uint8` made no difference; `--reduce-range` made it measurably worse (it's a CPU-overflow-avoidance knob, not an accuracy lever — leave it off unless the target hardware needs it).

**Side effect of `--dynamic` + tiling**: the quantization noise's *spatial texture* is blocky, tracking `tile_infer.py`'s tile-ownership regions (not a bug — verified directly). `tiled_infer_rggb` calls `sess.run()` once per tile, and `DynamicQuantizeLinear` computes its scale fresh from each call's own input tensor — confirmed by extracting the actual scale from the graph for two differently-scaled synthetic tiles (range `[0,5]` → scale `0.0196`; range `[0,150]`, 30x larger → scale `0.588`, also ~30x larger). So each tile's int8 rounding precision depends on everything else in that tile (e.g. a bright highlight elsewhere in the same tile coarsens quantization for unrelated flat regions in it too), and two visually-similar flat patches in different tiles can end up with different quantization noise levels. Confirmed absent when comparing two different fp32 models on the same images (no quantization involved) — the tiling/stitching code itself is not the source. Doesn't change the headline numbers above, only their spatial distribution; see CLAUDE.md for the full verification.

**An external per-frame rescale was tried and does NOT work — ruled out by the math itself, not just an empirical shortfall.** The idea was a per-frame affine, applied outside a static-quantized graph, cancelling `cvt_k(iso)`'s magnitude compression before the network and undoing it after. Working through what it actually computes: `net_input' = img01*inp_scale`, exactly, for every ISO — i.e. it doesn't cancel an "incidental" magnitude side-effect while leaving KSigma's noise-normalization intact, it undoes KSigma's noise-normalization completely, back to the raw un-normalized noise level. `cvt_k` isn't a generic scalar that can be freely stripped off — it *is* the multiplier that maps the real, ISO-specific noise level onto the anchor's target level; magnitude compression and noise-level normalization are the same multiplication, not separable. Confirmed with real data: running the plain fp32 model with vs. without this rescale (no quantization involved at all) already shows mean abs diff 0.006–0.031 on high-ISO `dataset/test` samples — the rescale silently discards the correct denoising strength (network denoises as if the noise were anchor-level when it's actually far higher). There's no code for this in the repo.

QAT (see above) was also tried against this issue specifically: **doesn't help, and being more aggressive about it (10x the learning rate) makes things worse, not better** — the higher LR visibly degraded the checkpoint's own fp32 quality (confirmed both numerically and visually), which then also shows up as worse quantized output. Keep QAT learning rates conservative (`~1e-5`) for this checkpoint/dataset combination.

### If the target can't do dynamic quantization: hybrid int8+fp16 (quantize_onnx_hybrid.py)

A third option, staying **fully static** (no runtime scale computation, no tiling blockiness): keep specific *layers* in fp16 while quantizing the rest to int8, instead of trying to rescale around a fully-quantized graph. Reverse-engineered from a real deployed model the user confirmed looks subjectively lossless on `dataset/test`: it keeps `conv0`/`enc1` (input-adjacent) and `dec3.proj_conv`/`dec4`/`out0`/`out1` (output-adjacent, plus whatever consumes `enc1`'s/`conv0`'s skip connection) in fp16, quantizing the deep middle (`enc2`-`enc4`/`encdec`/`dec1`/`dec2`/most of `dec3`) to int8 — targeting the root cause directly, since the fp16 layers are exactly the ones that touch the raw ISO-varying input or must produce the precision-critical output residual.

```
python3 quantize_onnx_hybrid.py models/torch_pretrained_fp32_256x256.onnx --benchmark dataset/calibrate/calibrate.json --calibrate-method percentile --output models/torch_pretrained_hybrid_256x256.onnx
```

`--fp16-modules` (default: `conv0 enc1 dec3.proj_conv dec4 out0 out1`) takes dotted PyTorch module-name prefixes; every node under one is excluded from quantization via `nodes_to_exclude` then converted to actual float16 via `onnxconverter_common.float16.convert_float_to_float16` (`keep_io_types=True`, so the graph's external I/O stays float32 like every other model in this repo). The script also always protects whichever node produces the graph's final output tensor, even if it falls outside every `--fp16-modules` prefix — `Network.forward`'s `pred = inp + x` is a bare op, not inside any named submodule, and missing it left the graph's declared output as `tensor(float16)` instead of `tensor(float)` (caught by actually checking `sess.get_outputs()[0].type`, not assumed).

Verified on the same 8 high-ISO `dataset/test` samples: mean abs diff vs. fp32 was 0.0068–0.0255 with `--calibrate-method minmax` (~40% better than plain static PTQ's 0.012–0.043), and **0.0041–0.0141 with `--calibrate-method percentile`** (visually confirmed clean on the worst sample) — a real, usable improvement, though not quite matching `--dynamic`'s 0.0022–0.0034. See CLAUDE.md for the full derivation, including guidance on choosing `--fp16-modules` for a different network architecture.

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
