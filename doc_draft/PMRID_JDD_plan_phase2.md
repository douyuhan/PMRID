# 适配 run_benchmark_pytorch.py / export_onnx.py / tile_infer.py 以支持 NetworkDemosaic (JDD)

## Context

`train_JDD_pytorch.py`已经训练出`models/checkpoints_jdd/epoch_best_13.ckp`（`NetworkDemosaic`，
denoise+demosaic共享trunk，双输出`(denoised_rggb, rgb)`）。现在要看这个checkpoint的实际效果，但
目前`run_benchmark_pytorch.py`/`export_onnx.py`/`tile_infer.py`三个脚本都只认识单输出的`Network`，
完全不知道怎么处理双输出模型——这正是Phase 1计划里明确列为"Not yet done"的一项，现在要补上。

**范围已经和用户确认**：只做外部KSigma路径的适配（现状`tile_infer.py`所有测试都是用这条路径），
不给`NetworkDemosaic`加`--bake-ksigma`支持（export_onnx.py现有的这个选项保持只对`Network`可用）。

**一个关键的domain细节，贯穿三个文件的改动**：`NetworkDemosaic`的demosaic分支（直接的`rgb`输出）
是用Phase 1的公开sRGB图片（Kodak24/McMaster/DIV2K/Flickr2K经反Gamma近似线性化）训练出来的——这些
训练数据本身就是"正常照片"，没有做从raw sensor视角的WB/CCM矫正。所以`rgb`输出的"配套后处理"只需要
**正向Gamma**，**不需要**`RawUtils.bayer2rgb`里的WB增益和CCM矩阵那一套（那套是给真实sensor raw用
的）。在真实benchmark数据（Reno10x/`dataset/test`）上跑，这个`rgb`输出预期会看起来偏色——这是已知
的、已经在CLAUDE.md记录过的Phase 1局限（demosaic分支从没见过真实sensor raw的样子），不是这次改动
要修的bug，只是要在代码注释/输出文件命名上说清楚，别让人误以为是实现错误。

**连带后果**：`rgb`输出在真实benchmark数据上和`meta.ROIs`+`CCM`算出来的GT RGB不是同一个色彩空间，
算PSNR/SSIM是在比较两个不同color space的东西，数字没有意义还可能误导人——所以这次改动**只给demosaic
分支的`rgb`输出存可视化PNG，不算PSNR/SSIM**（denoise分支原有的PSNR/SSIM逻辑完全不变）。

## Design

### 1. `run_benchmark_pytorch.py` —— 新增 `DemosaicDenoiser` + 一个纯显示用的gamma helper

```python
from models.net_torch import Network, NetworkDemosaic   # 新增 NetworkDemosaic

def demosaic_rgb_to_display(rgb_01, gamma=2.2):
    """NetworkDemosaic.rgb 输出已经是直接的、线性域、"色彩已经对的"图(见Context里的domain说明)，
    展示时只需要正向gamma，不能走 RawUtils.bayer2rgb 的 WB/CCM 那条路(那是给真实sensor raw用的)。"""
    return np.clip(rgb_01, 0, 1) ** (1 / gamma)


class DemosaicDenoiser:
    """和 Denoiser 结构基本一致，但包的是 NetworkDemosaic，run() 额外多返回一个全分辨率 rgb
    (已经做完 KSigma 逆归一化，[0,1]线性域，还没做展示用的gamma)。"""

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0, device='cpu'):
        net = NetworkDemosaic()
        net.load_state_dict(torch.load(str(model_path), map_location=device))  # strict=True：
        # JDD训练出的checkpoint已经包含全部key，不是from_pretrained那种"只加载denoise部分"的场景
        net.eval()
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01):
        # 和 Denoiser.pre_process 完全一样，直接复制/复用(两个类目前没有共同基类，Phase 1暂不重构，
        # 后续如果还要加第三个Denoiser变体再考虑提取公共基类)
        ...

    def run(self, bayer_01, iso):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale
        inp = np.ascontiguousarray(inp_rggb, dtype=np.float32)
        with torch.no_grad():
            inp_t = torch.from_numpy(inp).to(self.device)
            pred_rggb_t, pred_rgb_t = self.net(inp_t)
            pred_rggb = (pred_rggb_t[0] / self.inp_scale).cpu().numpy().transpose(1, 2, 0)
            pred_rgb_full = (pred_rgb_t[0] / self.inp_scale).cpu().numpy().transpose(1, 2, 0)
        pred_rggb = self.ksigma(pred_rggb, iso, inverse=True)
        pred_rgb_full = self.ksigma(pred_rgb_full, iso, inverse=True)   # 同一个per-frame仿射，
        # 对rgb和rggb两个tensor都适用(训练时DataAug.transform就是这么对两者施加完全相同的变换的)

        ph, pw = self.ph, self.pw
        pred_rggb = pred_rggb[ph:-ph, pw:-pw]
        # rgb分辨率是RGGB-plane的2倍，pre_process里的padding是2*ph/2*pw个raw像素(= ph/pw个
        # RGGB-plane像素)，换算到rgb的全分辨率空间正好还是2*ph/2*pw —— 裁剪量和原始padding量相等
        pred_rgb_full = pred_rgb_full[2*ph:-2*ph, 2*pw:-2*pw]
        return RawUtils.rggb2bayer(pred_rggb), pred_rgb_full
```

`main()`/`run_benchmark()` 新增 `--jdd` 开关：为真时用`DemosaicDenoiser`替代`Denoiser`，`pred_bayer`
走向不变(denoise PSNR/SSIM、`_pred.raw`、`_pred.png`逻辑完全不动)，额外把`pred_rgb_full`经
`RawUtils.to_canonical_rggb(..., pattern=pattern)`转回原始bayer_pattern朝向(和`inp_rgb`/
`pred_rgb`/`gt_rgb`那次转换合并成一次调用)，再用`demosaic_rgb_to_display`转展示域，
`save_rgb_png(save_dir/f'{name}_pred_demosaic.png', ...)`存盘——复用现成的`save_rgb_png`，不需要
新的保存函数。不计算这个输出的PSNR/SSIM(原因见Context)，但在循环里打印一行提示说明。

### 2. `export_onnx.py` —— 新增 `--jdd` 开关

```python
from models.net_torch import Network, NetworkDemosaic
```
`main()`里：
- `--jdd`存在时，`net = NetworkDemosaic(); net.load_state_dict(torch.load(opt.model, map_location='cpu')); net.eval()`
  (跳过`load_net()`现有的只认`Network`的版本，或者给`load_net`加个`network_cls`参数更干净)。
- `--jdd`和`--bake-ksigma`同时给出时直接报错退出(`raise ValueError(...)`)，提示"NetworkDemosaic暂不
  支持bake-ksigma"——按本仓库一贯的"宁可报错也不要默默算错"的风格，不要静默忽略其中一个。
- `output_names`：`--jdd`时用`['denoised_rggb', 'rgb']`，否则保持`['output']`不变。
  `torch.onnx.export`对tuple输出会自动按顺序导出成多个graph输出，不需要额外处理。
- 导出文件名：在现有`f'{stem}_{dtype}_{height}x{width}{suffix}.onnx'`的`suffix`里加上`_jdd`
  (`--bake-ksigma`已经是用`_ksigma`这个suffix的模式，保持一致)。
- 验证环节（现有代码对比ONNX和PyTorch输出的max abs diff）：改成对`onnx_out`/`torch_out`两边都是
  长度为1或2的list，循环按`output_names`逐个打印diff，而不是假设只有一个输出。

### 3. `tile_infer.py` —— 新增双输出tiling + `--jdd` 开关

新函数`tiled_infer_rggb_dual`，和`tiled_infer_rggb`并列（不是改造原函数——单输出场景的调用方不用动）：

```python
def tiled_infer_rggb_dual(sess, input_name, rggb, tile_h, tile_w, margin, np_dtype, extra_feed=None):
    """同tiled_infer_rggb，但ONNX图有2个输出：denoised_rggb(和输入tile同分辨率，拼接逻辑和
    tiled_infer_rggb完全一样) + rgb(分辨率是输入tile的2倍，拼接坐标要整体乘2 —— 和
    ../../AIDMSC_research/DMSC_NestUnet/tile_infer.py给它自己的2x上采样L1 exit做的事完全同一个
    模式：tile position / ownership bounds 在输入(RGGB-plane)空间算一次，应用到高分辨率输出时
    所有坐标整体翻倍)。"""
    H, W = rggb.shape[:2]
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

    out_rggb = np.zeros((Hp, Wp, 4), np.float32)
    out_rgb = np.zeros((Hp * 2, Wp * 2, 3), np.float32)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            tile = rggb[y0:y0 + tile_h, x0:x0 + tile_w].transpose(2, 0, 1)[np.newaxis].astype(np_dtype)
            feed = {input_name: tile, **(extra_feed or {})}
            outputs = sess.run(None, feed)
            tile_rggb = outputs[0][0].transpose(1, 2, 0).astype(np.float32)
            tile_rgb = outputs[1][0].transpose(1, 2, 0).astype(np.float32)

            oy0, oy1 = y_bounds[i], y_bounds[i + 1]
            ox0, ox1 = x_bounds[j], x_bounds[j + 1]
            out_rggb[oy0:oy1, ox0:ox1] = tile_rggb[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
            out_rgb[2*oy0:2*oy1, 2*ox0:2*ox1] = tile_rgb[2*(oy0-y0):2*(oy1-y0), 2*(ox0-x0):2*(ox1-x0)]

    grid_info = f'{len(ys)}x{len(xs)} tiles (tile {tile_h}x{tile_w}, margin {margin}, stride {stride_h}x{stride_w})'
    return out_rggb[:H, :W], out_rgb[:H*2, :W*2], grid_info
```

新类`TiledOnnxDemosaicDenoiser`（和`TiledOnnxDenoiser`并列，复用`tile_positions`/
`ownership_bounds`之外的全部既有helper）：外部KSigma的normalize/denormalize对两路输出都做一遍
（同一个per-frame仿射，道理同`DemosaicDenoiser`），margin裁剪：`rggb`按现有逻辑裁`margin`
(RGGB-plane像素)，`rgb`要裁`2*margin`(换算到全分辨率空间，和原始padding量对应，推导同
`run_benchmark_pytorch.py`的`DemosaicDenoiser.run`)。

`main()`新增`--jdd`：
- 和`--bake-ksigma`一样，校验`len(sess.get_outputs())`应为2（不是1），不匹配直接报错（照抄现有
  `--bake-ksigma`那段输入数检查的写法）。
- `--jdd`和`--bake-ksigma`同时给出时报错（export_onnx.py本来就不会产出这种组合的模型）。
- `--compare-full`：`--jdd`时从`run_benchmark_pytorch.py`导入`DemosaicDenoiser`(而不是
  `Denoiser`)做未分块的参照，两路输出的diff都打印。
- 保存结果：`--jdd`时额外用`demosaic_rgb_to_display`（从`run_benchmark_pytorch.py`导入）+
  `save_rgb_png`存一份`{name}_pred_demosaic.png`，和现有的`_pred.png`/`_pred.raw`（denoise
  分支，逻辑完全不变）并列。

## Verification

```
# 1. 导出JDD onnx（--height/--width沿用export_onnx.py现有256默认值）
python export_onnx.py models/checkpoints_jdd/epoch_best_13.ckp --jdd --dtype fp32

# 2. tile_infer.py 实跑 + 和未分块PyTorch模型对比（用dataset/test或dataset/benchmark都行）
python tile_infer.py --onnx models/checkpoints_jdd/epoch_best_13_fp32_256x256_jdd.onnx \
    --jdd --benchmark dataset/test/meta_info.json --save-dir /tmp/jdd_out \
    --compare-full models/checkpoints_jdd/epoch_best_13.ckp

# 3. run_benchmark_pytorch.py --jdd 跑一遍确认denoise PSNR/SSIM逻辑没被破坏、demosaic可视化图能存出来
python run_benchmark_pytorch.py models/checkpoints_jdd/epoch_best_13.ckp --jdd \
    --benchmark dataset/test/meta_info.json --save-dir /tmp/jdd_bench_out
```
确认点：两个脚本都不报错；`--compare-full`打印的两路diff都是小量级(量级参考现有denoise-only
tile_infer.py的~1e-2，不应该离谱地大，离谱大说明2x坐标换算哪里算错了)；`_pred_demosaic.png`
目视检查——结构应该合理，色彩在真实raw数据上偏色是预期内的(见Context)，不是这次要修的问题；
`run_benchmark_pytorch.py`原有的denoise PSNR/SSIM数值应该和不加`--jdd`时跑`epoch_best_13.ckp`
对应的denoise分支完全一致(因为对`pred_bayer`的处理代码路径没有变化)。
