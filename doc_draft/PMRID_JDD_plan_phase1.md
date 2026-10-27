# 联合去噪+Demosaic：架构改造 + 训练pipeline（Phase 1）

## Context

PMRID 现在的 `models/net_torch.py::Network` 只做 raw 域去噪：输入/输出都是 H/2×W/2×4 的 RGGB
packed bayer。目标是在不破坏现有去噪能力、且能从 `models/torch_pretrained.ckp` 微调的前提下，
给网络加一个“demosaic 头”，让它同时输出 H×W×3 的全分辨率 RGB（去噪+demosaic 一次完成）。

已经讨论并确认的关键设计决策：
- **共享 trunk，双头并联**：新分支从现有 `out0` 之后的 16 通道特征（已经是 H/2×W/2，天然适合
  depth-to-space）分叉出去，不改动原有去噪分支的权重结构，`models/torch_pretrained.ckp` 可以
  直接加载到除新头以外的所有层。
- **全程保持线性域**：不能像纯demosaic网络（参考 `../../AIDMSC_research/DMSC_NestUnet`）那样把
  整个网络搬到 gamma 域——那样会让预训练权重失效、也会让 PMRID 现有的泊松+高斯噪声合成（物理上
  只在线性raw域成立）失真。所以 trunk 输入输出继续是线性域，demosaic 分支也在同一线性域监督。
- **训练数据两个来源互补**：
  - 真实 clean raw（已有 `dataset/SID_Sony_raw/index.json`，231张，RGGB）→ 噪声合成物理正确，
    但没有"真"RGB demosaic GT（本阶段*不*引入离线demosaic教师算法，只用它做去噪监督）。
  - 公开 sRGB 图（本阶段用 `../../AIDMSC_research/DMSC_NestUnet/DataSets/Evaluating/Kodak24`
    + `McMaster`，共42张，仅用于验证pipeline是否走得通）→ 反Gamma(2.2) 近似得到线性RGB，这个
    线性RGB本身就是精确的demosaic GT（不需要教师算法），代价是CCM≈单位阵的近似、且这42张图不足
    以训练出真正泛化的demosaic能力。
- 两个来源统一用同一套 `DataAug` 噪声合成/KSigma流程，只是`demosaic loss`只在 synthetic 分支
  上计算（真实raw分支本阶段没有RGB GT，demosaic loss权重=0）。

**本阶段（Phase 1）的目标**：把架构改造和训练pipeline跑通、验证正确性（loss下降、能从预训练
checkpoint正常微调、输出shape/数值合理），不追求最终质量，也不包含ONNX导出/tiled推理的适配——
那些留到pipeline验证通过、数据源scale up之后再做（用户已确认先用Kodak24/McMaster摸底）。

## Design

### 1. `models/net_torch.py` — 架构改造

把 `Network.forward` 拆分出一个共享特征方法（纯重构，不改变现有行为/权重名）：

```python
def _shared_features(self, inp):
    conv0 = self.conv0(inp)
    conv1 = self.enc1(conv0)
    conv2 = self.enc2(conv1)
    conv3 = self.enc3(conv2)
    conv4 = self.enc4(conv3)
    conv5 = self.encdec(conv4)
    up3 = self.dec1((conv5, conv3))
    up2 = self.dec2((up3, conv2))
    up1 = self.dec3((up2, conv1))
    x = self.dec4((up1, conv0))
    return self.out0(x)          # 16ch, H/2 x W/2

def forward(self, inp):
    x = self._shared_features(inp)
    x = self.out1(x)
    return inp + x
```
（现有调用方 `run_benchmark_pytorch.py`/`export_onnx.py`/`tile_infer.py`/`train_pytorch.py`
都只调用 `Network()(inp)`，这个重构对它们完全透明。）

新增 `NetworkDemosaic(Network)`：
```python
class NetworkDemosaic(Network):
    def __init__(self):
        super().__init__()
        self.out1_rgb = Conv2D(in_channels=16, out_channels=12, kernel_size=3,
                                stride=1, padding=1, is_seperable=False, has_relu=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # 零初始化，让初始 rgb_residual≈0，训练一开始输出=廉价demosaic基线
        # (Conv2D(is_seperable=False)返回的是 nn.Sequential({'conv': nn.Conv2d(...)})，
        # 所以是 .conv.weight/.bias，不是下标索引)
        nn.init.zeros_(self.out1_rgb.conv.weight)
        nn.init.zeros_(self.out1_rgb.conv.bias)

    @staticmethod
    def bilinear_rggb_to_rgb(rggb_denoised):
        # rggb_denoised: (N,4,H/2,W/2) = [R,Gr,Gb,B] -> 廉价双线性demosaic基线 (N,3,H,W)
        r = rggb_denoised[:, 0:1]
        g = (rggb_denoised[:, 1:2] + rggb_denoised[:, 2:3]) / 2
        b = rggb_denoised[:, 3:4]
        rgb_half = torch.cat([r, g, b], dim=1)
        return F.interpolate(rgb_half, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inp):
        feat = self._shared_features(inp)
        denoised_rggb = inp + self.out1(feat)
        rgb_residual = self.pixel_shuffle(self.out1_rgb(feat))
        rgb = self.bilinear_rggb_to_rgb(denoised_rggb) + rgb_residual
        return denoised_rggb, rgb

    @classmethod
    def from_pretrained(cls, ckp_path):
        net = cls()
        state = torch.load(str(ckp_path), map_location='cpu')
        missing, unexpected = net.load_state_dict(state, strict=False)
        new_head_keys = {'out1_rgb.conv.weight', 'out1_rgb.conv.bias'}
        assert not unexpected, f'unexpected keys in checkpoint: {unexpected}'
        assert set(missing) == new_head_keys, f'unexpected missing keys: {set(missing) - new_head_keys}'
        return net
```
`from_pretrained`里的`assert`是刻意照搬 `train_qat_pytorch.py::extract_float_state_dict`、
以及 CLAUDE.md 里“confirmed all 141 parameter tensors different”那类“显式验证而不是假设”的仓库
惯例。

PixelShuffle 的 12→3 通道排布：`out1_rgb`最后输出12通道，`PixelShuffle(2)`按
`c*4+i*2+j`展开（c=颜色: 0=R,1=G,2=B；i,j∈{0,1}为2x2块内行/列）。构造训练标签时要用对应的
`PixelUnshuffle`或手写pack函数产生完全一致的通道顺序（见下面`utils.py`的新函数）——不需要和
RGGB的`bayer2rggb`通道序对齐，这是两个独立的packing约定，各自自洽即可。

### 2. `utils.py` — 新增 RGB→mosaic 的packing函数

```python
@classmethod
def rgb_to_mosaic_rggb(cls, rgb_01):
    """给定 (H, W, 3) 的"假设已经是sensor能响应到的"RGB图，直接在每个2x2块的四个位置
    采样对应通道，模拟一次bayer CFA采样。输出(H/2, W/2, 4)，通道序与bayer2rggb一致:
    [top-left, top-right, bottom-left, bottom-right] = [R, G, G, B]."""
    r = rgb_01[0::2, 0::2, 0]
    g_tr = rgb_01[0::2, 1::2, 1]
    g_bl = rgb_01[1::2, 0::2, 1]
    b = rgb_01[1::2, 1::2, 2]
    return np.stack([r, g_tr, g_bl, b], axis=-1)
```
这是纯新增的 classmethod，不改动 `RawUtils` 里任何现有方法。

### 3. `dataset/synthetic_mosaic.py` — 新文件，公开sRGB图像数据源

```python
class SyntheticMosaicImages(Dataset):
    """从一个目录读取普通 sRGB 图片(png/tif/jpg)，反Gamma近似成线性RGB，随机裁剪，
    mosaic成RGGB，返回和 CleanRawImages 同构的 (rggb01_chw, g_mean_01) 二元组，外加
    精确的线性RGB裁剪块(用作demosaic分支的GT——这是它相对真实raw数据的优势：不需要
    任何demosaic教师算法，GT就是原图本身)。

    和 CleanRawImages.random_flip_and_crop 不同，这里不需要维护"相位offset"：因为
    是对完整RGB先裁剪/翻转、再mosaic(而不是对已经mosaic过的单通道阵列裁剪/翻转)，
    mosaic永远在固定相位(左上=R)采样，结果天然是canonical RGGB，不会因裁剪/翻转
    产生相位错位。
    """
    IMAGE_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, image_dir, opts: DataAugOptions, gamma: float = 2.2):
        self.paths = sorted(p for p in Path(image_dir).rglob('*') if p.suffix.lower() in self.IMAGE_EXTS)
        if not self.paths:
            raise ValueError(f'no images found under {image_dir}')
        self.opts = opts
        self.gamma = gamma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_bgr = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        img01 = img_bgr[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, [0,1]
        linear01 = img01 ** self.gamma                            # 近似反Gamma到线性域

        tH, tW = self.opts.output_shape
        H0, W0 = linear01.shape[:2]
        if H0 < tH or W0 < tW:
            raise ValueError(f'{self.paths[index]} ({H0}x{W0}) smaller than crop size {tH}x{tW}')
        y0 = np.random.randint(0, H0 - tH + 1) // 2 * 2
        x0 = np.random.randint(0, W0 - tW + 1) // 2 * 2
        crop = linear01[y0:y0+tH, x0:x0+tW]

        if np.random.rand() > 0.5:
            crop = np.flip(crop, axis=1)
        if np.random.rand() > 0.5:
            crop = np.flip(crop, axis=0)
        crop = np.ascontiguousarray(crop)

        rggb01 = RawUtils.rgb_to_mosaic_rggb(crop)
        rggb01_chw = np.ascontiguousarray(rggb01.transpose(2, 0, 1))
        rgb01_chw = np.ascontiguousarray(crop.transpose(2, 0, 1))
        g_mean_01 = float(crop[..., 1].mean())
        return rggb01_chw.astype(np.float32), g_mean_01, rgb01_chw.astype(np.float32)
```
裁剪起点 `// 2 * 2` 保证偶数，让2x2 mosaic phase在裁剪后依然对齐——这个仍然需要，只是不需要
`CleanRawImages`那种"按flip XOR offset"的bookkeeping。

### 4. `dataset/training_pytorch.py` — 扩展 `DataAug.transform`（新增可选参数，向后兼容）

现状：`brightness_aug`内部自己采样随机scale `s`。要让demosaic分支的RGB GT和去噪分支的RGGB GT
经历*完全相同*的brightness/KSigma/inp_scale变换（只是RGB GT不叠加噪声），需要把`s`的采样和
应用拆开，并给`transform`加一个可选的`batch_rgb01`参数：

```python
def brightness_aug(self, img_batch, orig_gmean):
    low, high = self.opts.target_brighness_range
    orig_gmean = np.asarray(orig_gmean)
    N = len(orig_gmean)
    btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N,)))
    s = np.clip(btarget / orig_gmean, 0.01, 1.0)
    s = torch.as_tensor(s, dtype=torch.float32, device=self.device).reshape(-1, 1, 1, 1)
    return img_batch * s, s        # <-- 新增返回 s

def transform(self, batch_img01, batch_g_mean, batch_rgb01=None):
    batch_imgs = batch_img01.to(self.device, dtype=torch.float32) * self.opts.camera_value_scale
    batch_gt, s = self.brightness_aug(batch_imgs, batch_g_mean)     # <-- 接收 s
    batch_imgs, batch_iso = self.add_noise(batch_gt)
    cvt_k, cvt_b = self.k_sigma(batch_iso)

    batch_imgs = batch_imgs * cvt_k + cvt_b
    batch_gt = batch_gt * cvt_k + cvt_b
    batch_imgs = batch_imgs / self.opts.camera_value_scale * self.opts.inp_scale
    batch_gt = batch_gt / self.opts.camera_value_scale * self.opts.inp_scale

    if batch_rgb01 is None:
        return batch_imgs, batch_gt, cvt_k

    # 对RGB GT施加完全相同的 brightness/KSigma/inp_scale 仿射链，但不加噪声
    batch_rgb = batch_rgb01.to(self.device, dtype=torch.float32) * self.opts.camera_value_scale
    batch_rgb = batch_rgb * s
    batch_rgb = batch_rgb * cvt_k + cvt_b
    batch_rgb = batch_rgb / self.opts.camera_value_scale * self.opts.inp_scale
    return batch_imgs, batch_gt, cvt_k, batch_rgb
```
`brightness_aug`签名变化（多返回一个`s`）是唯一有风险的改动——需要确认仓库里没有其它调用方
假设它只返回一个值（目前只有`transform`内部调用，搜索确认后再改）。`transform`本身因为新增
参数有默认值`None`，对 `train_pytorch.py`/`train_qat_pytorch.py`现有两个位置参数调用完全
向后兼容。

### 5. `train_demosaic_pytorch.py` — 新文件，训练脚本

结构参考 `train_qat_pytorch.py`（复用 `add_data_aug_args`/`build_data_aug_options`/
`get_loss_l1`，以及它的 best-checkpoint 追踪模式：`epoch_best_{epoch}.ckp`并删除旧best）：

```python
from itertools import cycle
from dataset.synthetic_mosaic import SyntheticMosaicImages
from models.net_torch import NetworkDemosaic
from train_pytorch import add_data_aug_args, build_data_aug_options, get_loss_l1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-data-dir', type=Path, required=True,
                         help='e.g. dataset/SID_Sony_raw (needs index.json)')
    parser.add_argument('--synth-image-dir', type=Path, required=True,
                         help='e.g. DMSC_NestUnet/DataSets/Evaluating/Kodak24/test (flat sRGB images)')
    add_data_aug_args(parser)
    parser.add_argument('--init-ckp', type=Path, default=Path('models/torch_pretrained.ckp'))
    parser.add_argument('--demosaic-weight', type=float, default=1.0)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--ckp-dir', default=Path('models/checkpoints_demosaic'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-5, type=float)
    parser.add_argument('--num-epoch', default=10, type=int)
    parser.add_argument('--save-every', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    args.ckp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    net = NetworkDemosaic.from_pretrained(args.init_ckp).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    aug_opts = build_data_aug_options(args)
    train_aug = DataAug(aug_opts, device=device)
    real_ds = CleanRawImages(data_dir=args.real_data_dir, opts=aug_opts)
    synth_ds = SyntheticMosaicImages(args.synth_image_dir, opts=aug_opts)
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # synth数据集很小(42张)，以它决定一个epoch的长度；real_loader循环取用
    real_iter_factory = lambda: cycle(real_loader)

    global_step = 0
    best_loss = float('inf')
    best_epoch = None
    for epoch in range(args.num_epoch):
        net.train()
        real_iter = real_iter_factory()
        epoch_losses = []
        for synth_batch in tqdm(synth_loader, dynamic_ncols=True):
            # --- 真实raw: 只有去噪loss ---
            imgs, g_means = next(real_iter)
            noisy, gt_rggb, norm_k = train_aug.transform(imgs, g_means)
            optimizer.zero_grad()
            pred_rggb, _ = net(noisy)
            loss_real = get_loss_l1(pred_rggb, gt_rggb, norm_k)
            loss_real.backward()
            optimizer.step()

            # --- 合成sRGB: 去噪loss + demosaic loss ---
            imgs, g_means, rgb01 = synth_batch
            noisy, gt_rggb, norm_k, gt_rgb = train_aug.transform(imgs, g_means, rgb01)
            optimizer.zero_grad()
            pred_rggb, pred_rgb = net(noisy)
            loss_denoise = get_loss_l1(pred_rggb, gt_rggb, norm_k)
            loss_demosaic = (pred_rgb - gt_rgb).abs().mean()
            loss_synth = loss_denoise + args.demosaic_weight * loss_demosaic
            loss_synth.backward()
            optimizer.step()

            epoch_losses.append(loss_real.item() + loss_synth.item())
            if global_step % 20 == 0:
                tqdm.write(f"epoch {epoch} step {global_step}: real={loss_real.item():.4f} "
                           f"denoise={loss_denoise.item():.4f} demosaic={loss_demosaic.item():.4f}")
            global_step += 1

        epoch_loss = float(np.mean(epoch_losses))
        tqdm.write(f"epoch {epoch} mean loss: {epoch_loss:.4f}")
        # ... 照抄 train_qat_pytorch.py 的 epoch_{epoch}.ckp / epoch_best_{epoch}.ckp 保存逻辑,
        #     torch.save(net.state_dict(), ...) 即可 (NetworkDemosaic的state_dict可以直接
        #     torch.load回NetworkDemosaic, 不需要像QAT那样做key过滤)
```

## Verification

1. **纯架构验证**（不需要数据）：构造随机`(1,4,64,64)`输入跑一次`NetworkDemosaic().forward`，
   确认返回 `(1,4,64,64)` 和 `(1,3,128,128)` 两个tensor，数值非NaN。
2. **重构不变性**：对同一随机输入，重构前后的`Network.forward`（即拆出`_shared_features`前后）
   输出完全一致（bit-exact），确认拆分没有引入行为变化。
3. **预训练加载**：`NetworkDemosaic.from_pretrained('models/torch_pretrained.ckp')`跑通，
   `missing keys`恰好是新头的2个参数名，`unexpected keys`为空。
4. **端到端微调冒烟测试**（实际在`AIISP_init`环境跑）：
   ```
   conda run -n AIISP_init python train_demosaic_pytorch.py \
       --real-data-dir dataset/SID_Sony_raw \
       --synth-image-dir "../../AIDMSC_research/DMSC_NestUnet/DataSets/Evaluating/Kodak24/test" \
       --output-shape 256 256 \
       --batch-size 2 --num-epoch 2 \
       --ckp-dir models/checkpoints_demosaic_smoketest
   ```
   （McMaster图片500x500，Kodak24图片768x512/512x768，`--output-shape`默认512x512对McMaster
   会裁剪越界，所以冒烟测试用256x256；先只用Kodak24验证，McMaster可以作为第二轮加进来。）
   确认：不报错、loss(real/denoise/demosaic三项)都能看到数值且不是NaN/Inf、几个epoch后
   `loss_demosaic`有下降趋势、保存的`.ckp`能被`NetworkDemosaic().load_state_dict(...)`
   (这次不用`strict=False`)干净加载回去。
5. **目视检查**：跑完冒烟测试后，加一小段脚本/临时代码，对一张Kodak24测试图跑`net.eval()`的
   `forward`，把`pred_rgb`存成png目视检查（应该能看出合理的图像结构，不要求此阶段质量多高）。

## 后续（本次不做，留作下一阶段）

- `export_onnx.py`/`tile_infer.py`对双输出、PixelShuffle→ONNX `DepthToSpace`、RGB分支tile
  坐标×2的适配。
- 把真实raw数据的demosaic GT也接上（引入`colour-demosaicing`的Menon2007做教师算法，这个包
  现在`AIISP_init`环境里没装，需要`pip install colour-demosaicing`）。
- 扩大synthetic数据源规模（超过Kodak24/McMaster这42张）。
- loss函数升级（当前只用L1，可以参考`../../AIDMSC_research/DMSC_NestUnet`引入SSIM）。
