# PMRID-JDD Demosaic偏绿问题：根因分析与候选方案（讨论稿，未实施）

## Context

Phase 2（`PMRID_JDD_plan_phase2.md`）给`run_benchmark_pytorch.py`/`export_onnx.py`/`tile_infer.py`
加上`--jdd`支持、在`dataset/test`真实raw数据上实跑`models/checkpoints_jdd/epoch_best_13.ckp`之后，
目视确认demosaic分支的`rgb`直接输出**明显偏绿**，结构（边缘、细节）和denoise分支的`_pred.png`高度
一致，色彩不对。本文档记录对这个现象的根因分析和5个候选修复方案的讨论结果——**这5个方案目前都只是
讨论，没有任何一个已经写进代码或训练过**，`epoch_best_13.ckp`本身是按"现状代码"（不含下面任何一个
方案）训练出来的。

### 根因

demosaic分支的训练数据来自`dataset/synthetic_mosaic.py::SyntheticMosaicImages`：公开sRGB照片
（Kodak24/McMaster/DIV2K/Flickr2K）经`img01 ** 2.2`反Gamma得到"线性"RGB，直接mosaic，**没有做任何
WB处理**。因为这些照片本来就是"正常照片"（已经经过原始相机的WB+CCM+Gamma处理，反Gamma只撤销了最后
一步Gamma，WB和CCM仍然"烤"在反Gamma后的图像里），所以这个训练用的mosaic是"隐式已经WB好"的——R、G、
B通道量级大致均衡。

而真实raw sensor数据（`dataset/test`，只做过BLC）在WB之前天然是G通道最亮、R和B通道系统性偏暗（CFA
滤光片响应+常见光源决定的）。demosaic分支从没见过这种量级失衡的输入，网络最早的`conv0`（非
depthwise，混合四个输入通道的那层）在"训练时见过的量级关系"和"推理时实际收到的量级关系"之间有
domain mismatch，表现为输出偏色；更深一层，这个跨通道计算如果在数据失衡的状态下进行，还可能导致
纹理/边缘处的伪色（类似传统demosaic算法"先WBG再DMSC"优于"先DMSC再WBG"的经验，对CNN demosaic大概率
同样成立，因为`conv0`在结构上也是一种跨通道的、依赖量级关系的运算）。

### 几个背景事实澄清（讨论中确认，影响方案评估）

- **`epoch_best_13.ckp`训练时两种数据每个iteration都用，不是分epoch轮流**：外层循环按
  `synth_loader`（3450张，DIV2K+Flickr2K）的长度走，`real_loader`（231张，`SID_Sony_raw`）循环
  取用；每个iteration先用一个real raw batch算denoise loss并更新一次，再用一个synth batch算
  denoise+demosaic loss并再更新一次。
- **denoise分支的权重在JDD训练过程中不是冻结的**：`NetworkDemosaic.from_pretrained()`只设定了
  训练**起点**（从`models/torch_pretrained.ckp`热启动），但`optimizer = optim.Adam(net.parameters(), lr=1e-5)`
  对共享trunk和`out1`（denoise头）、`out1_rgb`（demosaic头）**所有参数**都在更新，15个epoch下来，
  denoise相关权重已经偏离原始`torch_pretrained.ckp`，具体偏离多少目前没有直接测过（可以跑一次
  `epoch_best_13.ckp` vs `torch_pretrained.ckp`在同一benchmark上的denoise PSNR/SSIM对比来确认）。
- **real raw分支没有demosaic GT，是物理限制，不是疏漏**：真实raw只有单通道采样，没法"测量"出每个
  像素的真实RGB值（这正是demosaic要估计的东西）；denoise GT可以靠长曝光/多帧平均真实拍出来，
  demosaic GT不行。
- **real raw分支即使不能用于demosaic loss，依然有不可替代的价值**：这不是JDD才有的设计，是原始
  PMRID（`train_pytorch.py`，纯denoise）从一开始就确立的策略——两个分支的噪声合成过程完全一样（都
  走`DataAug.add_noise`），real raw真正贡献的是"干净信号"一侧的真实性：`SID_Sony_raw`是这个具体
  sensor真实拍出来的、未经任何demosaic/锐化处理的genuine raw；synthetic分支的"干净信号"是公开照片，
  早被**别的相机**的demosaic+锐化+处理流程处理过一遍，纹理统计和这个项目实际sensor的真实raw不是
  一回事。去掉real raw、只用synthetic训练在技术上可行，但会放弃这个已验证的denoise训练策略，预期
  会让denoise分支学到的"什么是细节、什么是噪声"这个判断标准脱离真实sensor的实际纹理特性——具体掉
  多少没有直接测过，可以做ablation验证。

## 候选方案

### 方案1：推理时对demosaic分支RGB输出做WBG（RGB域），再做2.2 Gamma

不改`epoch_best_13.ckp`，在`run_benchmark_pytorch.py`/`tile_infer.py`里，demosaic分支的`rgb`输出
算完之后乘一次WB gain，再做展示用的Gamma。

- **风险**：ISP pipeline的隐含顺序变成"先demosaic（在未WB的失衡数据上）、再WBG"——网络内部（尤其
  `conv0`这种跨通道层）用到的特征是在未WB的数据上算出来的，这一步算得"准不准"不会因为事后乘一个
  增益而改变。整体色偏能改善，但纹理/边缘处可能残留伪色，治标不治本。
- **工作量**：低，纯推理脚本后处理，不碰网络和训练。

### 方案2：推理时对raw输入做WBG（raw域），再送入网络

不改`epoch_best_13.ckp`，推理时先用真实`wb_gain`对raw做WB，再送入网络（denoise+demosaic一起算）。

- **风险**：denoise分支从没在WB过的数据上训练过；更精确地说，WB是对已经采集到的数字值做缩放，不
  改变单通道**内部**信噪比，但会放大**通道之间**噪声的绝对量级差异（R、B被WB放大后，绝对噪声量级
  和G的差距比WB前更大）。KSigma的`cvt_k`/`cvt_b`是不分通道的单一标量，标定时隐含假设各通道噪声
  量级相近——这个假设被打破，而且是套在**冻结权重**上，网络没机会去适应新分布，denoise效果大概率
  变差。
- **工作量**：低，纯推理脚本前处理，但有真实的denoise质量风险，需要实测量化代价。

### 方案3：训练时对synthetic分支做随机inv WB，微调（不改架构）

基于现有`epoch_best_13.ckp`继续微调；`SyntheticMosaicImages`在mosaic之前，对`linear01`做随机
inv WB（除以随机gain，模拟"未WB"状态），让synthetic分支的mosaic统计特性和real raw分支的真实
（未WB）domain对齐。

- **风险**：训练/推理domain一致了（demosaic分支不再是OOD输入），但网络内部pipeline顺序本质上还是
  "先demosaic（在未WB数据上）、从来没有过WBG这一步"——这个顺序本身可能带来的伪色上限没有解决，只是
  从"灾难性偏色"变成"训练时学出来的、相对温和的鲁棒性"。**不是方案1/2那种推理时硬凑的trade-off**：
  这是训练时让网络见过更大范围通道比例分布，是representation层面的改善，不是在一个frozen架构里
  按下葫芦起了瓢。真正的风险点是：效果上限取决于增强的gain分布和真实sensor的WB gain分布匹配得好不
  好（已实测`dataset/benchmark`的Rgain≈1.35~1.67、Bgain≈2.15~2.49，`dataset/test`的Rgain≈1.54~2.18、
  Bgain≈1.99~2.80，可以此为参考范围）。
- **工作量**：中，微调+一步数据增强，可用`epoch_best_13.ckp`热启动。

### 方案4：架构改成"先denoise再demosaic"的顺序两阶段结构，重新训练

denoise完全独立跑完（和现在`Network`架构一样），产出`denoised_rggb`；demosaic是基于这个**已完成**
的denoise结果的独立子网络（不复用`feat`中间特征），可以在demosaic子网络开始前插入WB。

- **优点**：彻底解耦两个分支的输入/domain要求——WB可以在demosaic子网络真正开始处理之前插入，不再
  受"两个分支共享同一个输入tensor"这个硬约束限制。
- **风险/代价**：demosaic子网络拿不到共享trunk内部特征复用的红利，计算量变大；demosaic子网络需要
  从头训练（denoise部分架构不变，可以复用/热启动`torch_pretrained.ckp`或`epoch_best_13.ckp`里
  denoise相关的权重，不是全部从头）。
- **工作量**：高，架构改动+重新训练一个新的demosaic子网络。

### 方案5：整个网络重新训练，统一成"WB之后的raw"作为网络输入约定

把"WB之后的raw"定为网络输入的统一约定，让denoise分支也在WB'd domain上重新训练，而不是在推理时
硬套。WBG顺序回到了demosaic之前（和传统ISP"先WBG再DMSC"的经验一致），denoise分支因为是重新训练
的，有机会学到"如何处理通道不均衡的RGGB输入"，而不是被动硬扛。

**两个分支的训练数据构造（本文档最终确认版，经过讨论修正）**：

- **real raw分支**（`CleanRawImages`，本来就是未WB的原生状态，不需要inv WB这一步）：
  ```
  真实bayer raw(4通道RGGB, 未WB) → add_noise(物理正确，在原生未WB尺度上) → WB加回去
  （4通道上做，随机gain，用 wb_gain[[0,1,1,2]] 这个现成的indexing方式展开到RGGB四通道，
  复用 RawUtils.bayer2rgb 里已经在用的同一套映射）
  ```

- **synthetic分支**（`SyntheticMosaicImages`，`linear01`本身是"隐式已WB"状态，需要先往回退）：
  ```
  sRGB → inv Gamma → (inv CCM，假设CCM是单位矩阵，无需处理) →
  inv WB（3通道RGB上做，随机gain，R除以Rgain、B除以Bgain、G不变）→
  mosaic（三通道各自独立subsample，不混合通道，inv WB的效果原样带进4通道RGGB）→
  add_noise（4通道上做，物理正确）→
  WB加回去（4通道上做，用**同一个**随机gain，即inv WB时用的那个值，不能用新的随机值）
  ```
  **为什么synthetic分支不能"保持现状不变"（这是对本方案早期讨论版本的修正）**：如果只是直接在
  `linear01`（已经均衡）的尺度上加噪声，三个通道会得到完全对称的噪声特性，这不真实——真实sensor里
  "WB gain越大的通道，WB之后噪声绝对量级越明显"是个真实现象（R/B通道往WB前推，原本信号更弱，噪声
  相对更明显）。用"inv WB → 加噪声 → 用同一个gain WB回去"这个往返操作，对**干净信号**是精确抵消的
  （`(L/G)×G = L`，GT不受影响，不需要改），但对**噪声**不是抵消的：在`L/G`这个更暗的尺度上合成的
  泊松噪声，乘回`G`之后，方差多了一个`G`倍的系数——这正好复现了真实sensor"WB gain大的通道噪声更
  明显"这个特征，real raw分支和synthetic分支的噪声合成逻辑才真正统一到了同一套物理假设下。
  - 3通道做inv WB、4通道做加噪声和WB加回去，这两个空间之间的切换是干净的：mosaic是纯粹的按位置
    subsample，不做任何跨通道混合，3通道时设定好的per-channel系数会原样带进4通道表示（G只是多出现
    一次，Gr和Gb都继承同一个G通道的处理），不会因为换了通道数而出问题。

- **推理侧**：`run_benchmark_pytorch.py`/`tile_infer.py`需要**永久性**插入raw域WB（用benchmark
  metadata里每张图真实的`wb_gain`），不再是方案1/2那种诊断用的可选开关——因为网络本身就是按"输入
  已经WB过"训练出来的。

- **风险/代价**：需要**整个网络重新训练**（可以用`epoch_best_13.ckp`热启动，`conv0`等前几层肯定要
  重新适应新的通道比例约定，大概率比从零训练收敛快，但没有把握）；推理pipeline要跟着永久改变。
- **工作量**：高，训练数据管线（两个分支都要改）+ 推理脚本永久改动 + 重新训练。

## 方案对比汇总

| | 性质 | 改动范围 | 核心风险 | 工作量 |
|---|---|---|---|---|
| 1 | 推理时后处理(RGB域WB) | 仅推理脚本 | 顺序仍是"先dmsc再wbg"，伪色风险未解决，治标不治本 | 低 |
| 2 | 推理时前处理(raw域WB) | 仅推理脚本 | denoise的KSigma噪声归一化假设被打破，且套在冻结权重上无法适应 | 低 |
| 3 | 训练时增强(synthetic分支随机inv WB) | 训练数据管线(synthetic) | 解决domain mismatch导致的灾难性偏色，但pipeline顺序本身的伪色上限未解决；效果上限取决于增强分布匹配度 | 中(可热启动) |
| 4 | 架构改动(顺序两阶段) | 架构+重新训练demosaic子网络 | 拿不到共享特征复用红利，计算量增大；demosaic子网络从头训练(denoise部分可复用) | 高 |
| 5 | 训练时统一WB'd domain(两分支都改) | 训练数据管线(两个分支)+推理脚本永久改动+重新训练 | 理论上最彻底，顺序回到"WBG在前"；denoise分支需要重新训练适应新domain，整体工作量最大 | 高 |

## 关联但独立的问题：是否要去掉real raw分支、只用synthetic数据训练

技术上可行，但不建议——会放弃PMRID项目从`train_pytorch.py`时代就确立的"用真实raw+模拟噪声训练
denoise"策略，synthetic数据的"干净信号"已经被别的相机处理过，纹理统计特性和本项目实际sensor的
真实raw不是一回事。具体影响多大，可以通过ablation实验（去掉real raw分支训练一版，对比`dataset/test`
上的denoise PSNR/SSIM）直接测出来，目前没有做过这个实验。

## 当前状态 / 下一步

以上5个方案和real raw的ablation问题，**目前都只是讨论，没有任何代码改动或训练任务已经发起**。

评估`epoch_best_13.ckp`现状，用方案1、2（纯推理时实验，不需要重新训练）最快能拿到对比结果；方案3、
5需要真正投入训练才能验证，方案4投入最大。下一步需要确认：先做哪个/哪几个方案的实验，以及是否要
顺带验证real raw ablation这个问题。

## 补充：方案1/2初次实现时引入的ISP pipeline不一致bug（已修复）

给`run_benchmark_pytorch.py`/`tile_infer.py`加`--wb-mode {none,output,input}`支持、实际跑出方案1/2
的结果之后，目视对比`_pred.png`（denoise分支）和`_pred_demosaic.png`（demosaic分支）发现两条分支
经过的ISP模块不一致，甚至有正确性bug：

- **所有`wb_mode`下，demosaic分支都缺CCM**：`demosaic_rgb_to_display`最初（phase2设计）就不加CCM
  （理由见上面Context部分），但这次为了让demosaic分支能和denoise分支公平对比，改成**始终加CCM**
  （`demosaic_rgb_to_display(rgb_01, CCM, gamma=2.2)`，`CCM`现在是必填参数）。
- **`wb_mode=input`下，denoise分支的`pred.png`偏紫**：根因是双重WB——`net_input_bayer`已经在喂进
  网络前做过一次raw域WB，`denoiser.run()`的残差输出`pred_bayer`因此已经在WB'd domain；但后续算
  `pred_rgb`时仍然对`pred_bayer`用`meta.wb_gain`做了第二次WB（`bayer2rgb`原本对
  `input_bayer`/`pred_bayer`/`gt_bayer`统一传同一个`wb_gain`）。修复：拆成两次`bayer2rgb`调用，
  `input_bayer`/`gt_bayer`始终用真实`meta.wb_gain`，`pred_bayer`在`wb_mode=='input'`时用单位增益
  `(1.0, 1.0, 1.0)`（因为它已经是WB'd的），其余模式仍用`meta.wb_gain`。这个修复顺带让
  `wb_mode=input`时的PSNR/SSIM也从"比较两个WB次数不同的东西"变成了正确的同domain比较。

修复后，两条分支在`wb_mode != 'none'`时都完整经过denoise、WB、demosaic、ccm、gamma这5个模块（顺序
因`wb_mode`而异，这正是方案1/2本身要对比的实验变量，不是bug）；`wb_mode='none'`时demosaic分支仍然
没有WB（这就是"无WB"实验组的定义），但现在有CCM了。用`epoch_best_13.ckp`在`dataset/test`上重新跑
三种`wb_mode`，`run_benchmark_pytorch.py`/`tile_infer.py`两个脚本都不再报错，且`tile_infer.py
--compare-full`的tile-vs-full diff量级和原有denoise-only baseline一致（没有引入新的数值问题）。

## 补充：方案1/2修复后的实际实验结果（已完成，结论确认）

用修复后的代码（见上一节），`export_onnx.py --jdd`导出的`models/checkpoints_jdd/epoch_best_13_fp32_256x256_jdd.onnx`
+ `tile_infer.py --jdd --wb-mode {none,output,input} --compare-full epoch_best_13.ckp`，在`dataset/test`
的8张真实raw样本上重新跑了一遍三种模式。三种模式均正常跑完，`--compare-full`打出的tile-vs-full diff
量级（mean ~1e-3级别、max ~1e-1级别）和原有denoise-only baseline一致，说明tiling/stitching本身没有
被`wb_mode`影响。

目视检查（局部放大对比，而不是只看缩略图）后，用户此前凭肉眼给出的初步判断被**确认**，且能具体定位到
两种代价分别体现在哪个维度：

- **`wb_mode=output`（方案1）：demosaic分支在高对比度边缘处的伪色（紫/品红色fringe）明显加重。**
  用`260527_real_dark_HDR_on_single_002_6300_256x_1`样本一条白色分隔线边缘局部放大对比：
  `wb_mode=none`时denoise分支`_pred.png`（走`cv2.COLOR_BAYER_BG2RGB_EA`经典demosaic）本身在这条边缘
  上就有一道很淡的紫边——这是edge-aware demosaic算法在强对比度边缘的固有伪色，可以当作baseline；
  `wb_mode=output`时demosaic分支`_pred_demosaic.png`同一条边缘的紫边明显更粗更饱和，清楚地比baseline
  重。denoise分支`_pred.png`在`wb_mode=output`下完全不受影响（方案1的WB只乘在demosaic分支自己的rgb
  输出上，不碰网络输入也不碰denoise输出），符合设计预期。
- **`wb_mode=input`（方案2）：denoise分支的去噪效果确实下降，不是此前"紫色爆炸"那次误判。**
  同一场景里一大片夜间天空平坦区域局部放大对比：`wb_mode=input`下`_pred.png`清楚可见更多残留噪点/
  颗粒感，`wb_mode=none`/`output`两种模式同一区域都是平滑干净的。双重WB bug修复之后，`wb_mode=input`
  的`_pred.png`色彩已经恢复正常（和另外两种模式视觉上一致，不再偏紫），但去噪质量这次是真实观测到
  变差，不是bug假象。
  额外发现：`wb_mode=input`下demosaic分支`_pred_demosaic.png`在同一条边缘上的伪色反而是三者中**最轻**
  的（比`wb_mode=none`的cv2 baseline还干净）——也就是说方案2在"边缘伪色"这一维度并不比baseline差，
  代价集中体现在"平坦区域去噪质量下降"这一个维度上。
- **`wb_mode=none`（原始baseline）**：demosaic分支偏绿问题依旧（设计上就是"无WB"实验组，预期内），
  方案1、方案2都能把这个偏色目视纠正到接近中性，色偏问题本身两个方案都"治好了"。

**结论**：用户此前的初步判断方向正确——方案1和方案2解决偏色问题各自付出的代价，分别集中在"边缘伪色"
和"平坦区域去噪质量"这两个**不同**维度上，不是同一种代价的强弱对比，两者都不是免费修复。这和方案1/2
文档本身标注的"均非推荐的长期修复方案，只是对已训练好的`epoch_best_13.ckp`的快速诊断"结论一致——要
同时解决偏色、不引入边缘伪色、不牺牲去噪质量，仍然需要方案3/4/5这类训练时介入的方案。

## 补充：`meta.json`缺`wb_gain`/`CCM`时的默认值兜底（已加）

`dataset/benchmark.py::RawMeta`原本把`wb_gain`/`CCM`定义成dataclass的必填字段（无默认值）——如果
某个数据集的`meta.json`条目里完全没写这两个key，`RawMeta(**x['meta'])`会直接抛
`TypeError: missing required positional argument`，报错位置和真实原因（缺标定数据）隔着一层不直观
的dataclass构造错误。

现在改成：`wb_gain`/`CCM`在dataclass里仍然声明在`ROIs`之后、`raw_bitWidth`之前（满足"有默认值的
字段必须排在无默认值字段之后"这条dataclass规则），但默认值是`None`这个"未提供"哨兵，真正的默认值
在`__post_init__`里判断再填入：

- `wb_gain`缺失 → 默认`(1.0, 1.0, 1.0)`（no-op WB gain，`RawUtils.bayer2rgb`内部
  `wb_gain[[0,1,1,2]]`展开后每个通道都乘1.0，等于没做WB）。
- `CCM`缺失 → 默认`np.eye(3)`单位矩阵（no-op CCM，和`dataset/convert_arw_to_raw.py`给SID Sony
  sensor用"没有标定CCM就假设单位矩阵"是同一个约定）。
- 两种情况都会触发一次`warnings.warn(...)`，而不是静默使用默认值——提醒这个sample的PSNR/SSIM和
  可视化图是在"没有真实WB/CCM标定"的情况下算出来的，不是真实色彩效果。

验证：构造一个不带`wb_gain`/`CCM`的`RawMeta`，确认得到预期默认值且触发2条warning；用现有
`dataset/test/meta_info.json`（每条都带真实`wb_gain`/`CCM`）跑一遍完整`BenchmarkLoader`，确认
0条warning触发（没有误伤正常数据）；`run_benchmark_pytorch.py --jdd`在该数据集上的mean PSNR/SSIM
和改动前完全一致（31.386dB / 0.892），确认这个改动对现有数据集零影响，只在缺字段时起兜底作用。
