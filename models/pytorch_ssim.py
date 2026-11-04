"""Differentiable SSIM loss, ported from ../../AIDMSC_research/DMSC_NestUnet/pytorch_ssim
(itself the standard Po-Hsun-Su/pytorch-ssim implementation) -- pulled in-repo rather than
pip-installed since no SSIM-capable package (pytorch_msssim, kornia, ...) is installed in
AIISP_init, and this is a small, self-contained, dependency-free module (just
torch.nn.functional.conv2d).

Used by train_JDD_pytorch.py's --demosaic-ssim-epochs schedule (see that file's docstring
for the DMSC_NestUnet-inspired SSIM-then-L1 schedule this feeds into).

The C1/C2 stabilization constants below (0.01**2, 0.03**2) assume img1/img2 are already
normalized to roughly [0, 1] (the standard SSIM convention, L=1 dynamic range) -- true for
DMSC_NestUnet's own sRGB [0,1] images, but NOT true of this repo's raw gt_rgb/pred_rgb
tensors, which live in a per-sample-ISO-dependent magnitude (KSigma's cvt_k/cvt_b affine,
then * inp_scale -- see dataset/training_pytorch.py::DataAug.transform). Callers here MUST
renormalize to ~[0,1] first (divide by norm_k * inp_scale, the same norm_k already computed
for get_loss_l1) before calling SSIM/ssim() -- otherwise these constants silently stop doing
their job (negligible relative to variance terms at the wrong magnitude) and the metric
degrades to the unstabilized raw formula.
"""
import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(device=img1.device, dtype=img1.dtype)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
