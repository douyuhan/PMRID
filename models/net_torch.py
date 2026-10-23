#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import OrderedDict

import numpy as np


def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=True, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True, has_relu=False)

        self.proj = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True, has_relu=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
            )
        )

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=False,
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True)
        # M.init.msra_normal_(self.upsample.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = Conv2D(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=64, num_blocks=2)
        self.enc2 = EncoderStage(in_channels=64, out_channels=128, num_blocks=2)
        self.enc3 = EncoderStage(in_channels=128, out_channels=256, num_blocks=4)
        self.enc4 = EncoderStage(in_channels=256, out_channels=512, num_blocks=4)

        self.encdec = Conv2D(in_channels=512, out_channels=64, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=64, skip_in_channels=256, out_channels=64)
        self.dec2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def _shared_features(self, inp):
        """Everything through out0: the 16-channel, H/2 x W/2 feature shared by the
        denoise head (out1) and, in NetworkDemosaic, the demosaic head. Pulled out of
        forward() as a pure refactor (same ops, same module names) so a subclass can
        reuse this feature without duplicating the encoder/decoder call sequence."""
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

        return self.out0(x)

    def forward(self, inp):
        x = self._shared_features(inp)
        x = self.out1(x)

        pred = inp + x
        return pred


class NetworkDemosaic(Network):
    """Network, plus a second head that demosaics the already-denoised RGGB output
    into a full-resolution RGB image, sharing the same encoder/decoder trunk (and so
    the same pretrained weights -- see from_pretrained) as the denoise head.

    The new head taps the same out0 feature as the denoise head (16ch, H/2 x W/2) --
    the exact resolution a depth-to-space(x2) needs to land on a full-res H x W output
    -- so no additional up/downsampling stages are needed. It predicts a residual on
    top of a cheap bilinear demosaic of the *denoised* RGGB output (not the noisy
    input), initialized to ~0 so training starts from that sane baseline rather than
    from noise (see __init__'s zero-init of out1_rgb).
    """

    def __init__(self):
        super().__init__()
        self.out1_rgb = Conv2D(
            in_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1,
            is_seperable=False, has_relu=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        # Conv2D(is_seperable=False) returns nn.Sequential({'conv': nn.Conv2d(...)}),
        # so the leaf module is .conv, not a plain attribute or a [-1] index.
        nn.init.zeros_(self.out1_rgb.conv.weight)
        nn.init.zeros_(self.out1_rgb.conv.bias)

    @staticmethod
    def bilinear_rggb_to_rgb(rggb):
        """Cheap non-learned demosaic baseline: (N,4,H/2,W/2) RGGB -> (N,3,H,W) RGB,
        used only as the residual's starting point (see __init__)."""
        r = rggb[:, 0:1]
        g = (rggb[:, 1:2] + rggb[:, 2:3]) / 2
        b = rggb[:, 3:4]
        rgb_half = torch.cat([r, g, b], dim=1)
        return nn.functional.interpolate(rgb_half, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inp):
        feat = self._shared_features(inp)
        denoised_rggb = inp + self.out1(feat)
        rgb_residual = self.pixel_shuffle(self.out1_rgb(feat))
        rgb = self.bilinear_rggb_to_rgb(denoised_rggb) + rgb_residual
        return denoised_rggb, rgb

    @classmethod
    def from_pretrained(cls, ckp_path):
        """Load a plain Network() denoise-only checkpoint (e.g. models/torch_pretrained.ckp)
        into everything but the new demosaic head, which stays at its zero-init. Asserts
        the checkpoint/model key mismatch is exactly the new head's 2 parameters, rather
        than silently tolerating e.g. a typo'd module name leaving some other layer
        un-loaded (same "verify, don't assume" spirit as train_qat_pytorch.py's
        extract_float_state_dict)."""
        net = cls()
        state_dict = torch.load(str(ckp_path), map_location='cpu')
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        new_head_keys = {'out1_rgb.conv.weight', 'out1_rgb.conv.bias'}
        if unexpected:
            raise RuntimeError(f'unexpected keys in {ckp_path}: {unexpected}')
        if set(missing) != new_head_keys:
            raise RuntimeError(
                f'expected only the new demosaic head {new_head_keys} to be missing from '
                f'{ckp_path}, got {set(missing)}'
            )
        return net


if __name__ == "__main__":
    net = Network()
    # img = mge.tensor(np.random.randn(1, 4, 64, 64).astype(np.float32))
    img = torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    out = net(img)

    import IPython; IPython.embed()

# vim: ts=4 sw=4 sts=4 expandtab
