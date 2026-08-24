#!/usr/bin/env python3
import cv2
import numpy as np


def round_half_up(x):
    """Round to the nearest integer, ties rounding away from zero (traditional
    "四舍五入" / round-half-up) -- unlike numpy.round, which uses banker's
    rounding (round half to even) and so pulls exact .5 values down to the
    nearest even integer roughly half the time. That's a real bias for
    fixed-point quantization here: pixel values land on an exact .5 boundary
    often enough (e.g. whenever the upstream float32 math produces a value a
    clean fraction of 1/65536 etc.) that banker's rounding measurably skews
    quantized output versus true rounding.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


class RawUtils:

    # per-pattern flip needed to bring a 2x2 tile to the canonical RGGB
    # layout (top-left=R, top-right=G, bottom-left=G, bottom-right=B) that
    # bayer2rggb/rggb2bayer/bayer2rgb assume. Each entry is (flip_rows, flip_cols).
    # The mapping is an involution: applying it twice with the same pattern
    # is the identity, so the same call also converts canonical -> original.
    _PATTERN_FLIPS = {
        'RGGB': (False, False),
        'BGGR': (True, True),
        'GRBG': (False, True),
        'GBRG': (True, False),
    }
    BAYER_PATTERNS = tuple(_PATTERN_FLIPS)

    @classmethod
    def to_canonical_rggb(cls, *bayers, pattern):
        if pattern not in cls._PATTERN_FLIPS:
            raise ValueError(f'unsupported bayer_pattern: {pattern!r}, expected one of {cls.BAYER_PATTERNS}')

        flip_rows, flip_cols = cls._PATTERN_FLIPS[pattern]
        res = []
        for bayer in bayers:
            if flip_rows:
                bayer = bayer[::-1, :]
            if flip_cols:
                bayer = bayer[:, ::-1]
            res.append(bayer)

        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def bggr2rggb(cls, *bayers):
        res = []
        for bayer in bayers:
            res.append(bayer[::-1, ::-1])
        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def rggb2bggr(cls, *bayers):
        return cls.bggr2rggb(*bayers)

    @classmethod
    def bayer2rggb(cls, *bayers):
        res = []
        for bayer in bayers:
            H, W = bayer.shape
            res.append(
                bayer.reshape(H//2, 2, W//2, 2)
                .transpose(0, 2, 1, 3)
                .reshape(H//2, W//2, 4)
            )
        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def rggb2bayer(cls, *rggbs):
        res = []
        for rggb in rggbs:
            H, W, _ = rggb.shape
            res.append(
                rggb.reshape(H, W, 2, 2)
                .transpose(0, 2, 1, 3)
                .reshape(H*2, W*2)
            )

        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def bayer2rgb(cls, *bayer_01s, wb_gain, CCM, gamma=2.2):

        wb_gain = np.array(wb_gain)[[0, 1, 1, 2]]
        res = []
        for bayer_01 in bayer_01s:
            bayer = cls.rggb2bayer(
                (cls.bayer2rggb(bayer_01) * wb_gain).clip(0, 1)
            ).astype(np.float32)
            # quantize/dequantize this fixed 16-bit intermediate the same
            # shift-friendly way as everywhere else: multiply/divide by 2**16,
            # not by the true max value 2**16 - 1 (still the clip bound, since
            # that's the largest value a uint16 container can actually hold)
            bayer = round_half_up(np.ascontiguousarray(bayer) * 65536).clip(0, 65535).astype(np.uint16)
            rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB_EA).astype(np.float32) / 65536
            rgb = rgb.dot(np.array(CCM).T).clip(0, 1)
            rgb = rgb ** (1/gamma)
            res.append(rgb.astype(np.float32))

        if len(res) == 1:
            return res[0]
        return res


# vim: ts=4 sw=4 sts=4 expandtab
