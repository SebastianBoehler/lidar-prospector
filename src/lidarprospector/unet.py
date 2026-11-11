"""Dense barrow segmenter: a small U-Net whose encoder is the self-supervised terrain encoder's
convolutional body, so it can start from the pretrained weights or from scratch."""

import torch
import torch.nn.functional as F
from torch import nn

from .embed import Encoder, _block

TAPS = (0, 2, 4, 5)  # body blocks whose outputs feed the decoder: full, 1/2, 1/4, 1/8 resolution


class UNet(nn.Module):
    def __init__(self, ssl_state: dict | None = None):
        super().__init__()
        enc = Encoder()
        if ssl_state is not None:
            enc.load_state_dict(ssl_state)
        self.body = enc.body
        self.d3 = nn.Sequential(_block(192 + 128, 96, 1), _block(96, 96, 1))
        self.d2 = nn.Sequential(_block(96 + 64, 64, 1), _block(64, 64, 1))
        self.d1 = nn.Sequential(_block(64 + 32, 32, 1), _block(32, 32, 1))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        taps = []
        for i, block in enumerate(self.body):
            x = block(x)
            if i in TAPS:
                taps.append(x)
        f1, f2, f4, f8 = taps
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d = self.d3(torch.cat([up(f8, f4), f4], 1))
        d = self.d2(torch.cat([up(d, f2), f2], 1))
        d = self.d1(torch.cat([up(d, f1), f1], 1))
        return self.head(d)[:, 0]  # logits, (B, H, W)
