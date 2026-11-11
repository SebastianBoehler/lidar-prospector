"""Dense barrow segmenter: a small U-Net. The encoder body has the same layout as the self-supervised
terrain encoder, so it can start from the pretrained weights (default widths, 2 input channels) or
from scratch (any widths / input channels)."""

import torch
import torch.nn.functional as F
from torch import nn

from .embed import _block

TAPS = (0, 2, 4, 5)  # body blocks whose outputs feed the decoder: full, 1/2, 1/4, 1/8 resolution
STRIDES = (1, 2, 1, 2, 1, 2)
DEFAULT_WIDTHS = (32, 64, 64, 128, 128, 192)


def scaled_widths(factor: float) -> tuple[int, ...]:
    return tuple(int(round(w * factor / 8)) * 8 for w in DEFAULT_WIDTHS)


class UNet(nn.Module):
    def __init__(
        self, ssl_state: dict | None = None, in_ch: int = 2, widths: tuple[int, ...] = DEFAULT_WIDTHS
    ):
        super().__init__()
        w = widths
        chans = (in_ch, *w)
        self.body = nn.Sequential(*[_block(chans[i], chans[i + 1], STRIDES[i]) for i in range(6)])
        if ssl_state is not None:
            if in_ch != 2 or widths != DEFAULT_WIDTHS:
                raise ValueError("SSL weights only fit the default encoder (2 channels, default widths)")
            body = {k[len("body.") :]: v for k, v in ssl_state.items() if k.startswith("body.")}
            self.body.load_state_dict(body)
        f1, f2, f4, f8 = w[0], w[2], w[4], w[5]
        self.d3 = nn.Sequential(_block(f8 + f4, f4 // 4 * 3, 1), _block(f4 // 4 * 3, f4 // 4 * 3, 1))
        self.d2 = nn.Sequential(_block(f4 // 4 * 3 + f2, f2, 1), _block(f2, f2, 1))
        self.d1 = nn.Sequential(_block(f2 + f1, f1, 1), _block(f1, f1, 1))
        self.head = nn.Conv2d(f1, 1, 1)

    def forward(self, x):
        taps = []
        for i, block in enumerate(self.body):
            x = block(x)
            if i in TAPS:
                taps.append(x)
        f1, f2, f4, f8 = taps

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        d = self.d3(torch.cat([up(f8, f4), f4], 1))
        d = self.d2(torch.cat([up(d, f2), f2], 1))
        d = self.d1(torch.cat([up(d, f1), f1], 1))
        return self.head(d)[:, 0]  # logits, (B, H, W)
