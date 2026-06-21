import torch
import torch.nn as nn


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PaperDecoder(nn.Module):
    """Lightweight decoder aligned with Jiahua's DINOv3-UNet notebook.

    Architecture (for 224×224 input with 16-px patch size):
      [B, hidden_dim, 14, 14]
      → up_block(hidden_dim, 192) → [B, 192, 28, 28]
      → up_block(192, 96)         → [B,  96, 56, 56]
      → up_block(96,  48)         → [B,  48, 112, 112]
      → up_block(48,  24)         → [B,  24, 224, 224]
      → Conv2d(24, 1, 1)          → [B,   1, 224, 224]

    No skip connections: all information must pass through the bottleneck
    so that the output is derived from clean semantic features only.
    """

    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        self.blocks = nn.Sequential(
            _up_block(hidden_dim, 192),
            _up_block(192, 96),
            _up_block(96, 48),
            _up_block(48, 24),
        )
        self.head = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.head(x)


if __name__ == "__main__":
    dec = PaperDecoder(hidden_dim=384)
    z = torch.randn(2, 384, 14, 14)
    out = dec(z)
    print(f"Input:  {tuple(z.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == (2, 1, 224, 224)
    params = sum(p.numel() for p in dec.parameters())
    print(f"Decoder params: {params:,}")
