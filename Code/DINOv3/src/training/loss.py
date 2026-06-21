import torch
import torch.nn as nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class DenoisingLoss(nn.Module):
    """Combined MSE + MS-SSIM loss (paper Eq. 6).

    L = (1 - lam) * MSE + lam * (1 - MS-SSIM)

    lam=0.5 assigns equal weight to amplitude fidelity and structural preservation.
    """

    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        # MS-SSIM expects inputs in [0, 1]; our data is Z-scored so we shift into
        # a comparable range by min-max normalising across the batch for the metric.
        pred_n = _minmax(pred)
        target_n = _minmax(target)
        ms_ssim_val = self.ms_ssim(pred_n, target_n)
        return (1.0 - self.lam) * mse + self.lam * (1.0 - ms_ssim_val)


def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalise to [0, 1] so MS-SSIM receives valid input range.

    Z-scored seismic patches have arbitrary sign and scale; this brings each image
    into [0, 1] independently without altering inter-image relative magnitudes.
    """
    mn, mx = x.amin(dim=(-2, -1), keepdim=True), x.amax(dim=(-2, -1), keepdim=True)
    denom = (mx - mn).clamp(min=1e-8)
    return (x - mn) / denom


if __name__ == "__main__":
    loss_fn = DenoisingLoss()
    a = torch.randn(4, 1, 224, 224)
    print("Loss (random):", loss_fn(a, torch.randn_like(a)).item())
    print("Loss (identical):", loss_fn(a, a).item())
