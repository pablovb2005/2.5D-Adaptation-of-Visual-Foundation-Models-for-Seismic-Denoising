"""Per-batch metric functions.

Predictions and clean targets are shaped [B, 1, H, W]. Noisy inputs may have
one or more channels; residual metrics use the central noisy channel because
the model predicts the clean central slice.
"""

import torch
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)


_ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
_psnr = PeakSignalNoiseRatio(data_range=1.0)


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.amin(dim=(-2, -1), keepdim=True), x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn).clamp(min=1e-8)


def _central_channel(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected tensor shaped [B, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x
    c = x.shape[1] // 2
    return x[:, c:c + 1]


def compute_ms_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MS-SSIM between predicted clean and ground-truth clean (higher = better)."""
    return _ms_ssim(_to_unit(pred).cpu(), _to_unit(target).cpu()).item()


def compute_ms_ssim_r(denoised: torch.Tensor, noisy: torch.Tensor) -> float:
    """MS-SSIM of the residual (noisy − denoised) vs the noisy input.

    Measures signal leakage into the noise estimate. Lower is better.
    Does not require a clean reference, so usable on field data.
    """
    noisy_center = _central_channel(noisy)
    residual = noisy_center - denoised
    return _ms_ssim(_to_unit(residual).cpu(), _to_unit(noisy_center).cpu()).item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return _psnr(_to_unit(pred).cpu(), _to_unit(target).cpu()).item()


def compute_all(
    pred: torch.Tensor,
    target: torch.Tensor,
    noisy: torch.Tensor,
) -> dict[str, float]:
    return {
        "ms_ssim": compute_ms_ssim(pred, target),
        "ms_ssim_r": compute_ms_ssim_r(pred, noisy),
        "mse": compute_mse(pred, target),
        "psnr": compute_psnr(pred, target),
    }
